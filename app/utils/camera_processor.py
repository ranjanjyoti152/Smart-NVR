import cv2
import numpy as np
import torch
import time
import os
import json
import threading
import queue
import logging
from datetime import datetime, timedelta
import uuid
from shapely.geometry import Point, Polygon
import requests

logger = logging.getLogger(__name__)

class CameraProcessor:
    """Process RTSP camera streams with YOLOv5 object detection"""
    
    def __init__(self, camera, model_path=None, confidence_threshold=None):
        """Initialize camera processor
        
        Args:
            camera: Camera object from database
            model_path: Path to YOLOv5 model file (if None, use camera's model)
            confidence_threshold: Detection confidence threshold (if None, use camera's threshold)
        """
        self.camera = camera
        self.model_path = model_path or self._get_model_path()
        self.confidence_threshold = confidence_threshold or camera.confidence_threshold or 0.45
        self.cap = None
        self.model = None
        self.running = False
        self.recording = False
        self.thread = None
        self.recording_thread = None
        self.detection_thread = None
        self.frame_queue = queue.Queue(maxsize=10)  # Queue for frames to process
        self.recording_queue = queue.Queue(maxsize=30)  # Queue for frames to record
        self.last_frame = None
        self.fps = 0
        self.last_detection_time = None
        self.current_video_path = None
        self.video_writer = None
        self.video_start_time = None
        self.detection_regions = self._load_detection_regions()
        self.current_detections = []  # Store current detections for API access
        self.detection_lock = threading.Lock()  # Lock for thread-safe detection updates
        
    def _get_model_path(self):
        """Get path to YOLOv5 model file from camera config or use default"""
        from app import db
        from app.models.ai_model import AIModel
        
        if self.camera.model_id:
            model = AIModel.query.get(self.camera.model_id)
            if model and os.path.exists(model.file_path):
                return model.file_path
                
        # Use default model if camera model not found
        default_model = AIModel.query.filter_by(is_default=True).first()
        if default_model and os.path.exists(default_model.file_path):
            return default_model.file_path
            
        # Fall back to YOLOv5s
        return os.path.join('models', 'yolov5s.pt')
        
    def _load_detection_regions(self):
        """Load detection regions (ROIs) for this camera"""
        from app.models.roi import ROI
        
        regions = []
        rois = ROI.query.filter_by(camera_id=self.camera.id).all()
        
        for roi in rois:
            if not roi.is_active:
                continue
                
            try:
                # Parse ROI coordinates and allowed classes
                import json
                coords = json.loads(roi.coordinates) if isinstance(roi.coordinates, str) else roi.coordinates
                classes = json.loads(roi.detection_classes) if roi.detection_classes and isinstance(roi.detection_classes, str) else None
                
                # Log the loaded ROI for debugging
                logger.info(f"Loading ROI {roi.id}: {roi.name}, classes: {classes}, email_notifications: {roi.email_notifications}")
                logger.info(f"ROI coordinates: {coords}")
                
                if len(coords) >= 4:  # Need at least 3 points for a polygon
                    # Create polygon from coordinates - note these are normalized (0-1 range)
                    # We store them as normalized but need to keep track of this when checking points
                    regions.append({
                        'id': roi.id,
                        'name': roi.name,
                        'normalized_coords': True,  # Flag to indicate coordinates are normalized
                        'polygon': Polygon(coords),
                        'classes': classes,
                        'email_notifications': roi.email_notifications
                    })
                    logger.info(f"Successfully created ROI polygon for {roi.name}")
            except Exception as e:
                logger.error(f"Error loading ROI {roi.id}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
        logger.info(f"Loaded {len(regions)} ROIs for camera {self.camera.id}")
        return regions

    def reload_rois(self):
        """Reload detection regions (ROIs) from the database"""
        logger.info(f"Reloading ROIs for camera {self.camera.id}")
        self.detection_regions = self._load_detection_regions()
        return len(self.detection_regions)
        
    def start(self):
        """Start processing camera stream"""
        if self.running:
            return False
            
        # Initialize video capture
        rtsp_url = self.camera.rtsp_url
        if self.camera.username and self.camera.password:
            # Insert credentials into RTSP URL if needed
            if '://' in rtsp_url:
                protocol, rest = rtsp_url.split('://', 1)
                rtsp_url = f"{protocol}://{self.camera.username}:{self.camera.password}@{rest}"
        
        # Set OpenCV backend to FFMPEG with specific parameters to avoid threading issues
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;10485760|stimeout;1000000"
        
        # Open video capture with optimized parameters
        if rtsp_url.startswith('rtsp://'):
            # For RTSP streams, use these specific parameters to avoid the async_lock crash
            self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            # Important: Disable multi-threading in FFmpeg which causes the crash
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))  # Use MJPEG
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Small buffer 
        else:
            # For other sources (like local files or HTTP streams)
            self.cap = cv2.VideoCapture(rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        # Check if stream opened successfully
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera stream: {rtsp_url}")
            return False
            
        logger.info(f"Successfully opened camera stream: {rtsp_url}")
        
        # Initialize YOLOv5 model
        try:
            logger.info(f"Loading YOLOv5 model from {self.model_path}")
            
            # Try loading model directly if it's a local file
            if os.path.exists(self.model_path) and self.model_path.endswith('.pt'):
                try:
                    # Use local model file with direct YOLOv5 loading
                    # Check if we're in the yolov5 models directory or root directory
                    if os.path.basename(self.model_path) in ["yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]:
                        # Standard YOLOv5 model - use the appropriate size
                        model_size = os.path.basename(self.model_path).replace('yolov5', '').replace('.pt', '')
                        logger.info(f"Loading standard YOLOv5 model size: {model_size}")
                        self.model = torch.hub.load('ultralytics/yolov5', f'yolov5{model_size}', 
                                                  pretrained=True, 
                                                  trust_repo=True)
                    else:
                        # Custom model - try direct loading
                        logger.info(f"Loading custom model from {self.model_path}")
                        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                                  path=self.model_path,
                                                  trust_repo=True)
                except Exception as e:
                    logger.warning(f"Direct YOLOv5 loading failed: {str(e)}")
                    
                    # Get the basename of the model file for simpler handling
                    model_basename = os.path.basename(self.model_path)
                    
                    # Check if it's a standard YOLOv5 model by name
                    if model_basename in ["yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]:
                        model_size = model_basename.replace('yolov5', '').replace('.pt', '')
                        logger.info(f"Falling back to YOLOv5 {model_size} from PyTorch Hub")
                        self.model = torch.hub.load('ultralytics/yolov5', f'yolov5{model_size}', 
                                                  pretrained=True, 
                                                  trust_repo=True,
                                                  force_reload=True)
                    else:
                        # Last resort - use default YOLOv5s
                        logger.warning(f"Falling back to default YOLOv5s model")
                        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 
                                                  pretrained=True, 
                                                  trust_repo=True)
            else:
                # Model path doesn't exist or isn't a .pt file - use default YOLOv5s
                logger.warning(f"Model path {self.model_path} not valid, using default YOLOv5s")
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 
                                           pretrained=True, 
                                           trust_repo=True)
                
            # Set confidence threshold
            self.model.conf = self.confidence_threshold
            
            # Use GPU if available
            if torch.cuda.is_available():
                self.model.cuda()
                logger.info("Using CUDA for inference")
            else:
                logger.info("Using CPU for inference")
            
            logger.info(f"Successfully loaded YOLOv5 model")
        except Exception as e:
            logger.error(f"Failed to load YOLOv5 model: {str(e)}")
            self.cap.release()
            return False
            
        # Start processing thread
        self.running = True
        self.thread = threading.Thread(target=self._process_frames)
        self.thread.daemon = True
        self.thread.start()
        
        # Give the main processing thread a moment to initialize
        time.sleep(1.0)
        
        # Start recording thread if enabled
        if self.camera.recording_enabled:
            self.recording = True
            self.recording_thread = threading.Thread(target=self._record_frames)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
        # Start detection thread if enabled
        if self.camera.detection_enabled:
            self.detection_thread = threading.Thread(target=self._detect_objects)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
        logger.info(f"Started processing camera: {self.camera.name}")
        return True
        
    def stop(self):
        """Stop processing camera stream"""
        self.running = False
        self.recording = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=1.0)
            
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
            
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
        logger.info(f"Stopped camera: {self.camera.name}")
        return True
        
    def get_frame(self):
        """Get the latest processed frame with detection boxes"""
        frame = self.last_frame.copy() if self.last_frame is not None else None
        
        if frame is not None:
            # Draw ROIs on frame
            for region in self.detection_regions:
                if not hasattr(region['polygon'], 'exterior'):
                    continue
                    
                # Convert polygon coordinates to pixel points
                height, width = frame.shape[:2]
                points = []
                for x, y in region['polygon'].exterior.coords:
                    # Convert from normalized to pixel coordinates if needed
                    if 0 <= x <= 1 and 0 <= y <= 1:  # Normalized coordinates
                        px = int(x * width)
                        py = int(y * height)
                    else:  # Already in pixel coordinates
                        px = int(x)
                        py = int(y)
                    points.append([px, py])
                
                # Convert points to numpy array
                if len(points) >= 3:
                    points = np.array(points, dtype=np.int32)
                    # Draw polygon
                    cv2.polylines(frame, [points], True, (0, 0, 255), 2)
                    # Add ROI name near the polygon
                    centroid_x = np.mean(points[:, 0])
                    centroid_y = np.mean(points[:, 1])
                    cv2.putText(frame, region['name'], (int(centroid_x), int(centroid_y)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Always draw current detections on the returned frame
            with self.detection_lock:
                for detection in self.current_detections:
                    if all(k in detection for k in ['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']):
                        x1 = int(detection['bbox_x'])
                        y1 = int(detection['bbox_y'])
                        x2 = int(detection['bbox_x'] + detection['bbox_width'])
                        y2 = int(detection['bbox_y'] + detection['bbox_height'])
                        class_name = detection['class_name']
                        conf = detection['confidence']
                        
                        # Draw the bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name} {conf:.2f}"
                        # Draw label with background
                        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
        
    def get_latest_detections(self):
        """Get the latest detections"""
        with self.detection_lock:
            return self.current_detections.copy()
        
    def _process_frames(self):
        """Main processing loop for camera frames"""
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning(f"Failed to read frame from camera {self.camera.name}, reconnecting...")
                    time.sleep(2)
                    
                    # Try to reconnect
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.camera.rtsp_url)
                    continue
                
                # Update FPS calculation every 30 frames
                frame_count += 1
                if frame_count >= 30:
                    end_time = time.time()
                    self.fps = frame_count / (end_time - start_time)
                    frame_count = 0
                    start_time = time.time()
                
                # Add timestamp overlay
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Add camera name overlay
                cv2.putText(frame, self.camera.name, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Store current frame
                self.last_frame = frame.copy()
                
                # Add frame to queues for processing and recording
                try:
                    if not self.frame_queue.full() and self.camera.detection_enabled:
                        self.frame_queue.put(frame, block=False)
                except queue.Full:
                    pass
                    
                try:
                    if not self.recording_queue.full() and self.recording:
                        self.recording_queue.put(frame, block=False)
                except queue.Full:
                    pass
                    
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                time.sleep(1)
                
    def _detect_objects(self):
        """Process frames for object detection"""
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1.0)
                
                # Get image dimensions for coordinate normalization
                height, width = frame.shape[:2]
                
                # Perform inference with YOLOv5
                results = self.model(frame)
                detections = results.pandas().xyxy[0]
                
                detected_objects = []
                
                # Process detection results
                for _, detection in detections.iterrows():
                    x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
                    conf = float(detection['confidence'])
                    class_id = int(detection['class'])
                    class_name = detection['name']
                    
                    # Skip if confidence is below threshold
                    if conf < self.confidence_threshold:
                        continue
                    
                    # Calculate center point of the detection
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Create point for ROI checking
                    pixel_center = Point(center_x, center_y)
                    # For normalized coordinates, we need to create a normalized point
                    norm_center = Point(center_x / width, center_y / height)
                    
                    # Initialize variables for ROI tracking
                    in_any_roi = False
                    matched_roi_id = None
                    matched_roi = None
                    
                    # First pass: Check if object is inside any ROI
                    for region in self.detection_regions:
                        # Choose the correct point based on whether coordinates are normalized
                        point_to_check = norm_center if region.get('normalized_coords', False) else pixel_center
                        
                        logger.debug(f"Checking if {class_name} at {point_to_check} is in ROI {region['name']}")
                        
                        if region['polygon'].contains(point_to_check):
                            in_any_roi = True
                            logger.info(f"Object {class_name} inside ROI {region['name']}")
                            
                            # Check if this object class is allowed in this ROI
                            if region['classes'] and len(region['classes']) > 0:
                                logger.debug(f"ROI {region['name']} has class restrictions: {region['classes']}")
                                if class_id in region['classes']:
                                    matched_roi_id = region['id']
                                    matched_roi = region
                                    logger.info(f"Object {class_name} (class_id={class_id}) matched ROI {region['name']} (id={region['id']})")
                                    break
                            else:
                                # No class restrictions, match any class
                                matched_roi_id = region['id']
                                matched_roi = region
                                logger.info(f"Object {class_name} in ROI {region['name']} with no class restrictions")
                                break
                    
                    # Include all detections for now, marking those in ROIs appropriately
                    detected_obj = {
                        'camera_id': self.camera.id,
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox_x': x1,
                        'bbox_y': y1,
                        'bbox_width': x2 - x1,
                        'bbox_height': y2 - y1,
                        'roi_id': matched_roi_id  # This is the key field for email notifications
                    }
                    
                    # For debugging: log if this detection has an ROI association
                    if matched_roi_id is not None:
                        logger.info(f"Detection of {class_name} associated with ROI ID {matched_roi_id}")
                    
                    detected_objects.append(detected_obj)
                    
                    # Draw detection rectangle on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # If objects detected, save frame and send notification
                if detected_objects:
                    self.last_detection_time = datetime.now()
                    
                    # Save detection image
                    detection_time = self.last_detection_time.strftime("%Y%m%d_%H%M%S")
                    image_dir = os.path.join('storage', 'recordings', 'images', str(self.camera.id))
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = os.path.join(image_dir, f"{detection_time}_{uuid.uuid4().hex[:8]}.jpg")
                    cv2.imwrite(image_path, frame)
                    
                    # Set video path if we're recording
                    video_path = self.current_video_path if self.recording else None
                    
                    # Add image path to detections
                    for obj in detected_objects:
                        obj['image_path'] = image_path
                        obj['video_path'] = video_path
                        obj['timestamp'] = self.last_detection_time
                    
                    # Update current detections for API
                    with self.detection_lock:
                        self.current_detections = detected_objects
                    
                    # Report detection to API
                    self._report_detection(detected_objects)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in object detection: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                time.sleep(1)
    
    def _record_frames(self):
        """Record video from camera frames"""
        while self.running and self.recording:
            try:
                # Check if we need to create a new video file
                current_time = datetime.now()
                
                # Create new file every hour or if no current file
                if (not self.video_writer or not self.video_start_time or 
                        (current_time - self.video_start_time).total_seconds() > 3600):
                    self._rotate_video_file(current_time)
                
                # Get frame from queue
                frame = self.recording_queue.get(timeout=1.0)
                
                # Write frame to video
                if self.video_writer:
                    self.video_writer.write(frame)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error recording video: {str(e)}")
                time.sleep(1)
    
    def _rotate_video_file(self, current_time=None):
        """Create a new video file for recording"""
        if not current_time:
            current_time = datetime.now()
            
        # Close current writer if exists
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        # Create recordings directory for this camera
        video_dir = os.path.join('storage', 'recordings', 'videos', str(self.camera.id))
        os.makedirs(video_dir, exist_ok=True)
        
        # Create video filename with timestamp
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(video_dir, f"{timestamp}.mp4")
        
        # Get frame dimensions from current frame
        if self.last_frame is not None:
            height, width = self.last_frame.shape[:2]
        else:
            # Default dimensions if no frame available
            ret, frame = self.cap.read()
            if ret:
                height, width = frame.shape[:2]
                self.last_frame = frame
            else:
                height, width = 720, 1280
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
        fps = 20.0  # Fixed recording FPS
        self.video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Update video information
        self.current_video_path = video_path
        self.video_start_time = current_time
        
        logger.info(f"Created new recording file: {video_path}")
        return video_path
    
    def _report_detection(self, detections):
        """Report detection to the API for database storage and notifications"""
        try:
            from app import app
            
            payload = {
                'camera_id': self.camera.id,
                'detections': detections
            }
            
            # Use direct database access if we're running in the same process
            # Otherwise make API call to detection endpoint
            if app:
                # We're in the Flask app context
                from app.routes.api_routes import report_detection
                
                # Create mock request with JSON payload
                class MockRequest:
                    def get_json(self):
                        return payload
                    
                    @property
                    def headers(self):
                        return {'X-API-Key': app.config.get('API_KEY', '')}
                
                # Call the API function directly within the app context
                with app.app_context():
                    report_detection(MockRequest())
            else:
                # Make external API call
                api_url = f"http://localhost:8000/api/detections"
                headers = {'X-API-Key': 'YOUR_API_KEY'}  # This should be properly configured
                requests.post(api_url, json=payload, headers=headers, timeout=2.0)
            
            logger.info(f"Reported {len(detections)} detections for camera {self.camera.id}")
            
        except Exception as e:
            logger.error(f"Error reporting detection: {str(e)}")

# Camera Manager to handle multiple camera instances
class CameraManager:
    """Manage multiple camera processors"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = CameraManager()
        return cls._instance
    
    def __init__(self):
        """Initialize camera manager"""
        self.cameras = {}  # Map camera_id to CameraProcessor
        self.lock = threading.Lock()
    
    def get_camera_processor(self, camera_id):
        """Get camera processor by ID"""
        return self.cameras.get(camera_id)
    
    def start_camera(self, camera):
        """Start processing a camera"""
        with self.lock:
            # Stop existing processor if any
            if camera.id in self.cameras:
                self.stop_camera(camera.id)
            
            # Create new processor
            processor = CameraProcessor(camera)
            if processor.start():
                self.cameras[camera.id] = processor
                return True
            return False
    
    def stop_camera(self, camera_id):
        """Stop processing a camera"""
        with self.lock:
            processor = self.cameras.get(camera_id)
            if processor:
                processor.stop()
                del self.cameras[camera_id]
                return True
            return False
    
    def start_all_cameras(self):
        """Start all enabled cameras from database"""
        from app.models import Camera
        
        cameras = Camera.query.filter_by(is_active=True).all()
        started = 0
        
        for camera in cameras:
            if self.start_camera(camera):
                started += 1
        
        return started
    
    def stop_all_cameras(self):
        """Stop all running cameras"""
        camera_ids = list(self.cameras.keys())
        stopped = 0
        
        for camera_id in camera_ids:
            if self.stop_camera(camera_id):
                stopped += 1
        
        return stopped

    def reload_rois(self, camera_id):
        """Reload ROIs for a specific camera
        
        Args:
            camera_id: ID of the camera to reload ROIs for
            
        Returns:
            int: Number of ROIs loaded or -1 if camera not found
        """
        with self.lock:
            processor = self.cameras.get(camera_id)
            if processor:
                return processor.reload_rois()
            return -1