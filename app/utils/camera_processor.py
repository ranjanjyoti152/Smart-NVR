import cv2
import numpy as np
import torch
import time
import os
import json
import threading
import queue
import logging
import traceback
from datetime import datetime, timedelta
import uuid
from shapely.geometry import Point, Polygon
import requests
import random # Import random for generating colors

# Explicitly import torchvision and its transforms to resolve circular import
import torchvision
import torchvision.transforms as transforms

from ultralytics import YOLO # Import YOLO from ultralytics
import re  # Import re for regex pattern matching
from bson.objectid import ObjectId  # Add missing import for MongoDB ObjectId
from app.utils.face_detection import build_face_detection_engine
from app.utils.face_recognition import get_face_recognition_engine
from app.models.face_profile import FaceProfile

logger = logging.getLogger(__name__)

# Define a color map for object classes (you can customize this)
# Using BGR format for OpenCV
CLASS_COLORS = {}

def get_color_for_class(class_name):
    """Get a consistent color for a class name."""
    if class_name not in CLASS_COLORS:
        # Generate a random color if class not in map
        CLASS_COLORS[class_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return CLASS_COLORS[class_name]

class CameraProcessor:
    """Process RTSP camera streams with YOLOv5 object detection"""

    def __init__(self, camera, model_path=None, confidence_threshold=None, device: str | None = None):
        """Initialize camera processor

        Args:
            camera: Camera object from database
            model_path: Path to YOLOv5 model file (if None, use camera's model)
            confidence_threshold: Detection confidence threshold (if None, use camera's threshold)
            device: Optional explicit torch device string (e.g. 'cuda:0'); if None, auto-select
        """
        self.camera = camera
        self.model_path = model_path or self._get_model_path()
        self.confidence_threshold = confidence_threshold or camera.confidence_threshold or 0.45
        self.device = device  # Preferred compute device for this camera (e.g., 'cuda:0')
        self.cap = None
        self.model = None
        self.running = False
        self.recording = False
        self.thread = None
        self.recording_thread = None
        self.detection_thread = None
        self.frame_queue = queue.Queue(maxsize=2)  # Very small queue for minimal latency
        self.recording_queue = queue.Queue(maxsize=10)  # Reduced for better memory management
        self.last_frame = None # Stores the latest raw frame from camera
        self.last_processed_frame = None # Stores the latest frame processed by detection
        self.last_processed_detections = [] # Stores detections for the last_processed_frame
        self.fps = 0
        self.last_detection_time = None
        self.current_video_path = None
        self.video_writer = None
        self.video_start_time = None
        self.detection_regions = self._load_detection_regions()
        self.current_detections = []  # Store latest detections for API access (might differ slightly from last_processed_detections)
        self.detection_lock = threading.Lock()
        # Initialize metrics for performance monitoring
        self.metrics = {
            'frame_loss': 0.0,
            'latency': 0.0,
            'frames_processed': 0,
            'frames_dropped': 0,
            'last_update': time.time(),
            'processing_time': 0.0,
            'avg_processing_time': 0.0
        }
        self.frame_timestamps = {}  # Track frame processing times
        self.face_detection_engine = None
        self.face_detection_settings = {}
        self.face_detection_enabled = False
        self._face_detection_state_logged = False
        self._last_face_settings_refresh = time.time()
        self.face_recognition_enabled = False
        self.face_recognition_threshold = 0.62
        self._face_recognition_auto_create = True
        self._face_recognition_warned = False
        self._load_face_detection_settings()
        
    def is_running(self):
        """Check if the camera processor is running"""
        # Check if the thread is alive and if we've processed frames recently
        thread_alive = self.thread is not None and self.thread.is_alive()
        
        # If thread isn't alive, processor isn't running
        if not thread_alive:
            return False
            
        # Check if we have a valid frame
        has_frame = self.last_frame is not None
        
        return thread_alive and has_frame and self.running

    def _get_model_path(self):
        """Get path to AI model file from camera config or use default"""
        from app import db
        from app.models.ai_model import AIModel
        from bson.objectid import ObjectId

        if self.camera.model_id:
            # Use MongoDB style query instead of SQLAlchemy
            model_data = db.ai_models.find_one({'_id': ObjectId(self.camera.model_id)})
            if model_data and os.path.exists(model_data.get('file_path')):
                logger.info(f"Using model specified by camera: {model_data.get('file_path')}")
                return model_data.get('file_path')

        # Use default model if camera model not found or invalid
        # Use MongoDB style query instead of SQLAlchemy
        default_model_data = db.ai_models.find_one({'is_default': True})
        if default_model_data and os.path.exists(default_model_data.get('file_path')):
            logger.info(f"Using default model: {default_model_data.get('file_path')}")
            return default_model_data.get('file_path')

        # Fallback logic (e.g., to yolov5s or a known existing model)
        fallback_path = os.path.join('models', 'yolov5s.pt') # Default fallback
        if os.path.exists(fallback_path):
             logger.warning(f"Falling back to default yolov5s.pt model: {fallback_path}")
             return fallback_path
        else:
            # If even the fallback doesn't exist, try finding *any* .pt file
            models_dir = 'models'
            if os.path.exists(models_dir):
                pt_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
                if pt_files:
                    first_model = os.path.join(models_dir, pt_files[0])
                    logger.warning(f"No specific or default model found. Falling back to first available model: {first_model}")
                    return first_model

        logger.error("No suitable AI model file found.")
        # Consider raising an error or returning None if no model can be found
        return None # Or raise an exception


    def _check_roi_match(self, pixel_center, norm_center):
        """Check if a point is within any ROI and if the detection class is allowed.
        
        Args:
            pixel_center: Point in pixel coordinates
            norm_center: Point in normalized coordinates (0-1 range)
            
        Returns:
            The ID of the first matching ROI, or None if no match
        """
        matched_roi_id = None
        
        for region in self.detection_regions:
            # Determine which point to check based on how the ROI coordinates were stored
            point_to_check = norm_center if region.get('normalized_coords', False) else pixel_center
            
            # Ensure polygon is valid before checking containment
            if region['polygon'].is_valid and region['polygon'].contains(point_to_check):
                # Default to allowed if no specific classes are set for ROI
                matched_roi_id = region['id']
                break  # Stop checking ROIs once a match is found
        
        return matched_roi_id
        
    def _load_detection_regions(self):
        """Load detection regions (ROIs) for this camera"""
        from app import db
        from bson.objectid import ObjectId
        from app.models.roi import ROI

        regions = []
        # Load all ROIs for this camera (not just active ones)
        rois_data = db.regions_of_interest.find({'camera_id': str(self.camera.id)})

        for roi_data in rois_data:
            try:
                # Create ROI object to check if it should be currently active
                roi = ROI(roi_data)
                
                # Only include if currently active based on time schedule
                if not roi.is_currently_active():
                    continue
                
                # Parse ROI coordinates and allowed classes
                import json
                coords = json.loads(roi_data['coordinates']) if isinstance(roi_data['coordinates'], str) else roi_data['coordinates']
                classes = json.loads(roi_data['detection_classes']) if roi_data.get('detection_classes') and isinstance(roi_data['detection_classes'], str) else None

                # Log the loaded ROI for debugging
                logger.info(f"Loading ROI {roi_data['_id']}: {roi_data['name']}, type: {roi.roi_type}, classes: {classes}, email_notifications: {roi_data.get('email_notifications', False)}")
                logger.info(f"ROI coordinates: {coords}")

                if len(coords) >= 4:  # Need at least 3 points for a polygon
                    # Create polygon from coordinates - note these are normalized (0-1 range)
                    # We store them as normalized but need to keep track of this when checking points
                    regions.append({
                        'id': str(roi_data['_id']),
                        'name': roi_data['name'],
                        'normalized_coords': True,  # Flag to indicate coordinates are normalized
                        'polygon': Polygon(coords),
                        'classes': classes,
                        'email_notifications': roi_data.get('email_notifications', False),
                        'roi_type': roi.roi_type
                    })
                    logger.info(f"Successfully created ROI polygon for {roi_data['name']}")
            except Exception as e:
                logger.error(f"Error loading ROI {roi_data.get('_id')}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

        logger.info(f"Loaded {len(regions)} currently active ROIs for camera {self.camera.id}")
        
        # Store the last reload time to check for time-based ROI updates
        self.last_roi_reload = datetime.now()
        
        return regions

    def _load_face_detection_settings(self):
        """Load global face detection settings and initialise engine if required."""
        try:
            from app.routes.main_routes import get_detection_settings

            detection_settings = get_detection_settings() or {}
        except Exception as exc:
            logger.warning("Unable to load detection settings for face detection: %s", exc)
            detection_settings = {}

        previous_settings = getattr(self, 'face_detection_settings', None)
        self.face_detection_settings = detection_settings
        self.face_detection_enabled = detection_settings.get('enable_face_detection', False) and self.camera.detection_enabled
        self.face_recognition_enabled = (
            detection_settings.get('enable_face_recognition', False)
            and self.camera.detection_enabled
            and self.face_detection_enabled
        )
        self.face_recognition_threshold = float(detection_settings.get('face_recognition_threshold', 0.62) or 0.62)
        self._face_recognition_auto_create = bool(detection_settings.get('face_recognition_auto_create', True))
        self._last_face_settings_refresh = time.time()

        if not self.face_detection_enabled:
            if self.face_detection_engine is not None:
                logger.info("Disabling face detection for camera %s based on updated settings", self.camera.id)
            self.face_detection_engine = None
            self._face_detection_state_logged = False
            self.face_recognition_enabled = False
            return

        # Initialise or refresh engine using current settings
        settings_changed = previous_settings != detection_settings or self.face_detection_engine is None
        self._ensure_face_detection_engine(force=settings_changed)

    def _ensure_face_detection_engine(self, force: bool = False):
        """Initialise the UniFace engine if enabled."""

        if not self.face_detection_enabled:
            return None

        if self.face_detection_engine is not None and not force and getattr(self.face_detection_engine, "ready", False):
            return self.face_detection_engine

        try:
            engine = build_face_detection_engine(self.face_detection_settings)
            self.face_detection_engine = engine

            if getattr(engine, "ready", False):
                if not self._face_detection_state_logged:
                    logger.info("Face detection enabled for camera %s", self.camera.id)
                    self._face_detection_state_logged = True
            else:
                if not self._face_detection_state_logged:
                    err = getattr(engine, "init_error", None)
                    if err:
                        logger.warning("Face detection unavailable for camera %s: %s", self.camera.id, err)
                    else:
                        logger.warning("Face detection engine not ready for camera %s", self.camera.id)
                    self._face_detection_state_logged = True
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to initialise face detection for camera %s: %s", self.camera.id, exc, exc_info=True)
            self.face_detection_engine = None
            self._face_detection_state_logged = True

        return self.face_detection_engine

    def _ensure_face_recognition_engine(self):
        """Prepare the recognition engine when enabled."""

        if not self.face_recognition_enabled:
            return None

        engine = get_face_recognition_engine()
        if engine is None or not engine.ready:
            if not self._face_recognition_warned:
                msg = getattr(engine, "init_error", None) if engine is not None else "not initialised"
                logger.warning("Face recognition unavailable for camera %s: %s", self.camera.id, msg)
                self._face_recognition_warned = True
            return None

        if self._face_recognition_warned:
            logger.info("Face recognition ready for camera %s", self.camera.id)
            self._face_recognition_warned = False
        return engine

    def reload_rois(self):
        """Reload detection regions (ROIs) from the database"""
        logger.info(f"Reloading ROIs for camera {self.camera.id}")
        self.detection_regions = self._load_detection_regions()
        self._load_face_detection_settings()
        return len(self.detection_regions)
    
    def _should_reload_rois(self):
        """Check if ROIs should be reloaded due to time-based schedule changes"""
        if not hasattr(self, 'last_roi_reload'):
            return True
        
        # Check every minute for time-based ROI changes
        time_since_reload = datetime.now() - self.last_roi_reload
        if time_since_reload.total_seconds() >= 60:  # Check every minute
            return True
        
        return False

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
            # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G')) # Use MJPEG - May not be needed/supported everywhere
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3) # Small buffer
        else:
            # For other sources (like local files or HTTP streams)
            self.cap = cv2.VideoCapture(rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # Check if stream opened successfully
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera stream: {rtsp_url}")
            return False

        logger.info(f"Successfully opened camera stream: {rtsp_url}")

        # Initialize AI model (YOLOv5, v8, v9, v10)
        try:
            import torch  # Make sure torch is imported here before using it
            
            if not self.model_path or not os.path.exists(self.model_path):
                 logger.error(f"Model path is invalid or file does not exist: {self.model_path}. Cannot start processor.")
                 self.cap.release()
                 return False

            logger.info(f"Loading AI model from {self.model_path}")
            model_filename = os.path.basename(self.model_path).lower()

            # Determine if this is a YOLOv5 custom model or another model (v8, v9, v10, etc.)
            # Check for YOLOv5 pattern in filename or use special handling for custom models
            is_yolov5_custom = re.search(r'yolov5\w*custom|custom.*yolov5|best|last', model_filename.lower()) is not None
            
            if model_filename.endswith('.pt'):
                if is_yolov5_custom:
                    # Use YOLOv5 specific loading method for custom YOLOv5 models
                    logger.info(f"Attempting to load custom YOLOv5 model {model_filename} using torch.hub")
                    try:
                        import torch
                        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, trust_repo=True)
                        logger.info(f"Successfully loaded custom YOLOv5 model {model_filename} using torch.hub")
                    except Exception as e:
                        logger.error(f"Failed to load model with torch.hub: {str(e)}")
                        # Fall back to ultralytics if torch.hub fails
                        logger.info(f"Falling back to ultralytics loader for {model_filename}")
                        self.model = YOLO(self.model_path)
                else:
                    # Use ultralytics YOLO for v8, v9, v10 and other models
                    logger.info(f"Attempting to load {model_filename} using ultralytics YOLO.")
                    try:
                        self.model = YOLO(self.model_path) # Load model using ultralytics
                        logger.info(f"Successfully loaded model {model_filename} using ultralytics.")
                    except (ModuleNotFoundError, ImportError) as import_err:
                        # Handle legacy model formats (models.yolo, models.common, etc.)
                        err_msg = str(import_err).lower()
                        if 'models.yolo' in err_msg or 'models.common' in err_msg:
                            logger.warning(f"Legacy YOLOv5 model detected ({import_err}). Creating simple model wrapper.")
                            # Instead of trying to load model weights, create a dummy wrapper directly
                            # This avoids trying to use torch.load which will fail with the same error
                            
                            try:
                                # Create a more functional wrapper object with predict method
                                class LegacyModelWrapper:
                                    def __init__(self, model_path, conf):
                                        self.model_path = model_path
                                        self.conf = conf
                                        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                        # Default values for class metadata
                                        self.class_names = []
                                        self.num_classes = 80  # Default COCO classes
                                        logger.info(f"Created legacy model wrapper with {self.num_classes} default COCO classes")
                                    
                                    def predict(self, img, conf=None, **kwargs):
                                        confidence = conf if conf is not None else self.conf
                                        
                                        # Create results object similar to YOLO's output
                                        class DetectionResults:
                                            def __init__(self):
                                                self.boxes = []
                                                self.names = {}
                                        
                                        # Just return the wrapper with no detections
                                        # Legacy models won't do actual detection, but the system will run
                                        results = DetectionResults()
                                        results.names = {i: f"class{i}" for i in range(self.num_classes)}
                                        if hasattr(self, 'class_names') and self.class_names:
                                            for i, name in enumerate(self.class_names):
                                                results.names[i] = name
                                        
                                        logger.warning("Legacy model detection functionality is limited - returning empty results")
                                        return [results]
                                    
                                    def to(self, device):
                                        self.device = torch.device(device)
                                        return self
                                
                                # Create wrapper with basic functionality
                                self.model = LegacyModelWrapper(self.model_path, self.confidence_threshold)
                                logger.info("Legacy model wrapper created successfully. Detection capabilities will be limited.")
                                logger.warning(f"Created legacy model wrapper for {model_filename}. Detection may be limited.")
                            except Exception as e:
                                logger.error(f"Failed to create legacy model wrapper: {str(e)}")
                                logger.error(traceback.format_exc())
                                raise
                        else:
                            # If it's another type of import error, re-raise it
                            logger.error(f"Failed to load model due to import error: {import_err}")
                            raise
            else:
                 logger.error(f"Unsupported model file format: {model_filename}. Only .pt files are supported.")
                 raise ValueError(f"Unsupported model file format: {model_filename}")


            # Confidence threshold is handled differently in ultralytics YOLO
            # It's typically set during inference, not on the model object directly like yolov5 hub
            # self.model.conf = self.confidence_threshold # This won't work for ultralytics YOLO

            # Device selection: prefer explicitly assigned GPU if provided
            if self.device is not None:
                device = self.device
            else:
                # Fallback to a single auto-selected device
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            try:
                self.model.to(device)  # Ensure model is on the correct device
                self.device = str(device)
                logger.info(f"Camera {self.camera.id} → using device {str(device).upper()} for inference")
            except Exception as dev_err:
                # Fallback to CPU on any device placement error
                logger.warning(f"Failed to move model to device '{device}' for camera {self.camera.id}: {dev_err}. Falling back to CPU.")
                device = 'cpu'
                self.model.to(device)
                self.device = 'cpu'
                logger.info(f"Camera {self.camera.id} → using device CPU for inference")

            logger.info(f"Successfully initialized AI model")

            # Warm up face detection engine if enabled
            self._ensure_face_detection_engine()
        except Exception as e:
            logger.error(f"Failed to load AI model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            if self.cap:
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
            logger.info(f"Started recording thread for camera {self.camera.name}")
        else:
            self.recording = False
            self.recording_thread = None
            logger.info(f"Recording disabled for camera {self.camera.name}")

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
        """Get the latest frame *that has been processed for detections*, with boxes drawn."""
        frame_to_display = None
        detections_to_draw = []

        with self.detection_lock:
            if self.last_processed_frame is not None:
                # Work on copies to release the lock quickly
                frame_to_display = self.last_processed_frame.copy()
                detections_to_draw = self.last_processed_detections.copy()

        if frame_to_display is None:
            # If detection hasn't processed any frame yet, return the latest raw frame
            # or None if capture hasn't started.
            with self.detection_lock: # Re-acquire briefly if needed
                 if self.last_frame is not None:
                     frame_to_display = self.last_frame.copy()
                 else:
                     return None # No frame available at all

        # --- Drawing happens on the retrieved frame_to_display ---

        # Draw ROIs on frame
        height, width = frame_to_display.shape[:2]
        for region in self.detection_regions:
            if not hasattr(region['polygon'], 'exterior'):
                continue
            points = []
            for x, y in region['polygon'].exterior.coords:
                px = int(x * width) if 0 <= x <= 1 else int(x)
                py = int(y * height) if 0 <= y <= 1 else int(y)
                points.append([px, py])

            if len(points) >= 3:
                points_np = np.array(points, dtype=np.int32)
                roi_color = (0, 0, 255) # Red for ROIs
                cv2.polylines(frame_to_display, [points_np], True, roi_color, 2)
                centroid_x = np.mean(points_np[:, 0])
                centroid_y = np.mean(points_np[:, 1])
                cv2.putText(frame_to_display, region['name'], (int(centroid_x), int(centroid_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)

        # Draw the specific detections corresponding to this frame
        for detection in detections_to_draw:
            if all(k in detection for k in ['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']):
                x1 = int(detection['bbox_x'])
                y1 = int(detection['bbox_y'])
                x2 = int(detection['bbox_x'] + detection['bbox_width'])
                y2 = int(detection['bbox_y'] + detection['bbox_height'])
                class_name = detection['class_name']
                conf = detection['confidence']
                color = get_color_for_class(class_name)

                cv2.rectangle(frame_to_display, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf:.2f}"
                if class_name.lower() == 'face':
                    status = detection.get('face_status')
                    name = detection.get('face_recognition_name')
                    score = detection.get('face_recognition_score')
                    if status == 'known' and name:
                        if score is not None:
                            label = f"{name} ({score:.2f})"
                        else:
                            label = f"{name} {conf:.2f}"
                    elif status in ('unlabeled', 'unknown'):
                        label = f"Face (?) {conf:.2f}"
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = y1 - 5
                label_bg_y1 = y1 - text_size[1] - 5
                if label_bg_y1 < 0:
                    label_bg_y1 = y1 + 5
                    label_y = y1 + text_size[1] + 5

                cv2.rectangle(frame_to_display, (x1, label_bg_y1), (x1 + text_size[0], y1 if label_bg_y1 < y1 else label_y), color, -1)
                cv2.putText(frame_to_display, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) # Black text

        # Timestamp and camera name should already be on the frame from _process_frames
        # If not, they could be added here if needed.

        return frame_to_display

    def get_latest_detections(self):
        """Get the latest detections"""
        with self.detection_lock:
            return self.current_detections.copy()

    def get_device(self) -> str:
        """Return the compute device currently used by this camera processor."""
        return self.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def _process_frames(self):
        """Main processing loop for camera frames"""
        frame_count = 0
        start_time = time.time()

        while self.running:
            try:
                if time.time() - self._last_face_settings_refresh > 60:
                    self._last_face_settings_refresh = time.time()
                    self._load_face_detection_settings()

                # Check if ROIs need to be reloaded for time-based scheduling
                if self._should_reload_rois():
                    self.reload_rois()
                
                ret, frame = self.cap.read()

                if not ret:
                    logger.warning(f"Failed to read frame from camera {self.camera.name}, reconnecting...")
                    time.sleep(2)
                    self.cap.release()
                    # Re-initialize capture object (ensure correct parameters)
                    rtsp_url = self.camera.rtsp_url # Get URL again
                    if self.camera.username and self.camera.password:
                         if '://' in rtsp_url:
                            protocol, rest = rtsp_url.split('://', 1)
                            rtsp_url = f"{protocol}://{self.camera.username}:{self.camera.password}@{rest}"

                    if rtsp_url.startswith('rtsp://'):
                        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                    else:
                        self.cap = cv2.VideoCapture(rtsp_url)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

                    # Check if stream opened successfully
                    if not self.cap.isOpened():
                         logger.error(f"Failed to reconnect to camera stream: {rtsp_url}")
                         time.sleep(5) # Wait longer before next attempt
                    continue


                # --- Add overlays BEFORE putting frame in detection queue ---
                # Add timestamp overlay
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)

                # Add camera name overlay
                cv2.putText(frame, self.camera.name, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)
                # --- End overlays ---

                # Store the latest raw frame (with overlays)
                with self.detection_lock:
                    self.last_frame = frame.copy()

                # Add frame to queues for processing and recording
                # Use frame.copy() if putting the same object into multiple queues
                # or if modification happens downstream before consumption.
                # Here, detection thread will use its own copy later.
                frame_copy_for_queues = frame.copy()
                try:
                    if self.camera.detection_enabled:
                         # Drop oldest frame if queue is full to prioritize newer frames
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait() # Discard oldest
                            except queue.Empty:
                                pass # Should not happen if full, but handle anyway
                        self.frame_queue.put(frame_copy_for_queues, block=False)
                except queue.Full:
                     logger.warning(f"Detection frame queue full for camera {self.camera.name}. Check detection performance.")
                     pass # Should be handled by the check above, but keep for safety

                try:
                    if self.recording:
                        # Recording queue can block or drop if needed, depending on requirements
                        # Current behavior drops if full
                        if self.recording_queue.full():
                             try:
                                 self.recording_queue.get_nowait() # Discard oldest recording frame
                             except queue.Empty:
                                 pass
                        self.recording_queue.put(frame_copy_for_queues, block=False)
                except queue.Full:
                    logger.warning(f"Recording queue full for camera {self.camera.name}.")
                    pass

                # Update FPS calculation
                frame_count += 1
                if frame_count >= 30:
                    end_time = time.time()
                    elapsed = end_time - start_time
                    if elapsed > 0: # Avoid division by zero
                         self.fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()


            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                time.sleep(1)

    def _detect_objects(self):
        """Process frames for object detection using the loaded model"""
        while self.running:
            try:
                # Get frame from queue with timeout to avoid blocking forever
                try:
                    frame_to_process = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    # If queue is empty, briefly pause to yield CPU
                    time.sleep(0.01)
                    continue

                # Get image dimensions
                height, width = frame_to_process.shape[:2]

                # Perform inference using model (handling both YOLOv5 and ultralytics YOLO API)
                # YOLOv5 from torch.hub uses __call__, while ultralytics YOLO uses predict
                try:
                    if hasattr(self.model, 'predict'):
                        # Ultralytics YOLO API
                        results = self.model.predict(source=frame_to_process, conf=self.confidence_threshold, verbose=False)
                    else:
                        # YOLOv5 torch.hub API
                        results = self.model(frame_to_process, size=640, augment=False)
                except RuntimeError as rt_err:
                    # Graceful fallback on CUDA OOM: move model to CPU and retry once
                    msg = str(rt_err).lower()
                    if 'out of memory' in msg and ('cuda' in msg or 'cublas' in msg):
                        try:
                            logger.warning(f"CUDA OOM during inference on camera {self.camera.id}. Falling back to CPU for this camera.")
                            self.model.to('cpu')
                            self.device = 'cpu'
                            if hasattr(self.model, 'predict'):
                                results = self.model.predict(source=frame_to_process, conf=self.confidence_threshold, verbose=False)
                            else:
                                results = self.model(frame_to_process, size=640, augment=False)
                        except Exception as cpu_fallback_err:
                            logger.error(f"Inference failed after CPU fallback on camera {self.camera.id}: {cpu_fallback_err}")
                            self.frame_queue.task_done()
                            time.sleep(0.05)
                            continue
                    else:
                        # Re-raise non-OOM runtime errors
                        raise

                # Process results (ultralytics results object is different)
                detected_objects = []
                current_time = datetime.now()

                # Process detection results
                try:
                    # Handle different result formats (YOLOv5 vs ultralytics YOLO)
                    if results is not None:
                        if hasattr(results, 'boxes'):
                            # Direct Detections object from ultralytics YOLO
                            res = results
                        elif isinstance(results, list) and len(results) > 0:
                            # List of results from YOLOv5
                            res = results[0]
                        elif str(type(results)) == "<class 'models.common.Detections'>":
                            # YOLOv5 models.common.Detections format
                            res = results
                        else:
                            # Unknown format, skip processing
                            logger.warning(f"Unknown results format: {type(results)}")
                            self.frame_queue.task_done()
                            continue
                            
                        # Process boxes based on the format of the detection results
                        if hasattr(res, 'boxes'):
                            # Process Ultralytics YOLO format boxes
                            self._process_ultralytics_boxes(res.boxes, res.names, height, width, current_time, detected_objects)
                        elif hasattr(res, 'pred') and hasattr(res, 'names'):
                            # Process YOLOv5 format boxes
                            self._process_yolov5_boxes(res.pred[0] if len(res.pred) > 0 else [], res.names, height, width, current_time, detected_objects)
                        else:
                            logger.warning(f"Cannot extract detection boxes from results: {type(res)}")
                            self.frame_queue.task_done()
                            continue
                finally:
                    # Explicitly release reference to large result objects to help garbage collection
                    results = None
                    res = None

                # Optional UniFace face detection augmentation
                try:
                    face_detections = self._run_face_detection(frame_to_process, current_time, height, width)
                    if face_detections:
                        self._augment_face_recognition(frame_to_process, face_detections, current_time)
                        detected_objects.extend(face_detections)
                except Exception as exc:
                    logger.debug("Face detection failed for camera %s: %s", self.camera.id, exc)

                # Update shared state under lock
                with self.detection_lock:
                    # Store the frame that was just processed and its detections
                    self.last_processed_frame = frame_to_process
                    self.last_processed_detections = detected_objects
                    # Also update current_detections for the API
                    self.current_detections = detected_objects

                # Handle post-detection actions if there are detections
                if detected_objects:
                    self.last_detection_time = current_time
                    self._handle_detection_image_saving(frame_to_process, detected_objects, current_time)
                    self._report_detection(detected_objects)

                # Mark the frame queue task as done
                self.frame_queue.task_done()

            except Exception as e:
                logger.error(f"Error in object detection: {str(e)}")
                logger.error(traceback.format_exc())
                # Ensure task_done is called even on error if a frame was retrieved
                if 'frame_to_process' in locals():
                     try:
                         self.frame_queue.task_done()
                     except ValueError: # Can happen if task_done called multiple times
                         pass
                time.sleep(1) # Longer sleep on error

    def _record_frames(self):
        """Record video from camera frames"""
        # Get clip length from application settings
        try:
            from app.routes.main_routes import get_recording_settings
            recording_settings = get_recording_settings()
            clip_length = int(recording_settings.get('clip_length', 60))  # Default to 60 seconds if not set
            # Convert clip_length from seconds to seconds (already in seconds)
            clip_duration = clip_length  # seconds
            logger.info(f"Using clip length of {clip_duration} seconds for camera {self.camera.id}")
        except Exception as e:
            logger.warning(f"Could not load recording settings: {str(e)}, using default clip length of 60 seconds")
            clip_duration = 60  # Default to 60 seconds

        while self.running and self.recording:
            try:
                # Check if we need to create a new video file
                current_time = datetime.now()

                # Create new file based on configured clip length or if no current file
                if (not self.video_writer or not self.video_start_time or
                        (current_time - self.video_start_time).total_seconds() > clip_duration):
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
        import os  # Add import here to ensure it's available

        if not current_time:
            current_time = datetime.now()

        # Close current writer if exists
        if self.video_writer:
            self.video_writer.release()

            # Create database entry for the completed recording
            if self.current_video_path and self.video_start_time:
                try:
                    from app import db
                    from app.models.recording import Recording

                    # Calculate duration from start time to now
                    duration = (current_time - self.video_start_time).total_seconds()

                    # Get file size if file exists
                    file_size = 0
                    if os.path.exists(self.current_video_path):
                        file_size = os.path.getsize(self.current_video_path)

                        # Only process valid video files (non-zero size)
                        if file_size > 0:
                            # Create or update database record using MongoDB
                            # Check if a record already exists
                            existing_recording_data = db.recordings.find_one({
                                'camera_id': str(self.camera.id),
                                'file_path': self.current_video_path
                            })

                            if not existing_recording_data:
                                # Create new recording
                                Recording.create(
                                    camera_id=self.camera.id,
                                    file_path=self.current_video_path,
                                    timestamp=self.video_start_time,
                                    duration=duration,
                                    file_size=file_size,
                                    recording_type='continuous'
                                )
                                logger.info(f"Added recording to database: {self.current_video_path}, duration: {duration:.1f}s")
                            else:
                                # Update existing recording
                                recording = Recording(existing_recording_data)
                                recording.duration = duration
                                recording.file_size = file_size
                                recording.save()
                                logger.debug(f"Updated existing recording in database: {self.current_video_path}")
                        else:
                            logger.warning(f"Skip adding empty recording file to database: {self.current_video_path}")
                    else:
                        logger.warning(f"Video file not found: {self.current_video_path}")

                except Exception as e:
                    logger.error(f"Error adding recording to database: {str(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")

        self.video_writer = None

        # Get storage path from application settings
        try:
            from app.routes.main_routes import get_recording_settings
            recording_settings = get_recording_settings()
            storage_base = recording_settings.get('storage_path', 'storage/recordings')
        except Exception as e:
            logger.warning(f"Could not load recording settings: {str(e)}, using defaults")
            storage_base = 'storage/recordings'

        # Create recordings directory for this camera
        video_dir = os.path.join(storage_base, 'videos', str(self.camera.id))
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

    def _process_ultralytics_boxes(self, boxes, class_names, height, width, current_time, detected_objects):
        """Process boxes from Ultralytics YOLO format
        
        Args:
            boxes: Detection boxes from the model
            class_names: Dictionary mapping class IDs to class names
            height: Frame height
            width: Frame width
            current_time: Current timestamp
            detected_objects: List to append detections to
        """
        for i in range(len(boxes)):
            try:
                box = boxes[i]
                conf = float(box.conf.item() if hasattr(box.conf, 'item') else box.conf[0])
                
                # Skip detections below confidence threshold
                if conf < self.confidence_threshold:
                    continue
                    
                class_id = int(box.cls.item() if hasattr(box.cls, 'item') else box.cls[0])
                class_name = class_names.get(class_id, f"Class_{class_id}")
                
                # Get bounding box coordinates (xyxy format)
                if hasattr(box.xyxy, 'item'):
                    xyxy = box.xyxy.cpu().numpy()[0]
                    x1, y1, x2, y2 = map(int, xyxy)
                else:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # ROI Check
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                pixel_center = Point(center_x, center_y)
                norm_center = Point(center_x / width, center_y / height)
                matched_roi_id = self._check_roi_match(pixel_center, norm_center)
                
                # Store detection details
                detected_obj = {
                    'camera_id': self.camera.id,
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox_x': x1, 'bbox_y': y1,
                    'bbox_width': x2 - x1, 'bbox_height': y2 - y1,
                    'roi_id': matched_roi_id,
                    'timestamp': current_time
                }
                detected_objects.append(detected_obj)
            except Exception as e:
                logger.error(f"Error processing ultralytics detection box: {str(e)}")
                continue

    def _process_yolov5_boxes(self, pred, class_names, height, width, current_time, detected_objects):
        """Process boxes from YOLOv5 format
        
        Args:
            pred: Detection prediction tensors from the model
            class_names: Dictionary mapping class IDs to class names
            height: Frame height
            width: Frame width
            current_time: Current timestamp
            detected_objects: List to append detections to
        """
        # Process YOLOv5 format boxes
        for *xyxy, conf, cls in pred:
            try:
                # Skip detections below confidence threshold
                if float(conf) < self.confidence_threshold:
                    continue
                    
                # Convert tensor values to Python types
                x1, y1, x2, y2 = map(int, xyxy)
                conf = float(conf)
                class_id = int(cls)
                class_name = class_names.get(class_id, f"Class_{class_id}")
                
                # ROI Check
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                pixel_center = Point(center_x, center_y)
                norm_center = Point(center_x / width, center_y / height)
                matched_roi_id = self._check_roi_match(pixel_center, norm_center)
                
                # Store detection details
                detected_obj = {
                    'camera_id': self.camera.id,
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox_x': x1, 'bbox_y': y1,
                    'bbox_width': x2 - x1, 'bbox_height': y2 - y1,
                    'roi_id': matched_roi_id,
                    'timestamp': current_time
                }
                detected_objects.append(detected_obj)
            except Exception as e:
                logger.error(f"Error processing YOLOv5 detection box: {str(e)}")
                continue

    def _run_face_detection(self, frame, current_time, height, width):
        """Run UniFace-based face detection for the current frame.
        
        The detection engine now handles quality filtering (min size, blur detection)
        and bounding box padding internally, so this method is simplified.
        """

        if not self.face_detection_enabled:
            return []

        engine = self._ensure_face_detection_engine()
        if not engine or not getattr(engine, "ready", False):
            return []

        # Pass frame dimensions for quality filtering and padding calculations
        face_results = engine.detect(frame, frame_height=height, frame_width=width)
        if not face_results:
            return []

        detections = []
        for face in face_results:
            try:
                # Face detection engine now provides pre-filtered, padded coordinates
                x1 = float(face.get('bbox_x', 0.0))
                y1 = float(face.get('bbox_y', 0.0))
                w = float(face.get('bbox_width', 0.0))
                h = float(face.get('bbox_height', 0.0))
                confidence = float(face.get('confidence', 0.0))

                # Note: confidence filtering now happens in the engine,
                # but we double-check against dynamic settings here
                min_conf = self.face_detection_settings.get('face_detection_confidence', 0.5)
                if confidence < min_conf:
                    continue

                # Clamp bounding box to frame boundaries (safety check)
                x1 = max(0.0, min(float(width - 1), x1))
                y1 = max(0.0, min(float(height - 1), y1))
                x2 = max(x1, min(float(width), x1 + max(0.0, w)))
                y2 = max(y1, min(float(height), y1 + max(0.0, h)))

                bbox_width = int(max(1.0, x2 - x1))
                bbox_height = int(max(1.0, y2 - y1))
                x1_int = int(round(x1))
                y1_int = int(round(y1))

                center_x = x1_int + bbox_width / 2.0
                center_y = y1_int + bbox_height / 2.0
                pixel_center = Point(center_x, center_y)
                norm_center = Point(center_x / max(1, width), center_y / max(1, height))
                matched_roi_id = self._check_roi_match(pixel_center, norm_center)

                # Get detector type from engine if available
                detector_type = getattr(engine, 'detector_type', None)
                source = f"uniface_{detector_type.value}" if detector_type else 'uniface'

                detection = {
                    'camera_id': self.camera.id,
                    'class_name': 'face',
                    'confidence': confidence,
                    'bbox_x': x1_int,
                    'bbox_y': y1_int,
                    'bbox_width': bbox_width,
                    'bbox_height': bbox_height,
                    'roi_id': matched_roi_id,
                    'timestamp': current_time,
                    'source': source
                }

                if 'landmarks' in face:
                    detection['landmarks'] = face['landmarks']
                
                if 'detector_type' in face:
                    detection['detector_type'] = face['detector_type']

                detections.append(detection)
            except Exception as exc:
                logger.debug("Failed to process UniFace detection: %s", exc)

        if detections:
            logger.debug("Face detection found %d faces for camera %s", len(detections), self.camera.id)

        return detections

    def _augment_face_recognition(self, frame, face_detections, current_time):
        """Run recognition on detected faces and persist crops."""

        if not face_detections:
            return

        height, width = frame.shape[:2]
        engine = self._ensure_face_recognition_engine() if self.face_recognition_enabled else None

        for detection in face_detections:
            x1 = int(max(0, detection.get('bbox_x', 0)))
            y1 = int(max(0, detection.get('bbox_y', 0)))
            w = int(max(1, detection.get('bbox_width', 0)))
            h = int(max(1, detection.get('bbox_height', 0)))
            x2 = min(width, x1 + w)
            y2 = min(height, y1 + h)

            if x2 - x1 < 16 or y2 - y1 < 16:
                detection['face_status'] = detection.get('face_status', 'unlabeled')
                continue

            crop = frame[y1:y2, x1:x2].copy()
            crop_path = self._store_face_crop(crop, current_time)
            if crop_path:
                detection['face_crop_path'] = crop_path
                detection.setdefault('attributes', {})['face_crop_path'] = crop_path

            if not engine or not engine.ready:
                detection['face_status'] = detection.get('face_status', 'unlabeled')
                continue

            embedding = engine.embed(crop)
            if embedding is None:
                detection['face_status'] = 'unlabeled'
                continue

            matched_profile, score = FaceProfile.find_best_match(embedding, self.face_recognition_threshold)
            if matched_profile:
                profile = matched_profile
                profile.register_match(embedding, crop_path, current_time, str(self.camera.id), None)
                detection['face_profile_id'] = profile.id
                detection['face_recognition_score'] = score
                detection['face_recognition_name'] = profile.name
                detection['face_status'] = 'known' if profile.name else 'unlabeled'
                detection.setdefault('attributes', {})['face_profile_id'] = profile.id
            elif self._face_recognition_auto_create:
                profile = FaceProfile.create(
                    embedding,
                    crop_path,
                    str(self.camera.id),
                    current_time,
                )
                detection['face_profile_id'] = profile.id
                detection['face_recognition_score'] = None
                detection['face_recognition_name'] = profile.name
                detection['face_status'] = 'unlabeled'
                detection.setdefault('attributes', {})['face_profile_id'] = profile.id
            else:
                detection['face_status'] = 'unlabeled'

    def _store_face_crop(self, crop, current_time):
        import cv2

        try:
            base_dir = os.path.join('storage', 'faces', str(self.camera.id))
            os.makedirs(base_dir, exist_ok=True)
            filename = f"{current_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
            path = os.path.join(base_dir, filename)
            cv2.imwrite(path, crop)
            return path
        except Exception as exc:
            logger.debug("Failed to persist face crop: %s", exc)
            return None

    def _handle_detection_image_saving(self, frame, detected_objects, current_time):
        """Handle saving detection images efficiently
        
        Args:
            frame: The frame with detections
            detected_objects: List of detection objects
            current_time: Current timestamp
        """
        try:
            # Check if image saving is enabled
            from app.routes.main_routes import get_detection_settings
            detection_settings = get_detection_settings()
            save_image = detection_settings.get('save_images', True)
            
            if not save_image or not detected_objects:
                # Skip image saving if disabled or no detections
                for obj in detected_objects:
                    obj['image_path'] = None
                return
            
            # Draw boxes on a copy of the processed frame for saving
            frame_with_boxes = frame.copy()
            
            for obj in detected_objects:
                # Draw detection boxes
                if all(k in obj for k in ['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']):
                    x1, y1 = int(obj['bbox_x']), int(obj['bbox_y'])
                    x2, y2 = x1 + int(obj['bbox_width']), y1 + int(obj['bbox_height'])
                    color = get_color_for_class(obj['class_name'])
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            # Save a single image for all detections in this frame
            detection_time_str = current_time.strftime("%Y%m%d_%H%M%S")
            base_image_dir = os.path.join('storage', 'recordings', 'images')
            os.makedirs(base_image_dir, exist_ok=True)
            
            # Create camera-specific directory
            image_dir = os.path.join(base_image_dir, str(self.camera.id))
            os.makedirs(image_dir, exist_ok=True)
            
            # Generate a unique filename
            image_path = os.path.join(image_dir, f"{detection_time_str}_{uuid.uuid4().hex[:8]}.jpg")
            
            try:
                # Save the image
                cv2.imwrite(image_path, frame_with_boxes)
                
                # Add image path to all detections
                for obj in detected_objects:
                    obj['image_path'] = image_path
                    
            except Exception as img_err:
                logger.error(f"Error saving detection image to {image_path}: {img_err}")
                for obj in detected_objects:
                    obj['image_path'] = None
            
            # Release memory
            frame_with_boxes = None
            
        except Exception as e:
            logger.error(f"Error in handling detection image saving: {str(e)}")
            # Ensure image paths are set to None in case of error
            for obj in detected_objects:
                obj['image_path'] = None
        
        # Set video path if recording
        video_path = self.current_video_path if self.recording else None
        for obj in detected_objects:
            obj['video_path'] = video_path

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
        # Discover available GPUs once; use round-robin assignment across them
        try:
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            gpu_count = 0
        self.available_gpus = [f"cuda:{i}" for i in range(gpu_count)] if gpu_count > 0 else []
        self._next_gpu_idx = 0
        if len(self.available_gpus) > 1:
            logger.info(f"Multi-GPU detected: {len(self.available_gpus)} GPUs available → {self.available_gpus}")
        elif len(self.available_gpus) == 1:
            logger.info("Single GPU detected: cuda:0")
        else:
            logger.info("No GPU detected; using CPU for inference")

    def _assign_device(self, camera_id) -> str | None:
        """Assign a compute device for a camera using round-robin across GPUs.

        Returns a torch device string like 'cuda:0' or None to auto-select later.
        """
        if not self.available_gpus:
            return None  # CPU-only or let processor auto-select
        # Round-robin across available GPUs
        device = self.available_gpus[self._next_gpu_idx]
        self._next_gpu_idx = (self._next_gpu_idx + 1) % len(self.available_gpus)
        logger.info(f"Assigning camera {camera_id} to device {device}")
        return device

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
            assigned_device = self._assign_device(camera.id)
            processor = CameraProcessor(camera, device=assigned_device)
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
        from app.models.camera import Camera
        
        # Use MongoDB style query instead of SQLAlchemy
        cameras = Camera.get_active_cameras()
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