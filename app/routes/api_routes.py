"""
API routes for the SmartNVR application
Provides endpoints for video streaming, camera control, and data retrieval
"""
from flask import Blueprint, request, jsonify, Response, send_file, abort
from flask_login import login_required, current_user
import os
import json
from datetime import datetime, timedelta
import time
from bson.objectid import ObjectId

from app import db
from app.models.camera import Camera
from app.models.recording import Recording
from app.models.detection import Detection
from app.models.roi import ROI
from app.utils.decorators import admin_required, api_key_required
from app.utils.system_monitor import get_system_stats

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# --- Camera API Endpoints ---

# Function to handle detection reports from camera processor
def report_detection(request):
    """
    Process detection report from camera processor
    Not a route, called directly from camera processor
    """
    try:
        import logging
        logger = logging.getLogger(__name__)
        
        data = request.get_json()
        
        if not data or 'camera_id' not in data or 'detections' not in data:
            return jsonify({
                'success': False,
                'message': 'Invalid detection data format'
            }), 400
            
        camera_id = data['camera_id']
        camera = Camera.get_by_id(camera_id)
        
        if not camera:
            return jsonify({
                'success': False,
                'message': f'Camera not found: {camera_id}'
            }), 404
            
        detections_data = data['detections']
        
        if not detections_data:
            # No detections to process
            return jsonify({
                'success': True,
                'message': 'No detections to process'
            })
            
        # Get or create recording based on timestamp of first detection
        recording = None
        detection_timestamp = None
        
        if detections_data and 'timestamp' in detections_data[0]:
            detection_timestamp = detections_data[0]['timestamp']
            if isinstance(detection_timestamp, str):
                detection_timestamp = datetime.fromisoformat(detection_timestamp)
                
            # Look for existing recording in the last minute
            recent_recordings = list(db.recordings.find({
                'camera_id': str(camera_id),
                'timestamp': {'$gte': detection_timestamp - timedelta(minutes=1)}
            }).sort('timestamp', -1).limit(1))
            
            if recent_recordings:
                recording_data = recent_recordings[0]
                recording = Recording(recording_data)
        
        # Process each detection
        new_detections = []
        for det_data in detections_data:
            # Debug ROI ID to track if it's being properly passed
            roi_id = det_data.get('roi_id')
            logger.debug(f"Processing detection with ROI ID: {roi_id}")
            
            # Create detection object
            detection = Detection.create(
                camera_id=camera_id,
                class_name=det_data.get('class_name', 'unknown'),
                confidence=det_data.get('confidence', 0.0),
                bbox_x=det_data.get('bbox_x', 0),
                bbox_y=det_data.get('bbox_y', 0),
                bbox_width=det_data.get('bbox_width', 0),
                bbox_height=det_data.get('bbox_height', 0),
                timestamp=det_data.get('timestamp', datetime.now()) if isinstance(det_data.get('timestamp'), datetime) else datetime.now(),
                recording_id=recording.id if recording else None,
                roi_id=roi_id,
                image_path=det_data.get('image_path'),
                video_path=det_data.get('video_path')
            )
            
            new_detections.append(detection)
        
        # Process email notifications for new detections
        for detection in new_detections:
            try:
                # Check if this detection is in an ROI with email notifications enabled
                roi = None
                
                if detection.roi_id:
                    roi = ROI.get_by_id(detection.roi_id)
                    
                    # Process email notifications (non-blocking since it's now handled asynchronously)
                    if roi and getattr(roi, 'email_notifications', False):
                        from app.utils.notifications import send_detection_email
                        # This will queue the email for sending in a background thread
                        send_detection_email(camera, detection, roi)
                        logger.info(f"Queued email notification for {detection.class_name} in ROI {roi.name}")
                
            except Exception as e:
                logger.error(f"Error processing notification: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            'success': True,
            'message': f'Processed {len(new_detections)} detections'
        })
    
    except Exception as e:
        import traceback
        logger.error(f"Error processing detection report: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@api_bp.route('/cameras')
@login_required
def get_cameras():
    """Get all active cameras"""
    cameras = Camera.get_active_cameras()
    return jsonify({
        'success': True,
        'cameras': [camera.to_dict() for camera in cameras]
    })

@api_bp.route('/cameras/<camera_id>')
@login_required
def get_camera(camera_id):
    """Get camera details"""
    camera = Camera.get_by_id(camera_id)
    if not camera:
        return jsonify({
            'success': False,
            'message': 'Camera not found'
        }), 404
        
    return jsonify({
        'success': True,
        'camera': camera.to_dict()
    })

@api_bp.route('/cameras/<camera_id>/frame')
@login_required
def get_camera_frame(camera_id):
    """Get a single frame from camera as JPEG image"""
    from app.utils.camera_processor import CameraManager
    import cv2
    
    camera = Camera.get_by_id(camera_id)
    if not camera:
        return jsonify({
            'success': False,
            'message': 'Camera not found'
        }), 404
    
    # Get camera processor or start it if not running
    manager = CameraManager.get_instance()
    processor = manager.get_camera_processor(camera_id)
    
    if not processor:
        # Try to start the camera
        if not manager.start_camera(camera):
            # If we can't start the camera, return placeholder
            return send_file('static/img/no-signal.png', mimetype='image/jpeg')
    
    # Get the latest frame
    frame = processor.get_frame()
    
    if frame is None:
        return send_file('static/img/no-signal.png', mimetype='image/jpeg')
    
    # Check quality parameter
    quality = request.args.get('quality', 'medium')
    quality_value = 90  # Default high quality
    
    if quality == 'low':
        # Low quality, reduce resolution and JPEG quality
        height, width = frame.shape[:2]
        frame = cv2.resize(frame, (width // 2, height // 2))
        quality_value = 50
    elif quality == 'medium':
        # Medium quality
        quality_value = 70
    
    # Convert frame to JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality_value]
    _, buffer = cv2.imencode('.jpg', frame, encode_params)
    
    # Return as response
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@api_bp.route('/cameras/<camera_id>/stream')
@login_required
def get_camera_stream(camera_id):
    """Get camera stream (MJPEG)"""
    from app.utils.camera_processor import CameraManager
    import cv2
    
    camera = Camera.get_by_id(camera_id)
    if not camera:
        return jsonify({
            'success': False,
            'message': 'Camera not found'
        }), 404
    
    # Get camera processor or start it if not running
    manager = CameraManager.get_instance()
    processor = manager.get_camera_processor(camera_id)
    
    if not processor:
        # Try to start the camera
        manager.start_camera(camera)
    
    def generate():
        """Generate MJPEG stream"""
        while True:
            # Get the processor again in case it was started after we checked
            processor = manager.get_camera_processor(camera_id)
            if not processor:
                # If camera isn't running, yield placeholder frame
                with open('static/img/no-signal.png', 'rb') as f:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + f.read() + b'\r\n')
                    time.sleep(1)
                    continue
            
            # Get the latest frame
            frame = processor.get_frame()
            
            if frame is None:
                # If no frame available, yield placeholder
                with open('static/img/no-signal.png', 'rb') as f:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + f.read() + b'\r\n')
            else:
                # Convert frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                
                # Yield frame for MJPEG stream
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Rate limit stream
            time.sleep(0.1)
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@api_bp.route('/cameras/<camera_id>/snapshot')
@login_required
def get_camera_snapshot(camera_id):
    """Get current camera snapshot"""
    # This is just an alias for the frame endpoint
    return get_camera_frame(camera_id)

@api_bp.route('/cameras/<camera_id>/roi', methods=['GET'])
@login_required
def get_camera_roi(camera_id):
    """Get camera regions of interest"""
    camera = Camera.get_by_id(camera_id)
    if not camera:
        return jsonify({
            'success': False,
            'message': 'Camera not found'
        }), 404
        
    rois = ROI.get_by_camera(camera_id)
    
    return jsonify({
        'success': True,
        'roi': [roi.to_dict() for roi in rois]
    })

@api_bp.route('/cameras/<camera_id>/roi', methods=['POST'])
@login_required
def create_camera_roi(camera_id):
    """Create new region of interest for camera"""
    camera = Camera.get_by_id(camera_id)
    if not camera:
        return jsonify({
            'success': False,
            'message': 'Camera not found'
        }), 404
        
    data = request.json
    
    if not data:
        return jsonify({
            'success': False,
            'message': 'No data provided'
        }), 400
        
    if not all(key in data for key in ['name', 'coordinates']):
        return jsonify({
            'success': False,
            'message': 'Missing required fields: name and coordinates'
        }), 400
    
    # Create new ROI
    try:
        roi = ROI.create(
            camera_id=camera_id,
            name=data['name'],
            coordinates=data['coordinates'],  # Pass as-is, ROI.create handles conversion
            detection_classes=data.get('detection_classes', []),  # Pass as-is
            is_active=data.get('is_active', True),
            email_notifications=data.get('email_notifications', False)
        )
        
        # Now try to reload ROIs for the camera
        try:
            from app.utils.camera_processor import CameraManager
            manager = CameraManager.get_instance()
            manager.reload_rois(camera_id)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to reload ROIs: {str(e)}")
            # Continue execution - this shouldn't fail the whole request
        
        return jsonify({
            'success': True,
            'roi': roi.to_dict()
        }), 201
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error creating ROI: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error creating ROI: {str(e)}"
        }), 500

@api_bp.route('/cameras/<camera_id>/roi/<roi_id>', methods=['PUT'])
@login_required
def update_camera_roi(camera_id, roi_id):
    """Update region of interest for camera"""
    roi = ROI.get_by_id(roi_id)
    if not roi or roi.camera_id != camera_id:
        return jsonify({
            'success': False,
            'message': 'ROI not found for this camera'
        }), 404
        
    data = request.json
    
    if not data:
        return jsonify({
            'success': False,
            'message': 'No data provided'
        }), 400
    
    # Update fields
    if 'name' in data:
        roi.name = data['name']
    if 'coordinates' in data:
        roi.coordinates = data['coordinates']  # Store as native array
    if 'detection_classes' in data:
        roi.detection_classes = data['detection_classes']  # Store as native array
    if 'is_active' in data:
        roi.is_active = data['is_active']
    if 'email_notifications' in data:
        roi.email_notifications = data['email_notifications']
    
    # Save the updated ROI
    roi.save()
    
    # Reload ROIs for the camera
    try:
        from app.utils.camera_processor import CameraManager
        manager = CameraManager.get_instance()
        manager.reload_rois(camera_id)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to reload ROIs: {str(e)}")
    
    return jsonify({
        'success': True,
        'roi': roi.to_dict()
    })

@api_bp.route('/cameras/<camera_id>/roi/<roi_id>', methods=['DELETE'])
@login_required
def delete_camera_roi(camera_id, roi_id):
    """Delete region of interest for camera"""
    roi = ROI.get_by_id(roi_id)
    if not roi or roi.camera_id != camera_id:
        return jsonify({
            'success': False,
            'message': 'ROI not found for this camera'
        }), 404
    
    # Delete the ROI
    roi.delete()
    
    # Reload ROIs for the camera
    try:
        from app.utils.camera_processor import CameraManager
        manager = CameraManager.get_instance()
        manager.reload_rois(camera_id)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to reload ROIs: {str(e)}")
    
    return jsonify({
        'success': True,
        'message': 'ROI deleted successfully'
    })

@api_bp.route('/cameras/<camera_id>/detections/latest')
@login_required
def get_latest_camera_detections(camera_id):
    """Get latest detections for a specific camera"""
    # Verify camera exists
    camera = Camera.get_by_id(camera_id)
    if not camera:
        return jsonify({
            'success': False,
            'message': 'Camera not found'
        }), 404
    
    try:
        # Get real-time detections directly from camera processor
        from app.utils.camera_processor import CameraManager
        manager = CameraManager.get_instance()
        processor = manager.get_camera_processor(camera_id)
        
        if processor and hasattr(processor, 'get_latest_detections'):
            # If processor has real-time detections, use those
            detections = processor.get_latest_detections()
            if detections:
                return jsonify(detections)
    except Exception as e:
        print(f"Error getting real-time detections: {str(e)}")
    
    # Fall back to database detections if no real-time detections available
    try:
        # Try directly getting camera detections
        detections = Detection.get_by_camera(camera_id, limit=20)
        
        # If no direct camera detections, try via recordings
        if not detections:
            recordings = Recording.get_by_camera(camera_id)
            recording_ids = [rec.id for rec in recordings]
            
            if recording_ids:
                all_detections = []
                for recording_id in recording_ids:
                    all_detections.extend(Detection.get_by_recording(recording_id))
                # Sort by timestamp and take latest 20
                detections = sorted(all_detections, key=lambda d: d.timestamp, reverse=True)[:20]
    except Exception as e:
        print(f"Error getting database detections: {str(e)}")
        detections = []
    
    # Convert detections to list of dicts with coordinates
    results = []
    for det in detections:
        # Skip detections without coordinates
        if not all(hasattr(det, attr) for attr in ['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']):
            continue
        
        results.append({
            'id': det.id,
            'class_name': det.class_name,
            'confidence': det.confidence,
            'coordinates': {
                'x_min': float(det.bbox_x),
                'y_min': float(det.bbox_y),
                'x_max': float(det.bbox_x) + float(det.bbox_width),
                'y_max': float(det.bbox_y) + float(det.bbox_height)
            },
            'timestamp': det.timestamp.isoformat() if isinstance(det.timestamp, datetime) else det.timestamp
        })
    
    return jsonify(results)

@api_bp.route('/cameras/<camera_id>/recordings', methods=['GET'])
@login_required
def get_camera_recordings_by_date(camera_id):
    """Get recordings for a specific camera by date"""
    import logging
    logger = logging.getLogger(__name__)
    
    # Verify camera exists
    camera = Camera.get_by_id(camera_id)
    if not camera:
        return jsonify({
            'success': False,
            'message': 'Camera not found'
        }), 404
    
    # Get query parameters
    date = request.args.get('date')
    events_only = request.args.get('events_only', '').lower() in ('true', '1', 'yes')
    object_type = request.args.get('object_type')
    
    logger.info(f"Getting recordings for camera {camera_id} on date {date}, events_only={events_only}, object_type={object_type}")
    
    # Date is required
    if not date:
        return jsonify({
            'success': False,
            'message': 'Date parameter is required in format YYYY-MM-DD'
        }), 400
    
    try:
        # Parse date
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        next_day = date_obj + timedelta(days=1)
        
        # Build MongoDB query for recordings
        recordings_query = {
            'camera_id': str(camera_id),
            'timestamp': {
                '$gte': date_obj,
                '$lt': next_day
            }
        }
        
        # Get recordings matching our criteria, sorted by timestamp
        recordings_cursor = db.recordings.find(recordings_query).sort('timestamp', 1)
        recordings = [Recording(rec) for rec in recordings_cursor]
        
        logger.info(f"Found {len(recordings)} recordings for camera {camera_id} on date {date}")
        
        # Build query for detections - used both for filtering recordings and returning detections
        detections_query = {
            'camera_id': str(camera_id),
            'timestamp': {
                '$gte': date_obj,
                '$lt': next_day
            }
        }
        
        # Filter by object type if provided
        if object_type:
            detections_query['class_name'] = object_type
        
        # Get detections
        detections_cursor = db.detections.find(detections_query).sort('timestamp', 1)
        detections = [Detection(det) for det in detections_cursor]
        
        logger.info(f"Found {len(detections)} detections for camera {camera_id} on date {date}")
        
        # If events_only is true, filter recordings to only include those with detections
        if events_only and detections:
            # Get unique recording_ids from detections that have them
            recording_ids_with_detections = set()
            for det in detections:
                if det.recording_id:
                    recording_ids_with_detections.add(det.recording_id)
            
            # Filter recordings to only include those with detections
            if recording_ids_with_detections:
                recordings = [rec for rec in recordings if rec.id in recording_ids_with_detections]
                logger.info(f"Filtered to {len(recordings)} recordings with detections")
        
        # Convert to dictionaries for JSON response
        recordings_dict = [r.to_dict() for r in recordings]
        detections_dict = [d.to_dict() for d in detections]
        
        # Return formatted response
        return jsonify({
            'success': True,
            'date': date,
            'camera_id': camera_id,
            'recordings': recordings_dict,
            'detections': detections_dict
        })
        
    except ValueError:
        return jsonify({
            'success': False,
            'message': 'Invalid date format. Use YYYY-MM-DD'
        }), 400
    except Exception as e:
        logger.error(f"Error getting recordings for camera {camera_id}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

# --- Recordings API Endpoints ---

@api_bp.route('/recordings')
@login_required
def get_recordings():
    """Get recordings with filters"""
    # Get query parameters
    camera_id = request.args.get('camera_id')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    recording_type = request.args.get('type')
    has_detections = request.args.get('has_detections', '').lower() in ('true', '1', 'yes')
    
    # Build MongoDB query
    query = {}
    
    if camera_id:
        query['camera_id'] = str(camera_id)
    
    # Date range filtering
    date_filter = {}
    if date_from:
        try:
            date_from_obj = datetime.strptime(date_from, '%Y-%m-%d')
            date_filter['$gte'] = date_from_obj
        except ValueError:
            return jsonify({
                'success': False,
                'message': 'Invalid date_from format. Use YYYY-MM-DD'
            }), 400
    
    if date_to:
        try:
            date_to_obj = datetime.strptime(date_to, '%Y-%m-%d') + timedelta(days=1)
            date_filter['$lt'] = date_to_obj
        except ValueError:
            return jsonify({
                'success': False,
                'message': 'Invalid date_to format. Use YYYY-MM-DD'
            }), 400
    
    if date_filter:
        query['timestamp'] = date_filter
    
    if recording_type:
        query['recording_type'] = recording_type
    
    # For has_detections, we need to handle differently
    if has_detections is not None:
        # Get distinct recording IDs from detections collection
        recordings_with_detections = list(db.detections.distinct('recording_id'))
        
        if has_detections:
            # Only include recordings with detections
            query['_id'] = {'$in': [ObjectId(rid) for rid in recordings_with_detections if rid]}
        else:
            # Only include recordings without detections
            query['_id'] = {'$nin': [ObjectId(rid) for rid in recordings_with_detections if rid]}
    
    # Paginate results
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # Calculate skip and limit for pagination
    skip = (page - 1) * per_page
    
    # Query MongoDB with pagination
    total = db.recordings.count_documents(query)
    recordings_cursor = db.recordings.find(query).sort('timestamp', -1).skip(skip).limit(per_page)
    recordings = [Recording(rec) for rec in recordings_cursor]
    
    # Calculate pagination info
    total_pages = (total + per_page - 1) // per_page  # ceiling division
    
    return jsonify({
        'success': True,
        'recordings': [recording.to_dict() for recording in recordings],
        'pagination': {
            'total': total,
            'pages': total_pages,
            'page': page,
            'per_page': per_page
        }
    })

@api_bp.route('/recordings/<recording_id>')
@login_required
def get_recording(recording_id):
    """Get recording details"""
    recording = Recording.get_by_id(recording_id)
    if not recording:
        return jsonify({
            'success': False,
            'message': 'Recording not found'
        }), 404
    
    return jsonify({
        'success': True,
        'recording': recording.to_dict()
    })

@api_bp.route('/recordings/<recording_id>/video')
@login_required
def get_recording_video(recording_id):
    """Stream recording video with transcoding support if needed"""
    import logging
    import os
    import subprocess
    import tempfile
    import re
    from flask import Response, abort, request, send_file

    logger = logging.getLogger(__name__)
    
    recording = Recording.get_by_id(recording_id)
    if not recording:
        return jsonify({
            'success': False,
            'message': 'Recording not found'
        }), 404
    
    if not os.path.exists(recording.file_path):
        logger.error(f"Recording file not found: {recording.file_path}")
        abort(404, description="Recording file not found")
    
    logger.info(f"Serving video file: {recording.file_path}")
    
    # Check if the browser supports direct playback (mobile browsers often have better codec support)
    user_agent = request.headers.get('User-Agent', '').lower()
    mobile = 'mobile' in user_agent or 'android' in user_agent or 'iphone' in user_agent or 'ipad'
    
    # Check video format - if it's MPEG-4 and not mobile, we'll transcode
    need_transcode = False
    try:
        # Use ffprobe to check video codec
        ffprobe_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', 
            '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1',
            recording.file_path
        ]
        codec = subprocess.check_output(ffprobe_cmd, stderr=subprocess.STDOUT).decode('utf-8').strip()
        logger.info(f"Video codec: {codec}")
        
        # Need to transcode if it's mpeg4 (not h264)
        need_transcode = codec == 'mpeg4' and not mobile
    except Exception as e:
        logger.error(f"Error checking video codec: {str(e)}")
        # If we can't check, assume no need to transcode
        need_transcode = False
    
    if need_transcode:
        logger.info(f"Transcoding video to H.264 for browser compatibility")
        
        # Check for range requests
        range_header = request.headers.get('Range', None)
        
        def generate_h264():
            """Stream H.264 transcoded video"""
            ffmpeg_cmd = [
                'ffmpeg', '-i', recording.file_path,
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-movflags', 'frag_keyframe+empty_moov+faststart',
                '-f', 'mp4', '-'
            ]
            
            process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
            
            try:
                while True:
                    chunk = process.stdout.read(1024*1024)  # 1MB chunks
                    if not chunk:
                        break
                    yield chunk
            finally:
                process.kill()
        
        return Response(
            generate_h264(),
            mimetype='video/mp4',
            headers={
                'Accept-Ranges': 'bytes',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'X-Content-Type-Options': 'nosniff',
                'Content-Disposition': f'inline; filename="{os.path.basename(recording.file_path)}"'
            }
        )
    else:
        # No transcoding needed, send file directly
        return send_file(
            recording.file_path,
            mimetype='video/mp4',
            conditional=True,
            etag=True,
        )

@api_bp.route('/recordings/<recording_id>/thumbnail')
@login_required
def get_recording_thumbnail(recording_id):
    """Get recording thumbnail"""
    recording = Recording.get_by_id(recording_id)
    if not recording:
        return jsonify({
            'success': False,
            'message': 'Recording not found'
        }), 404
    
    if recording.thumbnail_path and os.path.exists(recording.thumbnail_path):
        return send_file(recording.thumbnail_path, mimetype='image/jpeg')
    
    # Return default thumbnail if none exists
    return send_file('static/img/no-thumbnail.png', mimetype='image/jpeg')

@api_bp.route('/recordings/<recording_id>/download')
@login_required
def download_recording(recording_id):
    """Download recording file"""
    recording = Recording.get_by_id(recording_id)
    if not recording:
        return jsonify({
            'success': False,
            'message': 'Recording not found'
        }), 404
    
    if not os.path.exists(recording.file_path):
        abort(404, description="Recording file not found")
    
    # Generate a download filename based on camera name and timestamp
    camera = Camera.get_by_id(recording.camera_id)
    camera_name = camera.name if camera else f"camera-{recording.camera_id}"
    timestamp = recording.timestamp.strftime('%Y%m%d-%H%M%S') if isinstance(recording.timestamp, datetime) else "unknown"
    filename = f"{camera_name}-{timestamp}.mp4"
    
    return send_file(
        recording.file_path,
        as_attachment=True,
        download_name=filename,
        mimetype='video/mp4'
    )

@api_bp.route('/recordings/<recording_id>/flag', methods=['POST'])
@login_required
def flag_recording(recording_id):
    """Flag recording as important"""
    recording = Recording.get_by_id(recording_id)
    if not recording:
        return jsonify({
            'success': False,
            'message': 'Recording not found'
        }), 404
        
    recording.is_flagged = True
    recording.save()
    
    return jsonify({
        'success': True,
        'message': 'Recording flagged successfully'
    })

@api_bp.route('/recordings/<recording_id>/unflag', methods=['POST'])
@login_required
def unflag_recording(recording_id):
    """Remove flag from recording"""
    recording = Recording.get_by_id(recording_id)
    if not recording:
        return jsonify({
            'success': False,
            'message': 'Recording not found'
        }), 404
        
    recording.is_flagged = False
    recording.save()
    
    return jsonify({
        'success': True,
        'message': 'Recording unflagged successfully'
    })

@api_bp.route('/recordings/<recording_id>', methods=['DELETE'])
@login_required
def delete_recording(recording_id):
    """Delete recording and file"""
    recording = Recording.get_by_id(recording_id)
    if not recording:
        return jsonify({
            'success': False,
            'message': 'Recording not found'
        }), 404
    
    # Delete file if it exists
    if recording.file_path and os.path.exists(recording.file_path):
        os.remove(recording.file_path)
    
    # Delete thumbnail if it exists
    if recording.thumbnail_path and os.path.exists(recording.thumbnail_path):
        os.remove(recording.thumbnail_path)
    
    # Delete from database
    recording.delete()
    
    return jsonify({
        'success': True,
        'message': 'Recording deleted successfully'
    })

# --- Detection API Endpoints ---

@api_bp.route('/detections')
@login_required
def get_detections():
    """Get detections with filters"""
    # Get query parameters
    camera_id = request.args.get('camera_id')
    class_name = request.args.get('class_name')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    
    # Build MongoDB query
    query = {}
    
    if camera_id:
        query['camera_id'] = str(camera_id)
    
    if class_name:
        query['class_name'] = class_name
    
    # Date range filtering
    date_filter = {}
    if date_from:
        try:
            date_from_obj = datetime.strptime(date_from, '%Y-%m-%d')
            date_filter['$gte'] = date_from_obj
        except ValueError:
            return jsonify({
                'success': False,
                'message': 'Invalid date_from format. Use YYYY-MM-DD'
            }), 400
    
    if date_to:
        try:
            date_to_obj = datetime.strptime(date_to, '%Y-%m-%d') + timedelta(days=1)
            date_filter['$lt'] = date_to_obj
        except ValueError:
            return jsonify({
                'success': False,
                'message': 'Invalid date_to format. Use YYYY-MM-DD'
            }), 400
    
    if date_filter:
        query['timestamp'] = date_filter
    
    # Paginate results
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    # Calculate skip and limit for pagination
    skip = (page - 1) * per_page
    
    # Query MongoDB with pagination
    total = db.detections.count_documents(query)
    detections_cursor = db.detections.find(query).sort('timestamp', -1).skip(skip).limit(per_page)
    detections = [Detection(det) for det in detections_cursor]
    
    # Calculate pagination info
    total_pages = (total + per_page - 1) // per_page  # ceiling division
    
    return jsonify({
        'success': True,
        'detections': [detection.to_dict() for detection in detections],
        'pagination': {
            'total': total,
            'pages': total_pages,
            'page': page,
            'per_page': per_page
        }
    })

@api_bp.route('/detections/summary')
@login_required
def get_detection_summary():
    """Get summary of detections"""
    # Get time range parameters
    days = request.args.get('days', 7, type=int)
    start_date = datetime.now() - timedelta(days=days)
    
    # Get detection counts by class using MongoDB aggregation
    class_pipeline = [
        {'$match': {'timestamp': {'$gte': start_date}}},
        {'$group': {'_id': '$class_name', 'count': {'$sum': 1}}}
    ]
    
    class_counts = list(db.detections.aggregate(class_pipeline))
    
    # Get detection counts by camera
    camera_pipeline = [
        {'$match': {'timestamp': {'$gte': start_date}}},
        {'$group': {'_id': '$camera_id', 'count': {'$sum': 1}}}
    ]
    
    camera_counts = list(db.detections.aggregate(camera_pipeline))
    
    # Format results
    class_summary = {item['_id']: item['count'] for item in class_counts if item['_id']}
    
    # Get camera names for the summary
    camera_summary = {}
    for item in camera_counts:
        if item['_id']:
            camera = Camera.get_by_id(item['_id'])
            camera_name = camera.name if camera else f"camera-{item['_id']}"
            camera_summary[camera_name] = item['count']
    
    return jsonify({
        'success': True,
        'class_summary': class_summary,
        'camera_summary': camera_summary,
        'total': sum(item['count'] for item in class_counts),
        'time_range': {
            'days': days,
            'start_date': start_date.strftime('%Y-%m-%d')
        }
    })

@api_bp.route('/detections/<detection_id>')
@login_required
def get_detection(detection_id):
    """Get detection details"""
    detection = Detection.get_by_id(detection_id)
    if not detection:
        return jsonify({
            'success': False,
            'message': 'Detection not found'
        }), 404
    
    return jsonify({
        'success': True,
        'detection': detection.to_dict()
    })

@api_bp.route('/detections/<detection_id>/image')
@login_required
def get_detection_image(detection_id):
    """Get detection image"""
    detection = Detection.get_by_id(detection_id)
    if not detection:
        return jsonify({
            'success': False,
            'message': 'Detection not found'
        }), 404
    
    if detection.image_path and os.path.exists(detection.image_path):
        return send_file(detection.image_path, mimetype='image/jpeg')
    
    # Return default image if none exists
    return send_file('static/img/no-detection.png', mimetype='image/jpeg')

# --- System API Endpoints ---

@api_bp.route('/system/stats')
@login_required
def get_stats():
    """Get system statistics"""
    stats = get_system_stats()
    return jsonify({
        'success': True,
        'stats': stats
    })

@api_bp.route('/system/storage')
@login_required
def get_storage():
    """Get storage statistics"""
    # Total recordings size using MongoDB aggregation
    size_pipeline = [
        {'$group': {'_id': None, 'total_size': {'$sum': '$file_size'}}}
    ]
    
    total_size_result = list(db.recordings.aggregate(size_pipeline))
    total_size = total_size_result[0]['total_size'] if total_size_result else 0
    
    # Size by camera using MongoDB aggregation
    camera_size_pipeline = [
        {'$group': {'_id': '$camera_id', 'size': {'$sum': '$file_size'}}}
    ]
    
    camera_sizes_result = list(db.recordings.aggregate(camera_size_pipeline))
    
    # Format camera sizes
    camera_storage = []
    for item in camera_sizes_result:
        camera_id = item['_id']
        if camera_id:
            camera = Camera.get_by_id(camera_id)
            if camera:
                camera_storage.append({
                    'id': camera.id,
                    'name': camera.name,
                    'size': item['size'] or 0
                })
    
    return jsonify({
        'success': True,
        'total_size': total_size,
        'camera_storage': camera_storage
    })

@api_bp.route('/system/info')
@login_required
def get_system_info():
    """Get system information"""
    import platform
    import psutil
    import os
    
    # Get system information
    info = {
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor()
        },
        'python': {
            'version': platform.python_version(),
            'implementation': platform.python_implementation()
        },
        'disk': {
            'total': psutil.disk_usage('/').total,
            'used': psutil.disk_usage('/').used,
            'free': psutil.disk_usage('/').free,
            'percent': psutil.disk_usage('/').percent
        },
        'memory': {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available,
            'used': psutil.virtual_memory().used,
            'percent': psutil.virtual_memory().percent
        },
        'cpu': {
            'cores_physical': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'percent': psutil.cpu_percent(interval=0.1)
        },
        'app': {
            'uptime': int(time.time() - psutil.Process(os.getpid()).create_time())
        }
    }
    
    # Add MongoDB statistics
    try:
        info['database'] = {
            'type': 'MongoDB',
            'cameras': db.cameras.count_documents({}),
            'recordings': db.recordings.count_documents({}),
            'detections': db.detections.count_documents({}),
            'rois': db.regions_of_interest.count_documents({}),
            'users': db.users.count_documents({})
        }
    except Exception as e:
        info['database'] = {'error': str(e)}
    
    # Add GPU information if available
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            info['gpu'] = []
            for gpu in gpus:
                info['gpu'].append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory': {
                        'total': gpu.memoryTotal,
                        'used': gpu.memoryUsed,
                        'free': gpu.memoryTotal - gpu.memoryUsed,
                        'percent': (gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0
                    },
                    'temperature': gpu.temperature
                })
    except (ImportError, Exception) as e:
        info['gpu'] = {'error': str(e)}
    
    return jsonify({
        'success': True,
        'info': info
    })

@api_bp.route('/system/resources')
@login_required
def get_system_resources():
    """Get system resource usage"""
    from app.utils.system_monitor import get_system_resources
    
    resources = get_system_resources()
    
    return jsonify({
        'success': True,
        'resources': resources
    })

@api_bp.route('/test_email', methods=['POST'])
@login_required
def test_email():
    """Test email configuration"""
    # Import here to ensure the function is available
    from app.utils.notifications import send_test_email
    
    data = request.json
    
    if not data:
        return jsonify({
            'success': False,
            'error': 'No configuration provided'
        }), 400
        
    required_fields = ['smtp_server', 'smtp_port', 'smtp_username', 'smtp_password', 'email_recipients']
    if not all(field in data for field in required_fields):
        return jsonify({
            'success': False,
            'error': 'Missing required fields'
        }), 400
        
    # Parse recipients
    recipients = [email.strip() for email in data['email_recipients'].split(',') if email.strip()]
    if not recipients:
        return jsonify({
            'success': False,
            'error': 'No recipients specified'
        }), 400
    
    # Send test email
    result = send_test_email(
        smtp_server=data['smtp_server'],
        smtp_port=int(data['smtp_port']),
        smtp_username=data['smtp_username'],
        smtp_password=data['smtp_password'],
        recipients=recipients
    )
    
    if result['success']:
        return jsonify({
            'success': True,
            'message': 'Test email sent successfully'
        })
    else:
        return jsonify({
            'success': False,
            'error': result['error']
        }), 500

# --- Web Hooks & External API Endpoints ---

@api_bp.route('/hooks/detection', methods=['POST'])
@api_key_required
def hook_detection():
    """Receive detection from external system"""
    data = request.json
    
    # Validate required fields
    required_fields = ['camera_id', 'timestamp', 'class_name', 'confidence', 'bbox']
    if not data or not all(field in data for field in required_fields):
        return jsonify({
            'success': False,
            'message': 'Missing required fields'
        }), 400
    
    # Validate camera exists
    camera = Camera.get_by_id(data['camera_id'])
    if not camera:
        return jsonify({
            'success': False,
            'message': 'Camera not found'
        }), 404
    
    # Parse timestamp
    try:
        if isinstance(data['timestamp'], (int, float)):
            timestamp = datetime.fromtimestamp(data['timestamp'])
        else:
            timestamp = datetime.fromisoformat(data['timestamp'])
    except (ValueError, TypeError):
        return jsonify({
            'success': False,
            'message': 'Invalid timestamp format'
        }), 400
    
    # Find or create recording
    recording = None
    
    # Find recordings in the last minute
    recent_recordings = list(db.recordings.find({
        'camera_id': str(data['camera_id']),
        'timestamp': {'$gte': timestamp - timedelta(minutes=1)}
    }).sort('timestamp', -1).limit(1))
    
    if recent_recordings:
        recording_data = recent_recordings[0]
        recording = Recording(recording_data)
    
    # Create detection using our model
    detection = Detection.create(
        camera_id=data['camera_id'],
        class_name=data['class_name'],
        confidence=float(data['confidence']),
        bbox_x=data['bbox'][0],
        bbox_y=data['bbox'][1],
        bbox_width=data['bbox'][2],
        bbox_height=data['bbox'][3],
        timestamp=timestamp,
        recording_id=recording.id if recording else None
    )
    
    return jsonify({
        'success': True,
        'detection_id': detection.id,
        'message': 'Detection received and processed'
    }), 201

@api_bp.route('/model_classes', methods=['GET'])
def get_model_classes():
    """Get available detection classes from a YOLOv5 model"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # If model_id is provided, use that model
        model_id = request.args.get('model_id')
        from app.models.ai_model import AIModel
        
        if model_id:
            model = AIModel.get_by_id(model_id)
            if not model:
                return jsonify({'success': False, 'message': 'Model not found'}), 404
            model_path = model.file_path
        else:
            # Otherwise use the default model
            from app.routes.main_routes import load_settings
            settings = load_settings()
            default_model_name = settings.get('detection', {}).get('default_model', 'yolov5s')
            
            # Get the default model from MongoDB
            model = AIModel.get_default_model()
            if not model:
                # Try getting by name if no default is set
                model = AIModel.get_by_name(default_model_name)
            
            if model:
                model_path = model.file_path
                logger.info(f"Using model: {model.name} at path: {model_path}")
            else:
                # Fall back to standard yolov5s
                model_path = os.path.join('models', 'yolov5s.pt')
                logger.warning(f"No model found, falling back to: {model_path}")
        
        # Get classes from the model
        import torch
        import os
        
        if os.path.exists(model_path):
            try:
                # Try to load model directly to get class names
                logger.info(f"Loading model classes from {model_path}")
                model_instance = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True)
                classes = model_instance.names
                logger.info(f"Successfully loaded {len(classes)} classes from model")
            except Exception as e:
                logger.warning(f"Error loading model classes: {str(e)}, falling back to COCO")
                # Fall back to complete standard COCO classes
                classes = {
                    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
                    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
                    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
                    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
                    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
                    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
                    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
                    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
                    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
                    55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
                    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
                    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
                    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                    79: 'toothbrush'
                }
        else:
            logger.warning(f"Model file not found: {model_path}, falling back to COCO classes")
            # Use full COCO class list when model file isn't found
            classes = {
                0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
                11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
                16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
                22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
                27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
                32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
                36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
                40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
                50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
                55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
                65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
                69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
                74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                79: 'toothbrush'
            }
        
        # Format classes as array of objects for easy use in frontend
        class_list = [{"id": class_id, "name": class_name} for class_id, class_name in classes.items()]
        
        logger.info(f"Returning {len(class_list)} model classes")
        return jsonify({
            'success': True, 
            'classes': class_list
        })
        
    except Exception as e:
        logger.error(f"Error getting model classes: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500