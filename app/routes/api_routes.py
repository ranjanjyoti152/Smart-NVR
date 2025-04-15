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
        camera = Camera.query.get(camera_id)
        
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
            recent_recording = Recording.query.filter(
                Recording.camera_id == camera_id,
                Recording.timestamp >= detection_timestamp - timedelta(minutes=1)
            ).order_by(Recording.timestamp.desc()).first()
            
            recording = recent_recording
        
        # Process each detection
        new_detections = []
        for det_data in detections_data:
            # Debug ROI ID to track if it's being properly passed
            roi_id = det_data.get('roi_id')
            logger.debug(f"Processing detection with ROI ID: {roi_id}")
            
            # Create detection object
            detection = Detection(
                camera_id=camera_id,
                recording_id=recording.id if recording else None,
                roi_id=roi_id,
                timestamp=det_data.get('timestamp', datetime.now()) if isinstance(det_data.get('timestamp'), datetime) else datetime.now(),
                class_name=det_data.get('class_name', 'unknown'),
                confidence=det_data.get('confidence', 0.0),
                bbox_x=det_data.get('bbox_x', 0),
                bbox_y=det_data.get('bbox_y', 0),
                bbox_width=det_data.get('bbox_width', 0),
                bbox_height=det_data.get('bbox_height', 0),
                image_path=det_data.get('image_path'),
                video_path=det_data.get('video_path'),
                notified=False
            )
            
            db.session.add(detection)
            new_detections.append(detection)
        
        # Commit all detections to database first
        db.session.commit()
        
        # Process email notifications for new detections
        for detection in new_detections:
            try:
                # Check if this detection is in an ROI with email notifications enabled
                roi = None
                
                if detection.roi_id:
                    roi = ROI.query.get(detection.roi_id)
                    
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
    cameras = Camera.query.filter_by(is_active=True).all()
    return jsonify({
        'success': True,
        'cameras': [camera.to_dict() for camera in cameras]
    })

@api_bp.route('/cameras/<int:camera_id>')
@login_required
def get_camera(camera_id):
    """Get camera details"""
    camera = Camera.query.get_or_404(camera_id)
    return jsonify({
        'success': True,
        'camera': camera.to_dict()
    })

@api_bp.route('/cameras/<int:camera_id>/frame')
@login_required
def get_camera_frame(camera_id):
    """Get a single frame from camera as JPEG image"""
    from app.utils.camera_processor import CameraManager
    import cv2
    
    camera = Camera.query.get_or_404(camera_id)
    
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

@api_bp.route('/cameras/<int:camera_id>/stream')
@login_required
def get_camera_stream(camera_id):
    """Get camera stream (MJPEG)"""
    from app.utils.camera_processor import CameraManager
    import cv2
    
    camera = Camera.query.get_or_404(camera_id)
    
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

@api_bp.route('/cameras/<int:camera_id>/snapshot')
@login_required
def get_camera_snapshot(camera_id):
    """Get current camera snapshot"""
    # This is just an alias for the frame endpoint
    return get_camera_frame(camera_id)

@api_bp.route('/cameras/<int:camera_id>/roi', methods=['GET'])
@login_required
def get_camera_roi(camera_id):
    """Get camera regions of interest"""
    camera = Camera.query.get_or_404(camera_id)
    rois = ROI.query.filter_by(camera_id=camera.id).all()
    
    return jsonify({
        'success': True,
        'roi': [roi.to_dict() for roi in rois]
    })

@api_bp.route('/cameras/<int:camera_id>/roi', methods=['POST'])
@login_required
def create_camera_roi(camera_id):
    """Create new region of interest for camera"""
    camera = Camera.query.get_or_404(camera_id)
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
    roi = ROI(
        camera_id=camera.id,
        name=data['name'],
        coordinates=json.dumps(data['coordinates']),
        detection_classes=json.dumps(data.get('detection_classes', [])),
        is_active=data.get('is_active', True),
        email_notifications=data.get('email_notifications', False)
    )
    
    db.session.add(roi)
    db.session.commit()
    
    # Reload ROIs for the camera
    try:
        from app.utils.camera_processor import CameraManager
        manager = CameraManager.get_instance()
        manager.reload_rois(camera_id)
    except Exception as e:
        logger.error(f"Failed to reload ROIs: {str(e)}")
    
    return jsonify({
        'success': True,
        'roi': roi.to_dict()
    }), 201

@api_bp.route('/cameras/<int:camera_id>/roi/<int:roi_id>', methods=['PUT'])
@login_required
def update_camera_roi(camera_id, roi_id):
    """Update region of interest for camera"""
    roi = ROI.query.filter_by(id=roi_id, camera_id=camera_id).first_or_404()
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
        roi.coordinates = json.dumps(data['coordinates'])
    if 'detection_classes' in data:
        roi.detection_classes = json.dumps(data['detection_classes'])
    if 'is_active' in data:
        roi.is_active = data['is_active']
    if 'email_notifications' in data:
        roi.email_notifications = data['email_notifications']
    
    db.session.commit()
    
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

@api_bp.route('/cameras/<int:camera_id>/roi/<int:roi_id>', methods=['DELETE'])
@login_required
def delete_camera_roi(camera_id, roi_id):
    """Delete region of interest for camera"""
    roi = ROI.query.filter_by(id=roi_id, camera_id=camera_id).first_or_404()
    
    db.session.delete(roi)
    db.session.commit()
    
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

@api_bp.route('/cameras/<int:camera_id>/detections/latest')
@login_required
def get_latest_camera_detections(camera_id):
    """Get latest detections for a specific camera"""
    # Verify camera exists
    camera = Camera.query.get_or_404(camera_id)
    
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
        detections = Detection.query.filter_by(camera_id=camera_id).order_by(
            Detection.timestamp.desc()
        ).limit(20).all()
        
        # If no direct camera detections, try via recordings
        if not detections:
            recording_ids = db.session.query(Recording.id).filter_by(camera_id=camera_id).all()
            recording_ids = [r[0] for r in recording_ids]
            
            if recording_ids:
                detections = Detection.query.filter(
                    Detection.recording_id.in_(recording_ids)
                ).order_by(
                    Detection.timestamp.desc()
                ).limit(20).all()
    except Exception as e:
        print(f"Error getting database detections: {str(e)}")
        detections = []
    
    # Convert detections to list of dicts with coordinates
    results = []
    for det in detections:
        # Skip detections without coordinates
        if not all(hasattr(det, attr) for attr in ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']):
            continue
        
        results.append({
            'id': det.id,
            'class_name': det.object_class,  # Using object_class from model
            'confidence': det.confidence,
            'coordinates': {
                'x_min': float(det.bbox_x),
                'y_min': float(det.bbox_y),
                'x_max': float(det.bbox_x) + float(det.bbox_w),
                'y_max': float(det.bbox_y) + float(det.bbox_h)
            },
            'timestamp': det.timestamp.isoformat() if det.timestamp else None
        })
    
    return jsonify(results)

@api_bp.route('/cameras/<int:camera_id>/recordings')
@login_required
def get_camera_recordings(camera_id):
    """Get recordings for a specific camera with filters"""
    # Verify camera exists
    camera = Camera.query.get_or_404(camera_id)
    
    # Get query parameters
    date = request.args.get('date')  # Format: YYYY-MM-DD
    events_only = request.args.get('events_only', 'false').lower() == 'true'
    object_type = request.args.get('object_type', '')
    
    # Build query
    query = Recording.query.filter_by(camera_id=camera_id)
    
    # Filter by date
    if date:
        try:
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            next_day = date_obj + timedelta(days=1)
            query = query.filter(Recording.timestamp >= date_obj, 
                                Recording.timestamp < next_day)
        except ValueError:
            return jsonify({
                'success': False,
                'message': 'Invalid date format. Use YYYY-MM-DD'
            }), 400
    
    # Filter by recording type (events only)
    if events_only:
        # Join with detections to find recordings with events
        detections_subquery = db.session.query(Detection.recording_id).distinct().subquery()
        query = query.filter(Recording.id.in_(detections_subquery))
    
    # Filter by object type
    if object_type:
        # Join with detections to filter by object type
        recording_ids = db.session.query(Detection.recording_id).filter(
            Detection.class_name == object_type
        ).distinct().all()
        recording_ids = [r[0] for r in recording_ids]
        query = query.filter(Recording.id.in_(recording_ids))
    
    # Order by timestamp
    query = query.order_by(Recording.timestamp.desc())
    
    # Execute query
    recordings = query.all()
    
    # Format results
    results = []
    for rec in recordings:
        # Get detections for this recording if any
        detections = []
        for det in rec.detections:
            detections.append({
                'id': det.id,
                'class_name': det.class_name,
                'confidence': det.confidence,
                'timestamp': det.timestamp.isoformat() if det.timestamp else None
            })
            
        results.append({
            'id': rec.id,
            'timestamp': rec.timestamp.isoformat() if rec.timestamp else None,
            'duration': rec.duration,
            'file_path': rec.file_path,
            'thumbnail_path': rec.thumbnail_path,
            'recording_type': rec.recording_type,
            'is_flagged': rec.is_flagged,
            'file_size': rec.file_size,
            'video_url': f'/api/recordings/{rec.id}/video',
            'thumbnail_url': f'/api/recordings/{rec.id}/thumbnail',
            'detections': detections
        })
    
    return jsonify(results)

# --- Recordings API Endpoints ---

@api_bp.route('/recordings')
@login_required
def get_recordings():
    """Get recordings with filters"""
    # Get query parameters
    camera_id = request.args.get('camera_id', type=int)
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    recording_type = request.args.get('type')
    has_detections = request.args.get('has_detections', type=bool)
    
    # Build query
    query = Recording.query
    
    if camera_id:
        query = query.filter_by(camera_id=camera_id)
    
    if date_from:
        try:
            date_from = datetime.strptime(date_from, '%Y-%m-%d')
            query = query.filter(Recording.timestamp >= date_from)
        except ValueError:
            return jsonify({
                'success': False,
                'message': 'Invalid date_from format. Use YYYY-MM-DD'
            }), 400
    
    if date_to:
        try:
            date_to = datetime.strptime(date_to, '%Y-%m-%d') + timedelta(days=1)
            query = query.filter(Recording.timestamp < date_to)
        except ValueError:
            return jsonify({
                'success': False,
                'message': 'Invalid date_to format. Use YYYY-MM-DD'
            }), 400
    
    if recording_type:
        query = query.filter_by(recording_type=recording_type)
    
    if has_detections is not None:
        if has_detections:
            # Recordings with detections
            query = query.join(Detection, Recording.id == Detection.recording_id)
        else:
            # Recordings without detections
            # This is more complex - need to find IDs not in the join
            detection_recording_ids = db.session.query(Detection.recording_id).distinct()
            query = query.filter(~Recording.id.in_(detection_recording_ids))
    
    # Paginate results
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    pagination = query.order_by(Recording.timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'success': True,
        'recordings': [recording.to_dict() for recording in pagination.items],
        'pagination': {
            'total': pagination.total,
            'pages': pagination.pages,
            'page': pagination.page,
            'per_page': pagination.per_page
        }
    })

@api_bp.route('/recordings/<int:recording_id>')
@login_required
def get_recording(recording_id):
    """Get recording details"""
    recording = Recording.query.get_or_404(recording_id)
    
    return jsonify({
        'success': True,
        'recording': recording.to_dict()
    })

@api_bp.route('/recordings/<int:recording_id>/video')
@login_required
def get_recording_video(recording_id):
    """Stream recording video"""
    recording = Recording.query.get_or_404(recording_id)
    
    if not os.path.exists(recording.file_path):
        abort(404, description="Recording file not found")
    
    # Stream the video file
    return send_file(recording.file_path, conditional=True)

@api_bp.route('/recordings/<int:recording_id>/thumbnail')
@login_required
def get_recording_thumbnail(recording_id):
    """Get recording thumbnail"""
    recording = Recording.query.get_or_404(recording_id)
    
    if recording.thumbnail_path and os.path.exists(recording.thumbnail_path):
        return send_file(recording.thumbnail_path, mimetype='image/jpeg')
    
    # Return default thumbnail if none exists
    return send_file('static/img/no-thumbnail.png', mimetype='image/jpeg')

@api_bp.route('/recordings/<int:recording_id>/download')
@login_required
def download_recording(recording_id):
    """Download recording file"""
    recording = Recording.query.get_or_404(recording_id)
    
    if not os.path.exists(recording.file_path):
        abort(404, description="Recording file not found")
    
    # Generate a download filename based on camera name and timestamp
    camera = Camera.query.get(recording.camera_id)
    camera_name = camera.name if camera else f"camera-{recording.camera_id}"
    timestamp = recording.timestamp.strftime('%Y%m%d-%H%M%S')
    filename = f"{camera_name}-{timestamp}.mp4"
    
    return send_file(
        recording.file_path,
        as_attachment=True,
        download_name=filename,
        mimetype='video/mp4'
    )

@api_bp.route('/recordings/<int:recording_id>/flag', methods=['POST'])
@login_required
def flag_recording(recording_id):
    """Flag recording as important"""
    recording = Recording.query.get_or_404(recording_id)
    recording.is_flagged = True
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Recording flagged successfully'
    })

@api_bp.route('/recordings/<int:recording_id>/unflag', methods=['POST'])
@login_required
def unflag_recording(recording_id):
    """Remove flag from recording"""
    recording = Recording.query.get_or_404(recording_id)
    recording.is_flagged = False
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': 'Recording unflagged successfully'
    })

@api_bp.route('/recordings/<int:recording_id>', methods=['DELETE'])
@login_required
def delete_recording(recording_id):
    """Delete recording and file"""
    recording = Recording.query.get_or_404(recording_id)
    
    # Delete file if it exists
    if recording.file_path and os.path.exists(recording.file_path):
        os.remove(recording.file_path)
    
    # Delete thumbnail if it exists
    if recording.thumbnail_path and os.path.exists(recording.thumbnail_path):
        os.remove(recording.thumbnail_path)
    
    # Delete from database
    db.session.delete(recording)
    db.session.commit()
    
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
    camera_id = request.args.get('camera_id', type=int)
    class_name = request.args.get('class_name')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    
    # Build query
    query = Detection.query
    
    if camera_id:
        # Need to join with recordings to filter by camera
        recordings = Recording.query.filter_by(camera_id=camera_id).all()
        recording_ids = [r.id for r in recordings]
        query = query.filter(Detection.recording_id.in_(recording_ids))
    
    if class_name:
        query = query.filter_by(class_name=class_name)
    
    if date_from:
        try:
            date_from = datetime.strptime(date_from, '%Y-%m-%d')
            query = query.filter(Detection.timestamp >= date_from)
        except ValueError:
            return jsonify({
                'success': False,
                'message': 'Invalid date_from format. Use YYYY-MM-DD'
            }), 400
    
    if date_to:
        try:
            date_to = datetime.strptime(date_to, '%Y-%m-%d') + timedelta(days=1)
            query = query.filter(Detection.timestamp < date_to)
        except ValueError:
            return jsonify({
                'success': False,
                'message': 'Invalid date_to format. Use YYYY-MM-DD'
            }), 400
    
    # Paginate results
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    pagination = query.order_by(Detection.timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'success': True,
        'detections': [detection.to_dict() for detection in pagination.items],
        'pagination': {
            'total': pagination.total,
            'pages': pagination.pages,
            'page': pagination.page,
            'per_page': pagination.per_page
        }
    })

@api_bp.route('/detections/summary')
@login_required
def get_detection_summary():
    """Get summary of detections"""
    # Get time range parameters
    days = request.args.get('days', 7, type=int)
    start_date = datetime.now() - timedelta(days=days)
    
    # Get detection counts by class
    class_counts = db.session.query(
        Detection.class_name,
        db.func.count(Detection.id)
    ).filter(
        Detection.timestamp >= start_date
    ).group_by(
        Detection.class_name
    ).all()
    
    # Get detection counts by camera
    camera_counts = db.session.query(
        Camera.id,
        Camera.name,
        db.func.count(Detection.id)
    ).join(
        Recording, Camera.id == Recording.camera_id
    ).join(
        Detection, Recording.id == Detection.recording_id
    ).filter(
        Detection.timestamp >= start_date
    ).group_by(
        Camera.id
    ).all()
    
    # Format results
    class_summary = {name: count for name, count in class_counts}
    camera_summary = {name: count for id, name, count in camera_counts}
    
    return jsonify({
        'success': True,
        'class_summary': class_summary,
        'camera_summary': camera_summary,
        'total': sum(count for _, count in class_counts),
        'time_range': {
            'days': days,
            'start_date': start_date.strftime('%Y-%m-%d')
        }
    })

@api_bp.route('/detections/<int:detection_id>')
@login_required
def get_detection(detection_id):
    """Get detection details"""
    detection = Detection.query.get_or_404(detection_id)
    
    return jsonify({
        'success': True,
        'detection': detection.to_dict()
    })

@api_bp.route('/detections/<int:detection_id>/image')
@login_required
def get_detection_image(detection_id):
    """Get detection image"""
    detection = Detection.query.get_or_404(detection_id)
    
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
    # Total recordings size
    total_size = db.session.query(db.func.sum(Recording.file_size)).scalar() or 0
    
    # Size by camera
    camera_sizes = db.session.query(
        Camera.id,
        Camera.name,
        db.func.sum(Recording.file_size)
    ).join(
        Recording, Camera.id == Recording.camera_id
    ).group_by(
        Camera.id
    ).all()
    
    # Format camera sizes
    camera_storage = []
    for id, name, size in camera_sizes:
        camera_storage.append({
            'id': id,
            'name': name,
            'size': size or 0
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
    camera = Camera.query.get(data['camera_id'])
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
    
    # TODO: Find existing recording or create a new one based on the timestamp
    # This is a placeholder implementation
    
    # Create detection
    detection = Detection(
        recording_id=recording.id if recording else None,
        timestamp=timestamp,
        class_name=data['class_name'],
        confidence=float(data['confidence']),
        bbox_x=data['bbox'][0],
        bbox_y=data['bbox'][1],
        bbox_width=data['bbox'][2],
        bbox_height=data['bbox'][3],
    )
    
    db.session.add(detection)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'detection_id': detection.id,
        'message': 'Detection received and processed'
    }), 201