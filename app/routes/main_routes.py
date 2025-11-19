"""
Main routes for SmartNVR application
"""
from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user
from app.models.camera import Camera
from app.models.ai_model import AIModel

# Create blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Landing page / redirect to dashboard if logged in"""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return redirect(url_for('auth.login'))

@main_bp.route('/dashboard')
@login_required
def dashboard():
    """Dashboard route with live camera feeds"""
    cameras = Camera.get_active_cameras()
    return render_template('dashboard.html', title='Dashboard', cameras=cameras)

@main_bp.route('/playback')
@login_required
def playback():
    """Video playback route"""
    cameras = Camera.get_active_cameras()
    return render_template('playback.html', title='Playback', cameras=cameras)

@main_bp.route('/monitor')
@login_required
def monitor():
    """System monitoring route"""
    from app.utils.system_monitor import get_system_resources
    resources = get_system_resources()
    return render_template('monitor.html', title='System Monitor', resources=resources)

@main_bp.route('/profile')
@login_required
def profile():
    """User profile route"""
    return render_template('profile.html', title='My Profile')

@main_bp.route('/camera-management')
@login_required
def camera_management():
    """Camera management route"""
    cameras = Camera.get_all()
    models = AIModel.get_all()
    return render_template('camera_management.html', title='Camera Management', cameras=cameras, models=models)


@main_bp.route('/faces')
@login_required
def face_library():
    """Face management and labeling route"""
    return render_template('face_gallery.html', title='Face Library')

@main_bp.route('/add-camera', methods=['POST'])
@login_required
def add_camera():
    """Add new camera"""
    from flask import request, flash
    
    name = request.form.get('name')
    rtsp_url = request.form.get('rtsp_url')
    username = request.form.get('username')
    password = request.form.get('password')
    model_id = request.form.get('model_id')
    confidence = float(request.form.get('confidence', 0.45))
    
    # Validate inputs
    if not name or not rtsp_url:
        flash('Camera name and RTSP URL are required', 'danger')
        return redirect(url_for('main.camera_management'))
    
    # Create new camera using MongoDB model
    camera = Camera.create(
        name=name,
        rtsp_url=rtsp_url,
        username=username,
        password=password,
        model_id=model_id,
        confidence_threshold=confidence,
        is_active=True,
        recording_enabled='recording_enabled' in request.form,
        detection_enabled='detection_enabled' in request.form
    )
    
    # Start camera processor after adding the camera
    try:
        from app.utils.camera_processor import CameraManager
        manager = CameraManager.get_instance()
        if manager.start_camera(camera):
            flash(f'Camera {name} added and started successfully', 'success')
        else:
            flash(f'Camera {name} added but could not be started. Check camera URL and credentials.', 'warning')
    except Exception as e:
        flash(f'Camera {name} added but an error occurred when starting: {str(e)}', 'warning')
    
    return redirect(url_for('main.camera_management'))

@main_bp.route('/edit-camera/<camera_id>', methods=['POST'])
@login_required
def edit_camera(camera_id):
    """Edit existing camera"""
    from flask import request, flash
    
    # Find camera
    camera = Camera.get_by_id(camera_id)
    if not camera:
        flash('Camera not found', 'danger')
        return redirect(url_for('main.camera_management'))
    
    # Store old active state to check if we need to start/stop the camera
    old_is_active = camera.is_active
    
    # Update camera details
    camera.name = request.form.get('name', camera.name)
    camera.rtsp_url = request.form.get('rtsp_url', camera.rtsp_url)
    camera.username = request.form.get('username')
    camera.password = request.form.get('password')
    camera.is_active = 'enabled' in request.form
    camera.recording_enabled = 'recording_enabled' in request.form
    camera.detection_enabled = 'detection_enabled' in request.form
    
    if request.form.get('model_id'):
        camera.model_id = request.form.get('model_id')
        
    if request.form.get('confidence'):
        camera.confidence_threshold = float(request.form.get('confidence'))
    
    # Save to database
    camera.save()
    
    # Restart camera processor if needed
    try:
        from app.utils.camera_processor import CameraManager
        manager = CameraManager.get_instance()
        
        # If camera was active and still is, or settings changed, restart it
        if camera.is_active:
            # Stop camera if it's running
            manager.stop_camera(camera.id)
            
            # Start camera with new settings
            if manager.start_camera(camera):
                flash(f'Camera {camera.name} updated and restarted successfully', 'success')
            else:
                flash(f'Camera {camera.name} updated but could not be started. Check camera URL and credentials.', 'warning')
        elif old_is_active and not camera.is_active:
            # Camera was disabled, make sure it's stopped
            manager.stop_camera(camera.id)
            flash(f'Camera {camera.name} updated and stopped successfully', 'success')
        else:
            flash(f'Camera {camera.name} updated successfully', 'success')
    except Exception as e:
        flash(f'Camera {camera.name} updated but an error occurred when restarting: {str(e)}', 'warning')
    
    return redirect(url_for('main.camera_management'))

@main_bp.route('/delete-camera/<camera_id>', methods=['POST'])
@login_required
def delete_camera(camera_id):
    """Delete camera"""
    from flask import flash
    import os
    
    # Find camera
    camera = Camera.get_by_id(camera_id)
    if not camera:
        flash('Camera not found', 'danger')
        return redirect(url_for('main.camera_management'))
    
    camera_name = camera.name
    
    # Stop camera processor first if it's running
    try:
        from app.utils.camera_processor import CameraManager
        manager = CameraManager.get_instance()
        manager.stop_camera(camera_id)
    except Exception as e:
        flash(f'Warning: Error stopping camera processor: {str(e)}', 'warning')
    
    # Import models and db
    from app import db
    from app.models import Recording, Detection, ROI
    from app.routes.main_routes import get_recording_settings
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Get all recordings for this camera - we need to get the file paths before deleting them
        recordings = Recording.get_by_camera(camera_id)
        
        # Store the recording file paths so we can delete them after DB records are gone
        recording_files = []
        thumbnail_files = []
        
        for recording in recordings:
            if recording.file_path:
                # Handle both absolute and relative paths
                if os.path.isabs(recording.file_path):
                    file_path = recording.file_path
                else:
                    # If relative path, make it absolute relative to the app root
                    file_path = os.path.join(os.getcwd(), recording.file_path)
                
                if os.path.exists(file_path):
                    recording_files.append(file_path)
                else:
                    logger.warning(f"Recording file not found: {file_path}")
            
            if recording.thumbnail_path:
                # Handle both absolute and relative paths
                if os.path.isabs(recording.thumbnail_path):
                    thumbnail_path = recording.thumbnail_path
                else:
                    # If relative path, make it absolute relative to the app root
                    thumbnail_path = os.path.join(os.getcwd(), recording.thumbnail_path)
                
                if os.path.exists(thumbnail_path):
                    thumbnail_files.append(thumbnail_path)
                else:
                    logger.warning(f"Thumbnail file not found: {thumbnail_path}")
        
        logger.info(f"Deleting camera {camera_id} with {len(recordings)} recordings")
        
        # Delete related detections
        db.detections.delete_many({'camera_id': camera_id})
        logger.info(f"Deleted detections for camera {camera_id}")
        
        # Delete related recordings
        db.recordings.delete_many({'camera_id': camera_id})
        logger.info(f"Deleted recording records for camera {camera_id}")
        
        # Delete related ROIs
        db.regions_of_interest.delete_many({'camera_id': camera_id})
        logger.info(f"Deleted ROIs for camera {camera_id}")
        
        # Delete the camera
        camera.delete()
        
        # After successful DB deletion, delete physical files
        deleted_count = 0
        for file_path in recording_files:
            try:
                os.remove(file_path)
                deleted_count += 1
                logger.info(f"Deleted recording file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting recording file {file_path}: {str(e)}")
        
        for file_path in thumbnail_files:
            try:
                os.remove(file_path)
                logger.info(f"Deleted thumbnail file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting thumbnail file {file_path}: {str(e)}")
        
        # Also, try to remove the camera's recordings directory
        try:
            recording_settings = get_recording_settings()
            storage_base = recording_settings.get('storage_path', 'storage/recordings')
            
            # Check both relative and absolute paths
            camera_dir = os.path.join(storage_base, 'videos', str(camera_id))
            if not os.path.exists(camera_dir):
                camera_dir = os.path.join(os.getcwd(), storage_base, 'videos', str(camera_id))
            
            if os.path.exists(camera_dir) and os.path.isdir(camera_dir):
                # Check if directory is empty before removing
                remaining_files = os.listdir(camera_dir)
                if not remaining_files:
                    os.rmdir(camera_dir)
                    logger.info(f"Removed empty directory: {camera_dir}")
                else:
                    logger.info(f"Directory not empty, skipping removal: {camera_dir}")
            else:
                logger.info(f"Camera directory not found: {camera_dir}")
        except Exception as e:
            logger.error(f"Error removing camera directory: {str(e)}")
        
        flash(f'Camera {camera_name} and {deleted_count} recording files deleted successfully', 'success')
    except Exception as e:
        flash(f'Error deleting camera: {str(e)}', 'danger')
    
    return redirect(url_for('main.camera_management'))

@main_bp.route('/settings')
@login_required
def settings():
    """System settings route"""
    # Load current settings
    import os
    import json
    settings_file = os.path.join('config', 'settings.json')
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            
            # Ensure all required sections exist in settings
            if 'recording' not in settings:
                settings['recording'] = {
                    'retention_days': 30,
                    'storage_path': 'storage/recordings',
                    'clip_length': 60,
                    'format': 'mp4'
                }
                
            if 'notifications' not in settings:
                settings['notifications'] = {
                    'email_enabled': False,
                    'smtp_server': '',
                    'smtp_port': 587,
                    'smtp_username': '',
                    'smtp_password': '',
                    'from_email': '',
                    'email_to': ''
                }
                
            if 'system' not in settings:
                settings['system'] = {
                    'log_level': 'info'
                }
                
            if 'detection' not in settings:
                settings['detection'] = {
                    'default_confidence': 0.45,
                    'default_model': 'yolov5s',
                    'save_images': True,
                    'image_retention_days': 1,
                    'enable_gemini_ai': False,
                    'gemini_api_key': '',
                    'gemini_model': 'gemini-1.5-flash'
                }
            # Ensure Gemini AI settings exist in detection section
            elif 'detection' in settings:
                if 'enable_gemini_ai' not in settings['detection']:
                    settings['detection']['enable_gemini_ai'] = False
                if 'gemini_api_key' not in settings['detection']:
                    settings['detection']['gemini_api_key'] = ''
                if 'gemini_model' not in settings['detection']:
                    settings['detection']['gemini_model'] = 'gemini-1.5-flash'
                    
            # Get available AI models
            ai_models = AIModel.get_all()
            
            # Validate that the current default model exists
            if settings.get('detection', {}).get('default_model'):
                current_model = settings['detection']['default_model']
                model_exists = any(model.name == current_model for model in ai_models)
                if not model_exists and ai_models:
                    # Update to the first available model if current doesn't exist
                    settings['detection']['default_model'] = ai_models[0].name
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
            # Default settings if JSON is invalid
            settings = create_default_settings()
    else:
        # Default settings if file doesn't exist
        settings = create_default_settings()
        
    # Get available AI models
    ai_models = AIModel.get_all()
    
    return render_template('settings.html', title='Settings', settings=settings, ai_models=ai_models)

def create_default_settings():
    """Create default settings dictionary"""
    # Get the default model from the database, fallback to a hardcoded value if none exists
    default_model = AIModel.get_default_model()
    if not default_model:
        # Try to get the first available model
        all_models = AIModel.get_all()
        if all_models:
            default_model_name = all_models[0].name
        else:
            # Ultimate fallback - this should rarely happen after initialization
            default_model_name = 'yolov5s'
    else:
        default_model_name = default_model.name
    
    return {
        'recording': {
            'retention_days': 30,
            'storage_path': 'storage/recordings',
            'clip_length': 60,
            'format': 'mp4'
        },
        'notifications': {
            'email_enabled': False,
            'smtp_server': '',
            'smtp_port': 587,
            'smtp_username': '',
            'smtp_password': '',
            'from_email': '',
            'email_to': ''
        },
        'system': {
            'log_level': 'info'
        },
        'detection': {
            'default_confidence': 0.45,
            'default_model': default_model_name,
            'save_images': True,
            'image_retention_days': 1,
            'enable_gemini_ai': False,
            'gemini_api_key': '',
            'gemini_model': 'gemini-1.5-flash',
            'enable_face_detection': False,
            'face_detection_confidence': 0.5,
            'face_detection_model': 'MNET_V2',
            'face_detection_provider': 'auto',
            'face_detection_max_faces': 10,
            'enable_face_recognition': False,
            'face_recognition_threshold': 0.62,
            'face_recognition_auto_create': True
        }
    }

def load_settings():
    """Load settings from file for other modules to access"""
    import os
    import json
    
    settings_file = os.path.join('config', 'settings.json')
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            return settings
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
    
    return create_default_settings()

# Make the settings available to other parts of the application
def get_recording_settings():
    """Get recording settings for camera processor"""
    settings = load_settings()
    return settings.get('recording', {})

def get_detection_settings():
    """Get detection settings for camera processor"""
    settings = load_settings()
    return settings.get('detection', {})

@main_bp.route('/save-settings', methods=['POST'])
@login_required
def save_settings():
    """Save system settings"""
    from flask import request, flash
    import os
    import json
    
    # Get form data
    settings = {
        'recording': {
            'retention_days': int(request.form.get('retention_days', 30)),
            'storage_path': request.form.get('storage_path', 'storage/recordings'),
            'clip_length': int(request.form.get('clip_length', 60)),
            'format': request.form.get('format', 'mp4')
        },
        'notifications': {
            'email_enabled': 'email_enabled' in request.form,
            'smtp_server': request.form.get('smtp_server', ''),
            'smtp_port': int(request.form.get('smtp_port', 587)),
            'smtp_username': request.form.get('smtp_username', ''),
            'from_email': request.form.get('from_email', ''),
            'email_to': request.form.get('email_to', ''),
            'push_enabled': 'push_enabled' in request.form
        },
        'system': {
            'log_level': request.form.get('log_level', 'info')
        },
        'detection': {
            'default_confidence': float(request.form.get('default_confidence', 0.45)),
            'default_model': request.form.get('default_model', 'yolov5s'),
            'save_images': 'save_detection_images' in request.form,
            'image_retention_days': int(request.form.get('image_retention_days', 7)),
            # Add Gemini AI settings
            'enable_gemini_ai': 'enable_gemini_ai' in request.form,
            'gemini_api_key': request.form.get('gemini_api_key', ''),
            'gemini_model': request.form.get('gemini_model', 'gemini-1.5-flash'),
            # Face detection settings
            'enable_face_detection': 'enable_face_detection' in request.form,
            'face_detection_confidence': float(request.form.get('face_detection_confidence', 0.5)),
            'face_detection_model': request.form.get('face_detection_model', 'MNET_V2'),
            'face_detection_provider': request.form.get('face_detection_provider', 'auto'),
            'face_detection_max_faces': int(request.form.get('face_detection_max_faces', 10)),
            'enable_face_recognition': 'enable_face_recognition' in request.form,
            'face_recognition_threshold': float(request.form.get('face_recognition_threshold', 0.62)),
            'face_recognition_auto_create': 'face_recognition_auto_create' in request.form
        }
    }
    
    # Only update password if provided
    if request.form.get('smtp_password'):
        settings['notifications']['smtp_password'] = request.form.get('smtp_password')
    else:
        # Preserve existing password if it exists
        settings_file = os.path.join('config', 'settings.json')
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    old_settings = json.load(f)
                    if 'notifications' in old_settings and 'smtp_password' in old_settings['notifications']:
                        settings['notifications']['smtp_password'] = old_settings['notifications']['smtp_password']
                    
                    # Preserve existing Gemini API key if not provided
                    if not request.form.get('gemini_api_key') and 'detection' in old_settings and 'gemini_api_key' in old_settings['detection']:
                        settings['detection']['gemini_api_key'] = old_settings['detection']['gemini_api_key']
            except:
                pass
    
    # Ensure config directory exists
    os.makedirs('config', exist_ok=True)
    
    # Save settings to JSON file
    with open(os.path.join('config', 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=4)
    
    flash('Settings saved successfully', 'success')
    return redirect(url_for('main.settings'))

@main_bp.route('/update-profile', methods=['POST'])
@login_required
def update_profile():
    """Update user profile information"""
    from flask import request, flash
    from werkzeug.security import check_password_hash
    
    # Get form data
    email = request.form.get('email')
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    # Validate email
    if not email:
        flash('Email is required', 'danger')
        return redirect(url_for('main.profile'))
    
    # Check if another user already has this email
    from app.models.user import User
    existing_user = User.get_by_email(email)
    if existing_user and existing_user.id != current_user.id:
        flash('This email is already in use by another account', 'danger')
        return redirect(url_for('main.profile'))
    
    # Update email if changed
    if email != current_user.email:
        current_user.email = email
        current_user.save()
        flash('Email updated successfully', 'success')
    
    # Handle password change if requested
    if new_password:
        # Verify current password
        if not current_password or not check_password_hash(current_user.password_hash, current_password):
            flash('Current password is incorrect', 'danger')
            return redirect(url_for('main.profile'))
        
        # Validate new password
        if new_password != confirm_password:
            flash('New passwords do not match', 'danger')
            return redirect(url_for('main.profile'))
        
        # Update password
        current_user.set_password(new_password)
        current_user.save()
        flash('Password updated successfully', 'success')
    
    return redirect(url_for('main.profile'))

@main_bp.route('/update-notification-preferences', methods=['POST'])
@login_required
def update_notification_preferences():
    """Update user notification preferences"""
    from flask import request, flash
    from app import db
    
    # Get preferences from form
    email_notifications = 'email_notifications' in request.form
    push_notifications = 'push_notifications' in request.form
    
    # Initialize preferences if they don't exist
    if not hasattr(current_user, 'preferences') or current_user.preferences is None:
        current_user.preferences = {}
    
    # Update preferences
    current_user.preferences['email_notifications'] = email_notifications
    current_user.preferences['push_notifications'] = push_notifications
    
    # Update in database
    db.users.update_one(
        {'_id': current_user._id},
        {'$set': {'preferences': current_user.preferences}}
    )
    
    flash('Notification preferences updated', 'success')
    return redirect(url_for('main.profile'))