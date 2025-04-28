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
                    'default_model': 'yolov5s'
                }
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
    return {
        'recording': {
            'retention_days': 30,
            'storage_path': 'storage/recordings',
            'clip_length': 60,
            'format': 'mp4',
            'use_separate_db_storage': False,
            'db_storage_path': '/var/lib/mongodb',
            'db_storage_size': 20  # GB
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
            'default_model': 'yolov5s',
            'save_images': True,
            'image_retention_days': 1
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
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Get form data
    settings = {
        'recording': {
            'retention_days': int(request.form.get('retention_days', 30)),
            'storage_path': request.form.get('storage_path', 'storage/recordings'),
            'clip_length': int(request.form.get('clip_length', 60)),
            'format': request.form.get('format', 'mp4'),
            'use_separate_db_storage': 'use_separate_db_storage' in request.form
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
            'image_retention_days': int(request.form.get('image_retention_days', 7))
        }
    }
    
    # Handle database storage settings
    if settings['recording']['use_separate_db_storage']:
        settings['recording']['db_storage_path'] = request.form.get('db_storage_path', '/var/lib/mongodb')
        settings['recording']['db_storage_size'] = int(request.form.get('db_storage_size', 20))
        
        # Create directory if it doesn't exist
        db_storage_path = settings['recording']['db_storage_path']
        try:
            if not os.path.exists(db_storage_path):
                os.makedirs(db_storage_path, exist_ok=True)
                logger.info(f"Created database storage directory: {db_storage_path}")
                
                # Try to set appropriate permissions
                try:
                    import subprocess
                    subprocess.run(['chown', 'mongodb:mongodb', db_storage_path], check=False)
                    subprocess.run(['chmod', '770', db_storage_path], check=False)
                    logger.info(f"Set permissions on database storage directory: {db_storage_path}")
                except Exception as e:
                    logger.warning(f"Failed to set permissions on database directory: {str(e)}")
                    
        except Exception as e:
            flash(f'Warning: Could not create database storage directory: {str(e)}. MongoDB may not be able to use this location.', 'warning')
            logger.error(f"Failed to create database storage directory: {str(e)}")
    
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
            except:
                pass
    
    # Ensure config directory exists
    os.makedirs('config', exist_ok=True)
    
    # Save settings to JSON file
    with open(os.path.join('config', 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=4)
    
    # Check if we need to update MongoDB configuration
    if settings['recording']['use_separate_db_storage']:
        try:
            # Create a helper script to update MongoDB configuration
            create_mongodb_config_helper(settings['recording']['db_storage_path'])
            flash('Settings saved successfully. To apply database storage changes, you will need to run the MongoDB configuration helper script as root.', 'info')
        except Exception as e:
            logger.error(f"Failed to create MongoDB configuration helper: {str(e)}")
            flash(f'Settings saved successfully, but could not create MongoDB helper script: {str(e)}', 'warning')
    else:
        flash('Settings saved successfully', 'success')
    
    return redirect(url_for('main.settings'))

def create_mongodb_config_helper(db_path):
    """Create a helper script to update MongoDB configuration"""
    import os
    
    # Create a shell script to update MongoDB configuration
    script_path = os.path.join('config', 'update_mongodb_storage.sh')
    
    with open(script_path, 'w') as f:
        f.write(f"""#!/bin/bash
# MongoDB storage path configuration helper
# Created by Smart-NVR

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "This script must be run as root."
  exit 1
fi

# Create the MongoDB directory if it doesn't exist
if [ ! -d "{db_path}" ]; then
  echo "Creating MongoDB directory at {db_path}..."
  mkdir -p "{db_path}"
fi

# Set the correct ownership and permissions
echo "Setting permissions for {db_path}..."
chown -R mongodb:mongodb "{db_path}"
chmod -R 770 "{db_path}"

# Update the MongoDB configuration
CONFIG_FILE="/etc/mongodb.conf"
if [ -f "$CONFIG_FILE" ]; then
  echo "Backing up existing MongoDB configuration..."
  cp "$CONFIG_FILE" "$CONFIG_FILE.backup"
  
  echo "Updating MongoDB configuration..."
  if grep -q "^dbpath" "$CONFIG_FILE"; then
    # Replace the existing dbpath
    sed -i "s|^dbpath.*|dbpath={db_path}|g" "$CONFIG_FILE"
  else
    # Add the dbpath if it doesn't exist
    echo "dbpath={db_path}" >> "$CONFIG_FILE"
  fi
  
  echo "Restarting MongoDB service..."
  systemctl restart mongodb
  
  echo "MongoDB configuration updated successfully."
  echo "Data directory set to: {db_path}"
else
  echo "MongoDB configuration file not found at $CONFIG_FILE."
  echo "You may need to manually update your MongoDB configuration."
fi
""")
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    return script_path

@main_bp.route('/database-storage-info')
@login_required
def database_storage_info():
    """Display database storage information and configuration instructions"""
    import os
    import subprocess
    from flask import flash
    
    settings = load_settings()
    recording_settings = settings.get('recording', {})
    
    # Check if the MongoDB config helper script exists
    config_script_path = os.path.join('config', 'update_mongodb_storage.sh')
    script_exists = os.path.exists(config_script_path)
    
    # Get current MongoDB data path
    mongo_data_path = "/var/lib/mongodb"  # Default path
    try:
        if os.path.exists("/etc/mongodb.conf"):
            with open("/etc/mongodb.conf", "r") as f:
                for line in f:
                    if line.strip().startswith("dbpath="):
                        mongo_data_path = line.strip().split("=", 1)[1]
                        break
    except Exception as e:
        flash(f"Error reading MongoDB configuration: {str(e)}", "warning")
    
    # Check MongoDB storage usage
    mongo_storage_size = "Unknown"
    mongo_storage_used = "Unknown"
    mongo_storage_available = "Unknown"
    mongo_storage_percent = "Unknown"
    
    try:
        if os.path.exists(mongo_data_path):
            import shutil
            usage = shutil.disk_usage(mongo_data_path)
            
            # Convert to human-readable format
            def format_size(size_bytes):
                for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                    if size_bytes < 1024 or unit == 'TB':
                        return f"{size_bytes:.2f} {unit}"
                    size_bytes /= 1024
            
            mongo_storage_size = format_size(usage.total)
            mongo_storage_used = format_size(usage.used)
            mongo_storage_available = format_size(usage.free)
            mongo_storage_percent = f"{(usage.used / usage.total * 100):.1f}%"
    except Exception as e:
        flash(f"Error checking MongoDB storage usage: {str(e)}", "warning")
    
    # Get MongoDB service status
    mongo_status = "Unknown"
    try:
        result = subprocess.run(["systemctl", "is-active", "mongodb"], 
                               capture_output=True, text=True, check=False)
        mongo_status = result.stdout.strip()
    except Exception:
        pass
    
    # Render template with database storage information
    return render_template(
        'database_storage.html',
        title='Database Storage Configuration',
        script_exists=script_exists,
        script_path=config_script_path,
        mongo_data_path=mongo_data_path,
        configured_path=recording_settings.get('db_storage_path', '/var/lib/mongodb'),
        mongo_storage_size=mongo_storage_size,
        mongo_storage_used=mongo_storage_used,
        mongo_storage_available=mongo_storage_available, 
        mongo_storage_percent=mongo_storage_percent,
        mongo_status=mongo_status,
        use_separate_storage=recording_settings.get('use_separate_db_storage', False)
    )