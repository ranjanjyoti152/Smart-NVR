from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
import os
import json
import torch
import shutil
from werkzeug.utils import secure_filename
import requests # Add requests for downloading
from ultralytics import YOLO # Import YOLO for verification
from datetime import datetime

from app import db
from app.models.user import User # Fix import for User model
from app.models.camera import Camera
from app.models.ai_model import AIModel
from app.utils.decorators import admin_required

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/')
@login_required
@admin_required
def admin_index():
    """Admin dashboard"""
    return render_template('admin/dashboard.html', title='Admin Dashboard')

@admin_bp.route('/users')
@login_required
@admin_required
def user_management():
    """User management page"""
    users = User.get_all()
    return render_template('admin/users.html', title='User Management', users=users)

@admin_bp.route('/users/create', methods=['POST'])
@login_required
@admin_required
def create_user():
    """Create a new user"""
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    is_admin = True if request.form.get('is_admin') else False
    
    # Check if user already exists
    if User.get_by_username(username):
        flash('Username already exists.', 'danger')
        return redirect(url_for('admin.user_management'))
    
    if User.get_by_email(email):
        flash('Email already registered.', 'danger')
        return redirect(url_for('admin.user_management'))
    
    # Create new user
    User.create(username, email, password, is_admin=is_admin)
    
    flash('User created successfully.', 'success')
    return redirect(url_for('admin.user_management'))

@admin_bp.route('/users/<user_id>', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_user(user_id):
    """Edit a user"""
    user = User.get_by_id(user_id)
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('admin.user_management'))
    
    # Prevent admins from editing themselves (to avoid removing their own admin rights)
    if user.id == current_user.id:
        flash('You cannot edit your own account from here. Use profile page instead.', 'warning')
        return redirect(url_for('admin.user_management'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        is_admin = True if request.form.get('is_admin') else False
        password = request.form.get('password')
        
        # Update user
        if email != user.email:
            if User.get_by_email(email):
                flash('Email already registered.', 'danger')
                return redirect(url_for('admin.edit_user', user_id=user_id))
            # Update the email in MongoDB
            db.users.update_one({'_id': user._id}, {'$set': {'email': email}})
            user.email = email
        
        # Update admin status
        db.users.update_one({'_id': user._id}, {'$set': {'is_admin': is_admin}})
        user.is_admin = is_admin
        
        # Update password if provided
        if password:
            user.set_password(password)
        
        flash('User updated successfully.', 'success')
        return redirect(url_for('admin.user_management'))
    
    return render_template('admin/edit_user.html', title='Edit User', user=user)

@admin_bp.route('/users/<user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    """Delete a user"""
    user = User.get_by_id(user_id)
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('admin.user_management'))
    
    # Prevent admins from deleting themselves
    if user.id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('admin.user_management'))
    
    # Delete from MongoDB
    db.users.delete_one({'_id': user._id})
    
    flash('User deleted successfully.', 'success')
    return redirect(url_for('admin.user_management'))

# API endpoints
@admin_bp.route('/api/users', methods=['GET'])
@login_required
@admin_required
def api_get_users():
    """API endpoint to get all users"""
    users = User.get_all()
    return jsonify([user.to_dict(include_email=True) for user in users])

@admin_bp.route('/models')
@login_required
@admin_required
def manage_models():
    """AI model management page"""
    models = AIModel.get_all()
    return render_template('admin/models.html', title='AI Model Management', models=models)

@admin_bp.route('/models/add', methods=['POST'])
@login_required
@admin_required
def add_model():
    """Add a new AI model reference"""
    file_path = request.form.get('file_path')
    name = request.form.get('name')
    description = request.form.get('description', '')
    is_custom = True if request.form.get('is_custom') else False
    is_default = True if request.form.get('is_default') else False
    
    # Validate inputs
    if not all([file_path, name]):
        flash('Name and file path are required.', 'danger')
        return redirect(url_for('admin.manage_models'))
    
    # Check if file exists
    if not os.path.exists(file_path):
        flash(f'Model file not found at {file_path}. Please ensure the file exists.', 'danger')
        return redirect(url_for('admin.manage_models'))
    
    # Check if model name already exists
    if AIModel.get_by_name(name):
        flash('A model with this name already exists.', 'danger')
        return redirect(url_for('admin.manage_models'))
    
    # Create new model
    AIModel.create(name, file_path, description=description, is_default=is_default, is_custom=is_custom)
    
    flash('Model added successfully.', 'success')
    return redirect(url_for('admin.manage_models'))

@admin_bp.route('/models/upload', methods=['POST'])
@login_required
@admin_required
def upload_model():
    """Upload a new custom YOLO model (.pt)"""
    try:
        # Check if the post request has the file part
        if 'model_file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file part in the request'
            }), 400
            
        model_file = request.files['model_file']
        
        if model_file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            }), 400
        
        name = request.form.get('name')
        description = request.form.get('description', '')
        is_default = True if request.form.get('is_default') else False
        
        # Validate inputs
        if not name:
            return jsonify({
                'success': False,
                'message': 'Model name is required'
            }), 400
        
        # Check if model name already exists
        if AIModel.get_by_name(name):
            return jsonify({
                'success': False,
                'message': 'A model with this name already exists'
            }), 400
        
        # Print debugging information
        print(f"Received file: {model_file.filename}, size: {model_file.content_length}")
        
        # Sanitize filename and save model file
        filename = secure_filename(model_file.filename)
        if not filename.endswith('.pt'):
             # Allow only .pt files
             return jsonify({
                 'success': False,
                 'message': 'Invalid file type. Only .pt files are allowed.'
             }), 400

        models_dir = os.path.join('storage', 'models') # Save custom models to storage/models
        os.makedirs(models_dir, exist_ok=True)
        file_path = os.path.join(models_dir, filename)

        # If file exists, add timestamp to filename
        if os.path.exists(file_path):
            import time
            timestamp = int(time.time())
            name_parts = filename.rsplit('.', 1)
            filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
            file_path = os.path.join(models_dir, filename)

        print(f"Saving file to: {file_path}")
        model_file.save(file_path)
        print(f"File saved successfully")

        # Verify it's a valid YOLO model using ultralytics
        try:
            print(f"Attempting to verify model file with Ultralytics YOLO")
            # Try loading the model to verify it
            _ = YOLO(file_path)
            print(f"Model verified successfully with Ultralytics YOLO")
        except (ModuleNotFoundError, ImportError) as import_err:
            # Allow upload if the error is specifically about missing 'models.yolo' or similar legacy structure
            # This indicates an older YOLOv5 model structure
            err_msg = str(import_err).lower()
            if 'models.yolo' in err_msg or 'models.common' in err_msg: # Add other potential legacy modules if needed
                print(f"Warning: Model verification skipped due to potential legacy structure ({import_err}). Upload allowed, but runtime compatibility not guaranteed.")
                # Continue with upload despite this specific error
            else:
                # If it's a different import error, treat it as a failure
                if os.path.exists(file_path):
                    os.remove(file_path)
                print(f"Model verification failed due to unexpected import error: {import_err}")
                return jsonify({
                    'success': False,
                    'message': f'Incompatible model file (Import Error): {import_err}'
                }), 400
        except Exception as e:
            # Remove the invalid file
            if os.path.exists(file_path):
                os.remove(file_path)

            print(f"Model verification failed: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'Invalid or incompatible YOLO model file: {str(e)}'
            }), 400

        # Create model in MongoDB
        model = AIModel.create(
            name=name,
            file_path=file_path,
            description=description,
            is_default=is_default,
            is_custom=True
        )

        return jsonify({
            'success': True,
            'message': 'Model uploaded successfully',
            'model': {
                'id': model.id,
                'name': model.name,
                'file_path': model.file_path,
                'is_default': model.is_default
            }
        })
    except Exception as e:
        print(f"Unexpected error during model upload: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

# Define model download URLs (Update these URLs if they change)
PRETRAINED_MODEL_URLS = {
    # YOLOv5 (Ultralytics)
    'yolov5n': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt',
    'yolov5s': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt',
    'yolov5m': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt',
    'yolov5l': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt',
    'yolov5x': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt',
    # YOLOv8 (Ultralytics)
    'yolov8n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'yolov8s': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
    'yolov8m': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
    'yolov8l': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
    'yolov8x': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
    # YOLOv9 (WongKinYiu) - Assuming official release structure
    'yolov9c': 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9c.pt',
    'yolov9e': 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9e.pt',
    # YOLOv10 (THU-MIG)
    'yolov10n': 'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt',
    'yolov10s': 'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt',
    'yolov10m': 'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt',
    'yolov10b': 'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10b.pt',
    'yolov10l': 'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10l.pt',
    'yolov10x': 'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt',
}

@admin_bp.route('/models/download-pretrained', methods=['POST'])
@login_required
@admin_required
def download_pretrained_model():
    """Download a pre-trained YOLO model from a known URL"""
    data = request.json
    model_key = data.get('model') # e.g., 'yolov8s'

    if not model_key or model_key not in PRETRAINED_MODEL_URLS:
        return jsonify({
            'success': False,
            'message': f'Invalid or unsupported model key: {model_key}'
        }), 400

    model_url = PRETRAINED_MODEL_URLS[model_key]
    model_filename = os.path.basename(model_url) # e.g., yolov8s.pt
    model_name = model_filename.replace('.pt', '').upper() # e.g., YOLOV8S

    # Use the 'models' directory at the root for pre-trained models for consistency
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_filename)

    # Check if model file already exists
    if os.path.exists(model_path):
         # Check if it's already in the database
         existing_model = AIModel.get_by_name(model_name)
         if existing_model:
             return jsonify({
                 'success': True, # Indicate success, but file already exists
                 'message': f'Model {model_name} already exists at {model_path}',
                 'model_path': model_path
             })
         else:
             # File exists but not in DB, add it
             pass # Continue to add DB record below

    try:
        print(f"Downloading {model_name} from {model_url} to {model_path}...")
        response = requests.get(model_url, stream=True, timeout=300) # 5 min timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Download complete.")

        # Add model to database if not already present
        existing_model = AIModel.get_by_name(model_name)

        if not existing_model:
            # Determine if this should be the default (e.g., if it's yolov5s and no default exists)
            is_default_candidate = (model_key == 'yolov5s')
            make_default = is_default_candidate and not AIModel.get_default_model()
            
            model_record = AIModel.create(
                name=model_name,
                file_path=model_path,
                description=f"Pre-trained {model_name} model",
                is_custom=False,
                is_default=make_default
            )
            print(f"Added {model_name} to database.")
        else:
            print(f"Model {model_name} already exists in database.")

        return jsonify({
            'success': True,
            'message': f'Model {model_name} downloaded successfully',
            'model_path': model_path
        })

    except requests.exceptions.RequestException as e:
        print(f"Error downloading model: {str(e)}")
        # Clean up potentially incomplete file
        if os.path.exists(model_path):
             try: os.remove(model_path)
             except OSError: pass
        return jsonify({
            'success': False,
            'message': f'Error downloading model: {str(e)}'
        }), 500
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        # Clean up potentially incomplete file
        if os.path.exists(model_path):
             try: os.remove(model_path)
             except OSError: pass
        return jsonify({
            'success': False,
            'message': f'An unexpected error occurred: {str(e)}'
        }), 500

@admin_bp.route('/models/<model_id>/update', methods=['POST'])
@login_required
@admin_required
def update_model(model_id):
    """Update an AI model's metadata"""
    model = AIModel.get_by_id(model_id)
    if not model:
        flash('Model not found.', 'danger')
        return redirect(url_for('admin.manage_models'))
    
    name = request.form.get('name')
    description = request.form.get('description', '')
    
    # Validate inputs
    if not name:
        flash('Model name is required.', 'danger')
        return redirect(url_for('admin.manage_models'))
    
    # Check if name already exists for a different model
    existing = AIModel.get_by_name(name)
    if existing and existing.id != model.id:
        flash('A model with this name already exists.', 'danger')
        return redirect(url_for('admin.manage_models'))
    
    # Update model
    model.name = name
    model.description = description
    model.save()
    
    flash('Model updated successfully.', 'success')
    return redirect(url_for('admin.manage_models'))

@admin_bp.route('/models/<model_id>/set-default', methods=['POST'])
@login_required
@admin_required
def set_default_model(model_id):
    """Set an AI model as the default"""
    model = AIModel.get_by_id(model_id)
    if not model:
        return jsonify({
            'success': False,
            'message': 'Model not found'
        }), 404
    
    # Set as default (this method handles clearing other defaults)
    model.set_as_default()
    
    return jsonify({
        'success': True,
        'message': f'{model.name} set as default model'
    })

@admin_bp.route('/models/<model_id>/delete', methods=['DELETE'])
@login_required
@admin_required
def delete_model(model_id):
    """Delete an AI model"""
    model = AIModel.get_by_id(model_id)
    if not model:
        return jsonify({
            'success': False,
            'message': 'Model not found'
        }), 404
    
    # Prevent deleting default model
    if model.is_default:
        return jsonify({
            'success': False,
            'message': 'Cannot delete the default model. Set another model as default first.'
        }), 400
    
    # Delete file if it's a custom model
    if model.is_custom and os.path.exists(model.file_path):
        try:
            os.remove(model.file_path)
        except Exception as e:
            # Log error but continue with deletion
            print(f"Error deleting model file: {str(e)}")
    
    # Delete from database
    model.delete()
    
    return jsonify({
        'success': True,
        'message': 'Model deleted successfully'
    })

@admin_bp.route('/config', methods=['GET', 'POST'])
@login_required
@admin_required
def config():
    """Admin system configuration page"""
    from app.utils.system_monitor import get_system_stats
    import os
    import json
    
    config_file = os.path.join('config', 'settings.json')
    
    # Load current configuration
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        # Default configuration if file doesn't exist
        config = {
            "notifications": {
                "email_enabled": False,
                "smtp_server": "",
                "smtp_port": 587,
                "smtp_username": "",
                "smtp_password": "",
                "from_email": "",
                "email_to": ""
            },
            "recording": {
                "retention_days": 30
            },
            "detection": {
                "default_confidence": 0.45,
                "default_model": "yolov5s"
            }
        }
    
    if request.method == 'POST':
        # Update configuration from form submission
        email_enabled = request.form.get('email_enabled') == 'on'
        
        # Build updated config dictionary
        updated_config = {
            "notifications": {
                "email_enabled": email_enabled,
                "smtp_server": request.form.get('smtp_server', ''),
                "smtp_port": int(request.form.get('smtp_port', 587)),
                "smtp_username": request.form.get('smtp_username', ''),
                "from_email": request.form.get('from_email', ''),
                "email_to": request.form.get('email_to', '')
            },
            "recording": {
                "retention_days": int(request.form.get('retention_days', 30))
            },
            "detection": {
                "default_confidence": float(request.form.get('default_confidence', 0.45)),
                "default_model": request.form.get('default_model', 'yolov5s')
            }
        }
        
        # Only update password if a new one was provided
        password = request.form.get('smtp_password', '')
        if password:
            updated_config["notifications"]["smtp_password"] = password
        else:
            # Keep existing password
            updated_config["notifications"]["smtp_password"] = config["notifications"].get("smtp_password", "")
        
        # Save updated config to file
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(updated_config, f, indent=4)
            
            config = updated_config
            flash('Configuration saved successfully.', 'success')
        except Exception as e:
            flash(f'Error saving configuration: {str(e)}', 'danger')
    
    # Get system stats for display
    system_stats = get_system_stats()
    
    # Get available AI models
    ai_models = AIModel.get_all()
    
    return render_template('admin/config.html', title='System Configuration', 
                          config=config, system_stats=system_stats, ai_models=ai_models)