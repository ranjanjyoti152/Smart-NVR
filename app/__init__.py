"""
Smart-NVR-GPU App Package
"""
from flask import Flask, get_flashed_messages
from flask_pymongo import PyMongo
from flask_login import LoginManager
from bson.objectid import ObjectId

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config.from_object('config.Config')

# Configure max content length for large file uploads (increase to 1GB)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB

# Initialize MongoDB
mongo = PyMongo(app)
db = mongo.db  # Direct access to the database

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'

# Patch the Flask get_flashed_messages function to handle malformed messages
original_get_flashed_messages = get_flashed_messages

def safe_get_flashed_messages(*args, **kwargs):
    try:
        return original_get_flashed_messages(*args, **kwargs)
    except Exception:
        # If there's an error with category processing, fall back to simple message list
        if kwargs.get('with_categories', False):
            kwargs['with_categories'] = False
            messages = original_get_flashed_messages(*args, **kwargs)
            return [('info', msg) for msg in messages]
        return []

# Replace Flask's function with our safe version
import flask
flask.helpers.get_flashed_messages = safe_get_flashed_messages

# Register blueprints
from app.routes.main_routes import main_bp
from app.routes.auth_routes import auth_bp
from app.routes.api_routes import api_bp
from app.routes.admin_routes import admin_bp

app.register_blueprint(main_bp)
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(api_bp, url_prefix='/api')
app.register_blueprint(admin_bp, url_prefix='/admin')

# Import models
from app.models import User, Camera, AIModel, Recording, Detection, ROI

@login_manager.user_loader
def load_user(user_id):
    try:
        user_data = db.users.find_one({'_id': ObjectId(user_id)})
        if user_data:
            return User(user_data)
        app.logger.warning(f"No user found with ID: {user_id}")
    except Exception as e:
        app.logger.error(f"Error loading user {user_id}: {str(e)}")
    return None