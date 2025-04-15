"""
Smart-NVR-GPU App Package
"""
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config.from_object('config.Config')

# Configure max content length for large file uploads (increase to 1GB)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB

# Initialize database
db = SQLAlchemy(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'

# Register blueprints
from app.routes.main_routes import main_bp
from app.routes.auth_routes import auth_bp
from app.routes.api_routes import api_bp
from app.routes.admin_routes import admin_bp

app.register_blueprint(main_bp)
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(api_bp, url_prefix='/api')
app.register_blueprint(admin_bp, url_prefix='/admin')

# Import models to ensure they are registered with SQLAlchemy
from app.models import User, Camera, AIModel, Recording, Detection, ROI

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))