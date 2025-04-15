"""
Application configuration
"""
import os

class Config:
    """Base configuration class"""
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(24).hex())
    DEBUG = False
    
    # Database settings
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URI', 
        f'sqlite:///{os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance", "smart_nvr.db")}')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File storage
    UPLOAD_FOLDER = os.path.join('storage', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    
    # AI settings
    AI_MODELS_FOLDER = os.path.join('storage', 'models')
    DEFAULT_AI_MODEL = 'yolov5s'
    
    # Email notification settings
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'True').lower() in ('true', '1', 't')
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME', '')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD', '')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@smartnvr.com')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    # Production specific settings
    pass

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

# Select the appropriate configuration
config_name = os.environ.get('FLASK_CONFIG', 'development')
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}
Config = config_map.get(config_name, DevelopmentConfig)