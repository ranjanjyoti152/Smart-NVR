"""
Application configuration
"""
import os

class Config:
    """Base configuration class"""
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(24).hex())
    DEBUG = False
    
    # Database settings (MongoDB)
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/smart_nvr')
    MONGO_DBNAME = 'smart_nvr'
    
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
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/smart_nvr_dev')
    MONGO_DBNAME = 'smart_nvr_dev'

class ProductionConfig(Config):
    """Production configuration"""
    # Production specific settings
    pass

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/smart_nvr_test')
    MONGO_DBNAME = 'smart_nvr_test'

# Select the appropriate configuration
config_name = os.environ.get('FLASK_CONFIG', 'development')
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}
Config = config_map.get(config_name, DevelopmentConfig)