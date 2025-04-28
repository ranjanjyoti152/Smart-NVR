#!/usr/bin/env python3
"""
Database initialization script for Smart-NVR using MongoDB
This script creates the required collections and indexes in MongoDB
It also supports configuring custom storage locations for MongoDB data
"""
import os
import sys
import logging
import subprocess
import shutil
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING
from werkzeug.security import generate_password_hash
import json

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def hash_password(password):
    """Generate password hash using Werkzeug's method"""
    return generate_password_hash(password)

def configure_mongodb_storage(db_path=None):
    """Configure MongoDB to use a specific storage path"""
    if not db_path:
        return True
    
    # Create directory if it doesn't exist
    os.makedirs(db_path, exist_ok=True)
    
    try:
        # Check if we need to change MongoDB configuration
        mongodb_conf = "/etc/mongodb.conf"
        if os.path.exists(mongodb_conf):
            with open(mongodb_conf, 'r') as f:
                config_content = f.read()
            
            # Check if dbPath is already set to our path
            db_path_line = f"dbPath: {db_path}"
            if db_path_line in config_content:
                logger.info(f"MongoDB already configured to use {db_path}")
                return True
            
            # Backup existing config
            backup_file = f"{mongodb_conf}.bak"
            shutil.copy2(mongodb_conf, backup_file)
            logger.info(f"Backed up MongoDB config to {backup_file}")
            
            # Update config with new dbPath
            new_config = []
            db_path_set = False
            
            for line in config_content.splitlines():
                if line.strip().startswith("dbPath:"):
                    new_config.append(f"dbPath: {db_path}")
                    db_path_set = True
                else:
                    new_config.append(line)
            
            if not db_path_set:
                # Add dbPath if it wasn't in the config
                new_config.append(f"dbPath: {db_path}")
            
            # Write updated config
            with open(mongodb_conf, 'w') as f:
                f.write("\n".join(new_config))
            
            logger.info(f"Updated MongoDB configuration to use {db_path}")
            
            # Restart MongoDB service
            logger.info("Restarting MongoDB service...")
            result = subprocess.run(["systemctl", "restart", "mongodb"], 
                                    capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to restart MongoDB: {result.stderr}")
                return False
            
            logger.info("MongoDB service restarted successfully")
            return True
            
        else:
            logger.warning(f"MongoDB config file {mongodb_conf} not found. Manual configuration required.")
            return False
            
    except Exception as e:
        logger.error(f"Error configuring MongoDB storage: {e}")
        return False

def main():
    """Main function to initialize the MongoDB database"""
    # Load configuration
    try:
        from config import Config
        mongo_uri = Config.MONGO_URI
        mongo_dbname = Config.MONGO_DBNAME
        # Get custom DB path if configured
        db_storage_path = None
        if hasattr(Config, 'DB_STORAGE_PATH') and Config.DB_STORAGE_PATH:
            db_storage_path = Config.DB_STORAGE_PATH
    except ImportError:
        mongo_uri = "mongodb://localhost:27017/smart_nvr"
        mongo_dbname = "smart_nvr"
        db_storage_path = None
        
    # Check for command line argument for DB path
    if len(sys.argv) > 1 and sys.argv[1].startswith('--db-path='):
        db_storage_path = sys.argv[1].split('=')[1]
        logger.info(f"Using command line specified DB path: {db_storage_path}")
    
    # Configure MongoDB storage if a custom path is specified
    if db_storage_path:
        logger.info(f"Configuring MongoDB to use storage path: {db_storage_path}")
        if not configure_mongodb_storage(db_storage_path):
            logger.warning("Failed to configure MongoDB storage. Continuing with default storage location.")
    
    logger.info(f"Connecting to MongoDB at {mongo_uri}")
    
    try:
        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        db = client[mongo_dbname]
        
        # Check connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        
        # Drop existing collections if they exist
        existing_collections = db.list_collection_names()
        
        collections_to_create = [
            'users', 'ai_models', 'cameras', 'recordings', 
            'detections', 'regions_of_interest', 'system_settings'
        ]
        
        for collection in collections_to_create:
            if collection in existing_collections:
                logger.info(f"Dropping existing collection: {collection}")
                db.drop_collection(collection)
        
        # Create collections and indexes
        
        # Users collection
        logger.info("Creating users collection and indexes")
        db.create_collection("users")
        db.users.create_index([("username", ASCENDING)], unique=True)
        db.users.create_index([("email", ASCENDING)], unique=True)
        db.users.create_index([("api_key", ASCENDING)], unique=True, sparse=True)
        
        # AI Models collection
        logger.info("Creating ai_models collection and indexes")
        db.create_collection("ai_models")
        db.ai_models.create_index([("name", ASCENDING)], unique=True)
        
        # Cameras collection
        logger.info("Creating cameras collection and indexes")
        db.create_collection("cameras")
        db.cameras.create_index([("name", ASCENDING)])
        
        # Regions of Interest collection
        logger.info("Creating regions_of_interest collection and indexes")
        db.create_collection("regions_of_interest")
        db.regions_of_interest.create_index([("camera_id", ASCENDING)])
        
        # Recordings collection
        logger.info("Creating recordings collection and indexes")
        db.create_collection("recordings")
        db.recordings.create_index([("camera_id", ASCENDING)])
        db.recordings.create_index([("timestamp", DESCENDING)])
        db.recordings.create_index([("file_path", ASCENDING)], unique=True)
        
        # Detections collection
        logger.info("Creating detections collection and indexes")
        db.create_collection("detections")
        db.detections.create_index([("camera_id", ASCENDING)])
        db.detections.create_index([("recording_id", ASCENDING)])
        db.detections.create_index([("roi_id", ASCENDING)])
        db.detections.create_index([("timestamp", DESCENDING)])
        db.detections.create_index([("class_name", ASCENDING)])
        
        # System Settings collection for storing disk management settings
        logger.info("Creating system_settings collection")
        db.create_collection("system_settings")
        
        # Initialize system settings with disk management defaults
        system_settings = {
            "recording": {
                "storage_path": "storage/recordings",
                "retention_days": 30,
                "clip_length": 60,
                "format": "mp4",
                "use_separate_db_storage": db_storage_path is not None,
                "db_storage_path": db_storage_path or "/var/lib/mongodb",
                "db_storage_size": 20,
                "mount_options": "defaults,noatime",
                "allow_formatting": False,
                "auto_mount": True
            }
        }
        db.system_settings.insert_one(system_settings)
        
        # Insert default admin user
        logger.info("Creating default admin user: admin/admin")
        default_user = {
            "username": "admin",
            "email": "admin@example.com",
            "password_hash": hash_password("admin"),
            "is_admin": True,
            "is_active": True,
            "created_at": datetime.utcnow()
        }
        db.users.insert_one(default_user)
        
        # Insert default YOLOv5 models
        logger.info("Creating default AI models")
        models = [
            {
                "name": "YOLOv5s",
                "file_path": "models/yolov5s.pt",
                "description": "Small version of YOLOv5",
                "is_default": True,
                "is_custom": False,
                "created_at": datetime.utcnow()
            },
            {
                "name": "YOLOv5m",
                "file_path": "models/yolov5m.pt",
                "description": "Medium version of YOLOv5",
                "is_default": False,
                "is_custom": False,
                "created_at": datetime.utcnow()
            },
            {
                "name": "YOLOv5l",
                "file_path": "models/yolov5l.pt",
                "description": "Large version of YOLOv5",
                "is_default": False,
                "is_custom": False,
                "created_at": datetime.utcnow()
            }
        ]
        
        db.ai_models.insert_many(models)
        
        logger.info("MongoDB database initialization complete!")
        logger.info("You can now run 'python run.py' to start the application")
        logger.info("Login with username: admin, password: admin")
        
    except Exception as e:
        logger.error(f"Error initializing MongoDB: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()