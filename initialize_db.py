#!/usr/bin/env python3
"""
Database initialization script for Smart-NVR using MongoDB
This script creates the required collections and indexes in MongoDB
"""
import os
import sys
import logging
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

def main():
    """Main function to initialize the MongoDB database"""
    # Load configuration
    try:
        from config import Config
        mongo_uri = Config.MONGO_URI
        mongo_dbname = Config.MONGO_DBNAME
    except ImportError:
        mongo_uri = "mongodb://localhost:27017/smart_nvr"
        mongo_dbname = "smart_nvr"
        
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
            'detections', 'regions_of_interest'
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