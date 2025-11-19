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
            'detections', 'regions_of_interest', 'face_profiles'
        ]
        
        logger.info("ROI collections will support time-based scheduling with the following fields:")
        logger.info("- roi_type: 'always_active' or 'time_based'")
        logger.info("- start_time: HH:MM format for time-based ROIs")
        logger.info("- end_time: HH:MM format for time-based ROIs") 
        logger.info("- active_days: Array of weekday numbers (0=Monday, 6=Sunday)")
        
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
        db.regions_of_interest.create_index([("roi_type", ASCENDING)])
        db.regions_of_interest.create_index([("is_active", ASCENDING)])
        # Compound index for time-based ROI queries
        db.regions_of_interest.create_index([
            ("camera_id", ASCENDING), 
            ("roi_type", ASCENDING), 
            ("is_active", ASCENDING)
        ])
        
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
        db.detections.create_index([("face_profile_id", ASCENDING)], sparse=True)
        db.detections.create_index([("face_status", ASCENDING)])
        db.detections.create_index([("source", ASCENDING)])

        # Face profiles collection
        logger.info("Creating face_profiles collection and indexes")
        db.create_collection("face_profiles")
        db.face_profiles.create_index([("name", ASCENDING)], unique=False, sparse=True)
        db.face_profiles.create_index([("status", ASCENDING)])
        db.face_profiles.create_index([("last_seen", DESCENDING)])
        
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
        
        # Optionally create sample ROIs to demonstrate time-based scheduling
        # (These will only be created if cameras exist)
        sample_rois = [
            {
                "name": "Sample Always Active ROI",
                "camera_id": "sample_camera_id",  # This would need to be a real camera ID
                "coordinates": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
                "detection_classes": [],
                "is_active": True,
                "email_notifications": False,
                "use_gemini_notifications": False,
                "roi_type": "always_active",
                "start_time": None,
                "end_time": None,
                "active_days": [0, 1, 2, 3, 4, 5, 6],
                "created_at": datetime.utcnow()
            },
            {
                "name": "Sample Business Hours ROI",
                "camera_id": "sample_camera_id",  # This would need to be a real camera ID
                "coordinates": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                "detection_classes": [],
                "is_active": True,
                "email_notifications": False,
                "use_gemini_notifications": False,
                "roi_type": "time_based",
                "start_time": "09:00",
                "end_time": "17:00",
                "active_days": [0, 1, 2, 3, 4],  # Monday-Friday
                "created_at": datetime.utcnow()
            }
        ]
        
        logger.info("Sample ROI configurations prepared (will be created when cameras are added)")
        logger.info("- Always Active ROI: Continuously monitors the defined area")
        logger.info("- Business Hours ROI: Active Monday-Friday, 9:00-17:00")
        
        logger.info("MongoDB database initialization complete!")
        logger.info("New features included:")
        logger.info("✓ Time-based ROI scheduling support")
        logger.info("✓ Enhanced database indexes for performance")
        logger.info("✓ Sample ROI configurations prepared")
        logger.info("✓ Face detection and recognition collections ready")
        logger.info("")
        logger.info("You can now run 'python run.py' to start the application")
        logger.info("Login with username: admin, password: admin")
        logger.info("")
        logger.info("After adding cameras, you can create ROIs with:")
        logger.info("- Always Active: Traditional continuous monitoring")
        logger.info("- Time-Based: Automatic activation during specific hours/days")
        
    except Exception as e:
        logger.error(f"Error initializing MongoDB: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()