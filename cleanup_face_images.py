#!/usr/bin/env python3
"""
Cleanup script for face detection images
Removes old face images based on retention settings while preserving named/recognized faces
"""
import os
import sys
import logging
import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cleanup_faces.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_db_connection():
    """Create a connection to the MongoDB database"""
    try:
        # Use environment variable or default to localhost
        mongo_uri = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
        client = MongoClient(mongo_uri)
        db = client['smart_nvr']  # Database name
        return db
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        sys.exit(1)

def load_settings(db):
    """Load settings from database"""
    try:
        # Try to load settings from the database first
        settings_doc = db.settings.find_one({'type': 'global'})
        if settings_doc and 'detection' in settings_doc:
            return settings_doc.get('detection', {})
        
        # Fall back to settings file if database doesn't have it
        import json
        settings_file = os.path.join('config', 'settings.json')
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                settings = json.load(f)
                return settings.get('detection', {})
                
        return {'face_retention_days': 30}  # Default retention
    except Exception as e:
        logger.error(f"Error loading settings: {str(e)}")
        return {'face_retention_days': 30}  # Default retention

def cleanup_face_images(db, settings):
    """
    Clean up old face images based on retention settings
    """
    try:
        # Get face retention days from settings
        retention_days = settings.get('face_retention_days', 30)
        # For named/recognized faces, keep them longer
        named_retention_days = retention_days * 3  # Keep named faces 3x longer
        
        # Calculate cutoff dates
        cutoff_date_unknown = datetime.datetime.utcnow() - datetime.timedelta(days=retention_days)
        cutoff_date_named = datetime.datetime.utcnow() - datetime.timedelta(days=named_retention_days)
        
        logger.info(f"Cleaning up face images older than {retention_days} days for unknown faces")
        logger.info(f"Cleaning up face images older than {named_retention_days} days for named faces")
        
        # Find old unknown faces
        old_unknown_faces = db.faces.find({
            'name': 'Unknown',
            'detected_at': {'$lt': cutoff_date_unknown}
        })
        
        # Find old named faces
        old_named_faces = db.faces.find({
            'name': {'$ne': 'Unknown'},
            'detected_at': {'$lt': cutoff_date_named}
        })
        
        # Process unknown faces
        unknown_count = 0
        unknown_failed = 0
        for face in old_unknown_faces:
            try:
                # Check if image exists
                image_path = face.get('image_path')
                if image_path and os.path.exists(image_path):
                    # Remove the image file
                    os.remove(image_path)
                
                # Remove the database record
                db.faces.delete_one({'_id': face['_id']})
                unknown_count += 1
            except Exception as e:
                logger.error(f"Error deleting unknown face {face.get('_id')}: {str(e)}")
                unknown_failed += 1
        
        # Process named faces
        named_count = 0
        named_failed = 0
        for face in old_named_faces:
            try:
                # Check if image exists
                image_path = face.get('image_path')
                if image_path and os.path.exists(image_path):
                    # Remove the image file
                    os.remove(image_path)
                
                # Remove the database record
                db.faces.delete_one({'_id': face['_id']})
                named_count += 1
            except Exception as e:
                logger.error(f"Error deleting named face {face.get('_id')}: {str(e)}")
                named_failed += 1
        
        logger.info(f"Deleted {unknown_count} unknown faces, failed: {unknown_failed}")
        logger.info(f"Deleted {named_count} named faces, failed: {named_failed}")
        
        # Clean up empty directories
        face_storage_dir = os.path.join('storage', 'faces')
        if os.path.exists(face_storage_dir):
            for camera_dir in os.listdir(face_storage_dir):
                dir_path = os.path.join(face_storage_dir, camera_dir)
                if os.path.isdir(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    logger.info(f"Removed empty directory: {dir_path}")
        
        return unknown_count + named_count
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        return 0

def main():
    """Main execution function"""
    try:
        logger.info("Starting face images cleanup")
        
        # Create database connection
        db = create_db_connection()
        
        # Load settings
        settings = load_settings(db)
        
        # Run cleanup
        deleted_count = cleanup_face_images(db, settings)
        
        logger.info(f"Face images cleanup completed. Deleted {deleted_count} files.")
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()