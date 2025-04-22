#!/usr/bin/env python3
"""
Script to clean up old detection images based on retention settings.
This will delete detection images that are older than the specified retention period.
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import re

# Set up logging
logging.basicConfig(
    filename='logs/cleanup_images.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('cleanup_detection_images')

# Add stdout handler for console output
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

def parse_timestamp_from_filename(filename):
    """Parse timestamp from filename in format YYYYMMDD_HHMMSS_uuid.jpg"""
    timestamp_pattern = r'(\d{8}_\d{6})_'
    match = re.search(timestamp_pattern, filename)
    
    if match:
        timestamp_str = match.group(1)
        try:
            return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except ValueError:
            return None
    return None

def cleanup_detection_images():
    """Clean up old detection images based on retention settings"""
    try:
        from app import app
        from app.routes.main_routes import get_detection_settings
        
        with app.app_context():
            # Get detection settings
            detection_settings = get_detection_settings()
            image_retention_days = detection_settings.get('image_retention_days', 7)
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=image_retention_days)
            
            logger.info(f"Cleaning up detection images older than {image_retention_days} days (before {cutoff_date})")
            
            # Get storage path from application settings
            from app.routes.main_routes import get_recording_settings
            try:
                recording_settings = get_recording_settings()
                storage_base = recording_settings.get('storage_path', 'storage/recordings')
            except Exception as e:
                logger.warning(f"Could not load recording settings: {str(e)}, using defaults")
                storage_base = 'storage/recordings'
            
            # Get all camera directories in images folder
            images_base = os.path.join(storage_base, 'images')
            if not os.path.exists(images_base):
                logger.warning(f"Images directory does not exist: {images_base}")
                return {
                    'success': True,
                    'deleted_count': 0,
                    'message': 'Images directory does not exist'
                }
            
            # Process each camera's images directory
            total_deleted = 0
            total_processed = 0
            
            for camera_dir in Path(images_base).iterdir():
                if not camera_dir.is_dir():
                    continue
                
                logger.info(f"Processing images for camera ID: {camera_dir.name}")
                
                # Process all jpg files in the camera directory
                jpg_files = list(camera_dir.glob('*.jpg'))
                
                logger.info(f"Found {len(jpg_files)} images for camera {camera_dir.name}")
                deleted_count = 0
                
                for jpg_path in jpg_files:
                    try:
                        # Parse timestamp from filename
                        timestamp = parse_timestamp_from_filename(jpg_path.name)
                        
                        if timestamp is None:
                            logger.warning(f"Could not parse timestamp from filename: {jpg_path.name}, skipping")
                            continue
                        
                        # Check if file is older than retention period
                        if timestamp < cutoff_date:
                            # Delete file
                            os.remove(jpg_path)
                            deleted_count += 1
                            total_deleted += 1
                    except Exception as e:
                        logger.error(f"Error processing image {jpg_path}: {str(e)}")
                
                logger.info(f"Deleted {deleted_count} old images for camera {camera_dir.name}")
                total_processed += len(jpg_files)
            
            logger.info(f"Cleanup complete. Processed {total_processed} images, deleted {total_deleted} old images.")
            
            return {
                'success': True,
                'total_processed': total_processed,
                'deleted_count': total_deleted
            }
    
    except Exception as e:
        logger.error(f"Error cleaning up detection images: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == '__main__':
    logger.info("Starting detection images cleanup")
    result = cleanup_detection_images()
    
    if result and result.get('success'):
        print(f"\nCleanup complete!")
        print(f"Total images processed: {result.get('total_processed', 0)}")
        print(f"Images deleted: {result.get('deleted_count', 0)}")
    else:
        print("\nCleanup failed. Check logs for details.")