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

def memory_cleanup():
    """Release memory and optimize RAM usage by cleaning up references and invoking garbage collection
    
    This function helps recover memory by:
    1. Clearing detection image caches
    2. Forcing explicit garbage collection
    3. Releasing any detection mappings
    
    Returns:
        dict: Memory cleanup statistics
    """
    import gc
    import psutil
    import time
    import sys
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    logger.info(f"Starting memory cleanup. Initial memory usage: {initial_memory:.2f} MB")
    
    # Clear any module-level caches
    logger.info("Clearing module-level caches")
    global CLASS_COLORS
    if 'CLASS_COLORS' in globals():
        CLASS_COLORS.clear()
    
    # Clear detection notification trackers
    from app.utils.notifications import _tracked_objects, _last_email_time
    _tracked_objects.clear()
    _last_email_time.clear()
    
    # Release any large memory objects like detection mappings
    cleared_count = 0
    try:
        from app.utils.camera_processor import CameraManager
        camera_manager = CameraManager.get_instance()
        for camera_id, processor in camera_manager.cameras.items():
            # Clear detection caches
            with processor.detection_lock:
                processor.current_detections = []
                cleared_count += 1
    except Exception as e:
        logger.error(f"Error clearing camera processor caches: {str(e)}")
    
    # Force garbage collection
    collected = gc.collect(generation=2)
    
    # Get final memory usage
    time.sleep(0.5)  # Brief pause to let memory settle
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_diff = initial_memory - final_memory
    
    logger.info(f"Memory cleanup complete. Released: {memory_diff:.2f} MB, Objects collected: {collected}")
    
    # Return stats
    return {
        'success': True,
        'initial_memory_mb': initial_memory,
        'final_memory_mb': final_memory,
        'memory_freed_mb': memory_diff,
        'objects_collected': collected,
        'caches_cleared': cleared_count
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