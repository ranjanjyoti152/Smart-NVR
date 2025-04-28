#!/usr/bin/env python3
"""
Synchronize recording files with database
Scans the recordings directory and updates database with any new recordings
"""

import os
import time
import logging
import sys
from pathlib import Path
import cv2
from datetime import datetime, timedelta
import re
from bson.objectid import ObjectId

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def parse_timestamp_from_filename(filename):
    """Parse timestamp from a recording filename"""
    # Match patterns like "20230131_145523.mp4" or similar
    pattern = r'(\d{8}_\d{6})'
    match = re.search(pattern, filename)
    if match:
        timestamp_str = match.group(1)
        try:
            # Parse the timestamp string
            return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except ValueError:
            logger.warning(f"Could not parse timestamp from {timestamp_str}")
    return None

def get_file_duration(file_path):
    """Get the duration of a video file in seconds"""
    try:
        # Open the video file
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.warning(f"Could not open video file: {file_path}")
            return 0
            
        # Get the frame rate and frame count
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate duration
        duration = frame_count / fps if fps > 0 else 0
        
        # Release the video capture object
        cap.release()
        
        return duration
    except Exception as e:
        logger.warning(f"Error getting duration for {file_path}: {str(e)}")
        return 0

def sync_recordings():
    """Synchronize video files in storage with database records"""
    try:
        from app import app, db
        from app.models.camera import Camera
        from app.models.recording import Recording
        from app.routes.main_routes import get_recording_settings
        
        with app.app_context():
            # Get all active cameras
            cameras = Camera.get_active_cameras()
            
            if not cameras:
                logger.warning("No active cameras found in the database")
                return
            
            # Get recording settings to determine storage path
            try:
                recording_settings = get_recording_settings()
                storage_base = recording_settings.get('storage_path', 'storage/recordings')
            except Exception as e:
                logger.warning(f"Could not load recording settings: {str(e)}, using defaults")
                storage_base = 'storage/recordings'
            
            videos_base = os.path.join(storage_base, 'videos')
            if not os.path.exists(videos_base):
                logger.warning(f"Videos directory does not exist: {videos_base}")
                return
            
            total_files = 0
            new_entries = 0
            updated_entries = 0
            
            # Process each camera's recordings
            for camera in cameras:
                camera_id = str(camera.id)  # Ensure camera_id is a string for MongoDB
                logger.info(f"Camera ID type: {type(camera_id)}, value: {camera_id}")
                camera_dir = os.path.join(videos_base, str(camera_id))
                
                if not os.path.exists(camera_dir):
                    logger.info(f"No recordings directory for camera {camera.name} (ID: {camera_id})")
                    # Try alternative directory format - MongoDB ObjectIds are used as directory names now
                    alt_dirs = list(Path(videos_base).glob('*'))
                    for alt_dir in alt_dirs:
                        if alt_dir.is_dir():
                            logger.info(f"Checking alternative directory: {alt_dir.name}")
                            if str(alt_dir.name) == camera_id:
                                camera_dir = str(alt_dir)
                                logger.info(f"Found matching directory: {camera_dir}")
                                break
                    
                    if not os.path.exists(camera_dir):
                        logger.info(f"No matching directory found for camera {camera.name} (ID: {camera_id})")
                        continue
                
                logger.info(f"Scanning recordings for camera {camera.name} (ID: {camera_id}) in directory: {camera_dir}")
                
                # Get all MP4 files in the camera's recording directory
                mp4_files = list(Path(camera_dir).glob('*.mp4'))
                total_files += len(mp4_files)
                
                for mp4_path in mp4_files:
                    file_path = str(mp4_path.absolute())
                    file_size = os.path.getsize(file_path)
                    
                    # Skip empty files
                    if file_size == 0:
                        logger.warning(f"Skipping empty file: {file_path}")
                        continue
                    
                    # Parse timestamp from filename
                    timestamp = parse_timestamp_from_filename(mp4_path.name)
                    if not timestamp:
                        logger.warning(f"Could not parse timestamp from filename: {mp4_path.name}")
                        continue
                    
                    # Check if recording already exists in database
                    existing_recordings = list(db.recordings.find({
                        'camera_id': camera_id,
                        'file_path': file_path
                    }))
                    
                    if existing_recordings:
                        existing = existing_recordings[0]
                        # Update existing record if file size changed
                        if existing.get('file_size') != file_size:
                            # Get video duration
                            duration = get_file_duration(file_path)
                            
                            # Update the existing recording in MongoDB
                            db.recordings.update_one(
                                {'_id': existing['_id']},
                                {'$set': {
                                    'file_size': file_size,
                                    'duration': duration
                                }}
                            )
                            updated_entries += 1
                            logger.info(f"Updated existing recording: {file_path}")
                    else:
                        # Get video duration
                        duration = get_file_duration(file_path)
                        
                        # Create new database entry
                        recording_data = {
                            'camera_id': camera_id,  # Store camera_id as string
                            'file_path': file_path,
                            'timestamp': timestamp,
                            'duration': duration,
                            'file_size': file_size,
                            'recording_type': 'continuous',
                            'created_at': datetime.utcnow()
                        }
                        
                        result = db.recordings.insert_one(recording_data)
                        logger.info(f"Added recording with ID: {result.inserted_id}")
                        new_entries += 1
                        logger.info(f"Added new recording to database: {file_path}")
            
            logger.info(f"Synchronization complete. Scanned {total_files} files. "
                       f"Added {new_entries} new entries, updated {updated_entries} existing entries.")
            
            return {
                'success': True,
                'total_files': total_files,
                'new_entries': new_entries,
                'updated_entries': updated_entries
            }
            
    except Exception as e:
        logger.error(f"Error synchronizing recordings: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    # Run the synchronization function
    sync_recordings()