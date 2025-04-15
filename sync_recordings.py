#!/usr/bin/env python3
"""
Script to synchronize video recordings with database entries.
This ensures all video files in the storage directory are properly registered in the database.
"""

import os
import sys
import logging
import re
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    filename='logs/sync_recordings.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('sync_recordings')

# Add stdout handler for console output
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

def parse_timestamp_from_filename(filename):
    """Parse timestamp from filename in format YYYYMMDD_HHMMSS.mp4"""
    timestamp_pattern = r'(\d{8}_\d{6})\.mp4$'
    match = re.search(timestamp_pattern, filename)
    
    if match:
        timestamp_str = match.group(1)
        try:
            return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except ValueError:
            return None
    return None

def get_file_duration(file_path):
    """Get video duration using ffprobe"""
    import subprocess
    
    try:
        # Check if ffprobe is available
        result = subprocess.run(['which', 'ffprobe'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("ffprobe not found, estimating duration")
            return 3600.0  # Default to 1 hour for files where we can't determine duration
        
        # Get duration from ffprobe
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
               '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            duration = float(result.stdout.strip())
            return duration
        else:
            logger.warning(f"Could not determine duration for {file_path}")
            return 3600.0  # Default to 1 hour
    except Exception as e:
        logger.error(f"Error getting duration for {file_path}: {str(e)}")
        return 3600.0  # Default to 1 hour

def sync_recordings():
    """Synchronize video files in storage with database records"""
    try:
        from app import app, db
        from app.models.camera import Camera
        from app.models.recording import Recording
        from app.routes.main_routes import get_recording_settings
        
        with app.app_context():
            # Get all active cameras
            cameras = Camera.query.filter_by(is_active=True).all()
            
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
                camera_id = camera.id
                camera_dir = os.path.join(videos_base, str(camera_id))
                
                if not os.path.exists(camera_dir):
                    logger.info(f"No recordings directory for camera {camera.name} (ID: {camera_id})")
                    continue
                
                logger.info(f"Scanning recordings for camera {camera.name} (ID: {camera_id})")
                
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
                    existing = Recording.query.filter_by(
                        camera_id=camera_id,
                        file_path=file_path
                    ).first()
                    
                    if existing:
                        # Update existing record if file size changed
                        if existing.file_size != file_size:
                            # Get video duration
                            duration = get_file_duration(file_path)
                            
                            existing.file_size = file_size
                            existing.duration = duration
                            db.session.commit()
                            updated_entries += 1
                            logger.info(f"Updated existing recording: {file_path}")
                    else:
                        # Get video duration
                        duration = get_file_duration(file_path)
                        
                        # Create new database entry
                        new_recording = Recording(
                            camera_id=camera_id,
                            file_path=file_path,
                            timestamp=timestamp,
                            duration=duration,
                            file_size=file_size,
                            recording_type='continuous'
                        )
                        
                        db.session.add(new_recording)
                        db.session.commit()
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

if __name__ == '__main__':
    logger.info("Starting recording synchronization")
    result = sync_recordings()
    
    if result and result.get('success'):
        summary = result
        print(f"\nSynchronization complete!")
        print(f"Total files scanned: {summary['total_files']}")
        print(f"New entries added: {summary['new_entries']}")
        print(f"Existing entries updated: {summary['updated_entries']}")
    else:
        print("\nSynchronization failed. Check logs for details.")