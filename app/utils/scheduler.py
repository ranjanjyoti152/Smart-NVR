"""
Scheduler module for background tasks
Handles periodic tasks for the SmartNVR system
"""
import threading
import time
import logging
import os
import subprocess
from datetime import datetime, timedelta
import importlib.util

logger = logging.getLogger(__name__)

class BackgroundScheduler:
    """Background task scheduler for SmartNVR"""
    
    def __init__(self):
        """Initialize scheduler"""
        self.running = False
        self.thread = None
        self.tasks = []
        
    def start(self):
        """Start scheduler thread"""
        if self.thread and self.thread.is_alive():
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Scheduler started")
        
        # Register default tasks
        self._register_default_tasks()
        
    def stop(self):
        """Stop scheduler thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        
        logger.info("Scheduler stopped")
        
    def _run(self):
        """Main scheduler loop"""
        while self.running:
            self._check_tasks()
            time.sleep(30)  # Check every 30 seconds
            
    def _check_tasks(self):
        """Check if any tasks need to be executed"""
        now = datetime.now()
        
        for task in self.tasks:
            if not task.get('enabled', True):
                continue
                
            last_run = task.get('last_run')
            interval = task.get('interval', 3600)  # Default: 1 hour
            
            if not last_run or (now - last_run).total_seconds() >= interval:
                try:
                    # Run task
                    task_func = task.get('func')
                    if task_func:
                        if callable(task_func):
                            task_func()
                        else:
                            logger.warning(f"Task {task.get('name')} has invalid function")
                    
                    # Update last run time
                    task['last_run'] = now
                    logger.info(f"Executed task: {task.get('name')}")
                except Exception as e:
                    logger.error(f"Error executing task {task.get('name')}: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
    def register_task(self, name, func, interval=3600, enabled=True):
        """Register a new task
        
        Args:
            name: Task name
            func: Task function
            interval: Interval in seconds
            enabled: Whether task is enabled
        """
        # Check if task already exists
        for task in self.tasks:
            if task.get('name') == name:
                # Update existing task
                task['func'] = func
                task['interval'] = interval
                task['enabled'] = enabled
                return
                
        # Add new task
        self.tasks.append({
            'name': name,
            'func': func,
            'interval': interval,
            'enabled': enabled,
            'last_run': None
        })
        
        logger.info(f"Registered task: {name} (interval: {interval}s)")
        
    def unregister_task(self, name):
        """Unregister a task
        
        Args:
            name: Task name
        """
        self.tasks = [task for task in self.tasks if task.get('name') != name]
        logger.info(f"Unregistered task: {name}")
        
    def _register_default_tasks(self):
        """Register default system tasks"""
        # Register cleanup task
        self.register_task('cleanup', self._cleanup_task, interval=86400)  # Once per day
        
        # Register health check task
        self.register_task('health_check', self._health_check_task, interval=3600)  # Once per hour
        
        # Register database sync task - Run every 4 hours to keep database in sync with filesystem
        self.register_task('sync_recordings', self._sync_recordings_task, interval=14400)  # Every 4 hours
        
        # Register detection images cleanup task - Run every minute (for testing)
        self.register_task('cleanup_detection_images', self._cleanup_detection_images_task, interval=60)  # Every minute
        
        # Register memory cleanup task - Run every 10 seconds for aggressive RAM optimization
        self.register_task('memory_cleanup', self._memory_cleanup_task, interval=10)  # Every 10 seconds

        # Register face assimilation task - default every 6 hours
        self.register_task('auto_assimilate_faces', self._auto_assimilate_faces_task, interval=21600)
        
    def _cleanup_task(self):
        """Clean up old recordings based on retention settings"""
        from app.routes.main_routes import get_recording_settings
        from app import app, db
        from app.models.recording import Recording
        
        logger.info("Running cleanup task")
        
        with app.app_context():
            try:
                # Get recording settings
                settings = get_recording_settings()
                retention_days = int(settings.get('retention_days', 7))
                
                # Calculate cutoff date
                cutoff_date = datetime.now() - timedelta(days=retention_days)
                
                # Get recordings older than cutoff using MongoDB query
                old_recordings_data = db.recordings.find({
                    'timestamp': {'$lt': cutoff_date},
                    'is_flagged': False  # Don't delete flagged recordings
                })
                
                old_recordings = [Recording(rec) for rec in old_recordings_data]
                
                deleted_count = 0
                for recording in old_recordings:
                    try:
                        # Delete file if it exists
                        if recording.file_path and os.path.exists(recording.file_path):
                            os.remove(recording.file_path)
                            logger.info(f"Deleted recording file: {recording.file_path}")
                            
                        # Delete database record using MongoDB
                        db.recordings.delete_one({'_id': recording._id})
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting recording {recording.id}: {str(e)}")
                
                logger.info(f"Cleanup completed. Deleted {deleted_count} old recordings.")
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            
    def _health_check_task(self):
        """Check system health"""
        from app.utils.system_monitor import get_system_stats
        
        logger.info("Running health check task")
        
        try:
            # Get system statistics
            stats = get_system_stats()
            
            # Check disk space
            disk_percent = stats.get('disk', {}).get('percent', 0)
            if disk_percent > 90:
                logger.warning(f"Low disk space: {disk_percent}% used")
                
            # Check memory
            memory_percent = stats.get('memory', {}).get('percent', 0)
            if memory_percent > 90:
                logger.warning(f"High memory usage: {memory_percent}%")
                
        except Exception as e:
            logger.error(f"Error in health check task: {str(e)}")
    
    def _sync_recordings_task(self):
        """Synchronize recordings database with filesystem"""
        logger.info("Running recordings synchronization task")
        
        try:
            # Import the sync_recordings module
            spec = importlib.util.spec_from_file_location(
                "sync_recordings", 
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sync_recordings.py")
            )
            sync_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sync_module)
            
            # Run the synchronization
            result = sync_module.sync_recordings()
            
            if result and result.get('success'):
                logger.info(f"Sync completed successfully. Added {result.get('new_entries')} new recordings.")
            else:
                logger.error(f"Sync failed: {result.get('error') if result else 'Unknown error'}")
                
        except Exception as e:
            logger.error(f"Error in sync_recordings task: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _cleanup_detection_images_task(self):
        """Clean up old detection images based on retention settings"""
        from app.routes.main_routes import get_detection_settings
        from app import app
        import os
        import re
        from datetime import datetime, timedelta
        from pathlib import Path
        
        logger.info("Running detection images cleanup task")
        
        with app.app_context():
            try:
                # Get detection settings
                detection_settings = get_detection_settings()
                save_images = detection_settings.get('save_images', True)
                
                # Skip if image saving is disabled
                if not save_images:
                    logger.info("Detection image saving is disabled, skipping cleanup")
                    return
                
                image_retention_days = int(detection_settings.get('image_retention_days', 7))
                # Max images per camera (keep this number reasonably high but not excessive)
                max_images_per_camera = 1000
                
                # Calculate cutoff date
                cutoff_date = datetime.now() - timedelta(days=image_retention_days)
                
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
                    return
                
                # Process each camera's images directory
                total_deleted = 0
                
                for camera_dir in Path(images_base).iterdir():
                    if not camera_dir.is_dir():
                        continue
                    
                    logger.info(f"Processing images for camera ID: {camera_dir.name}")
                    
                    # Process all jpg files in the camera directory and sort them by timestamp
                    jpg_files = list(camera_dir.glob('*.jpg'))
                    
                    # Parse timestamps and associate with files
                    timestamped_files = []
                    for jpg_path in jpg_files:
                        try:
                            # Parse timestamp from filename (format: YYYYMMDD_HHMMSS_uuid.jpg)
                            timestamp_pattern = r'(\d{8}_\d{6})_'
                            match = re.search(timestamp_pattern, jpg_path.name)
                            
                            if match:
                                timestamp_str = match.group(1)
                                try:
                                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                                    timestamped_files.append((timestamp, jpg_path))
                                except ValueError:
                                    logger.warning(f"Could not parse timestamp from filename: {jpg_path.name}, skipping")
                            else:
                                logger.warning(f"Could not match timestamp pattern in filename: {jpg_path.name}, skipping")
                        except Exception as e:
                            logger.error(f"Error processing image {jpg_path}: {str(e)}")
                    
                    # Sort by timestamp (oldest first)
                    timestamped_files.sort(key=lambda x: x[0])
                    
                    logger.info(f"Found {len(timestamped_files)} images for camera {camera_dir.name}")
                    deleted_count = 0
                    
                    # First pass: Delete files older than retention period
                    files_to_delete = []
                    for timestamp, file_path in timestamped_files:
                        if timestamp < cutoff_date:
                            files_to_delete.append(file_path)
                    
                    # Second pass: If we still have more than max_images_per_camera, delete oldest ones
                    if len(timestamped_files) - len(files_to_delete) > max_images_per_camera:
                        # Calculate how many more we need to delete to get down to max_images_per_camera
                        extra_to_delete = len(timestamped_files) - len(files_to_delete) - max_images_per_camera
                        # Get the oldest files that weren't already marked for deletion
                        remaining_files = [f[1] for f in timestamped_files if f[1] not in files_to_delete]
                        # Add the oldest extra_to_delete files to our deletion list
                        files_to_delete.extend(remaining_files[:extra_to_delete])
                    
                    # Actually delete the files
                    for file_path in files_to_delete:
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                            total_deleted += 1
                        except Exception as e:
                            logger.error(f"Error deleting image {file_path}: {str(e)}")
                    
                    if deleted_count > 0:
                        logger.info(f"Deleted {deleted_count} images for camera {camera_dir.name} "
                                    f"({len(files_to_delete) - deleted_count} deletion failures)")
                    else:
                        logger.info(f"No images deleted for camera {camera_dir.name}")
                
                logger.info(f"Detection images cleanup completed. Deleted {total_deleted} images.")
                
            except Exception as e:
                logger.error(f"Error in detection images cleanup task: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
    
    def _memory_cleanup_task(self):
        """Periodically clean up memory to optimize RAM usage"""
        from app import app
        import importlib.util
        import os
        
        logger.info("Running memory cleanup task")
        
        try:
            # Import the cleanup_detection_images module
            spec = importlib.util.spec_from_file_location(
                "cleanup_detection_images", 
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cleanup_detection_images.py")
            )
            cleanup_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cleanup_module)
            
            # Run the memory cleanup
            result = cleanup_module.memory_cleanup()
            
            if result and result.get('success'):
                logger.info(f"Memory cleanup completed successfully. "
                           f"Released {result.get('memory_freed_mb', 0):.2f} MB of RAM. "
                           f"Collected {result.get('objects_collected', 0)} objects.")
            else:
                logger.error(f"Memory cleanup failed: {result.get('error') if result else 'Unknown error'}")
                
        except Exception as e:
            logger.error(f"Error in memory cleanup task: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def _auto_assimilate_faces_task(self):
        """Periodically assimilate unlabeled faces into known identities.
        
        Respects settings:
        - enable_auto_assimilate: Whether to run automatically
        - auto_assimilate_threshold: Similarity threshold
        - auto_assimilate_min_samples: Minimum samples required
        - auto_assimilate_interval: Interval in minutes (updates task interval)
        """
        from app import app
        from app.models.face_profile import FaceProfile
        from app.routes.main_routes import get_detection_settings

        try:
            with app.app_context():
                settings = get_detection_settings() or {}
                
                # Check if auto-assimilation is enabled
                if not settings.get('enable_auto_assimilate', False):
                    logger.debug("Auto-assimilate task skipped: disabled in settings")
                    return
                
                # Get configuration from settings
                threshold = float(settings.get('auto_assimilate_threshold', 0.9) or 0.9)
                min_samples = int(settings.get('auto_assimilate_min_samples', 2) or 2)
                
                logger.info("Running auto-assimilate faces task (threshold=%.2f, min_samples=%d)", 
                           threshold, min_samples)
                
                merges = FaceProfile.auto_assimilate_unlabeled(
                    threshold=threshold,
                    min_samples=min_samples,
                    require_multi_sample_agreement=True,
                )
                
                if merges:
                    logger.info("Auto-assimilate merged %d profiles", len(merges))
                else:
                    logger.info("Auto-assimilate found no high-confidence merges")
                    
        except Exception as exc:
            logger.error("Error during auto-assimilate task: %s", exc)
            import traceback
            logger.error(traceback.format_exc())

# Global scheduler instance
_scheduler = None

def get_scheduler():
    """Get global scheduler instance"""
    global _scheduler
    if not _scheduler:
        _scheduler = BackgroundScheduler()
    return _scheduler