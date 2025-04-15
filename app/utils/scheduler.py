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
                
                # Get recordings older than cutoff
                old_recordings = Recording.query.filter(
                    Recording.timestamp < cutoff_date,
                    Recording.is_flagged == False  # Don't delete flagged recordings
                ).all()
                
                deleted_count = 0
                for recording in old_recordings:
                    try:
                        # Delete file if it exists
                        if recording.file_path and os.path.exists(recording.file_path):
                            os.remove(recording.file_path)
                            logger.info(f"Deleted recording file: {recording.file_path}")
                            
                        # Delete database record
                        db.session.delete(recording)
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting recording {recording.id}: {str(e)}")
                
                # Commit changes
                db.session.commit()
                
                logger.info(f"Cleanup completed. Deleted {deleted_count} old recordings.")
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
            
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

# Global scheduler instance
_scheduler = None

def get_scheduler():
    """Get global scheduler instance"""
    global _scheduler
    if not _scheduler:
        _scheduler = BackgroundScheduler()
    return _scheduler