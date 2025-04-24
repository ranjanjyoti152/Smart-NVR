#!/usr/bin/env python3
"""
SmartNVR - Main entry point script
"""
import os
import logging
import threading
import time
import argparse
import sys
import signal
from app import app, db
from app.utils.camera_processor import CameraManager
from app.models.user import User
from app.utils.scheduler import get_scheduler

# Global flag for signaling shutdown
shutdown_requested = False

def setup_logging():
    """Configure logging for the application"""
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('logs/smart_nvr.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    if shutdown_requested:
        logger.info("Forced shutdown requested, exiting immediately")
        sys.exit(1)
    
    logger.info("Shutdown signal received, stopping application...")
    shutdown_requested = True
    
    # Stop all camera processors
    logger.info("Stopping camera processors...")
    camera_manager = CameraManager.get_instance()
    camera_manager.stop_all_cameras()
    
    # Allow a short time for cleanup
    logger.info("Cleanup complete, exiting")
    sys.exit(0)

def initialize_database():
    """Verify database exists and is accessible"""
    logger.info("Verifying MongoDB connection...")
    
    with app.app_context():
        try:
            # Verify we can connect to the MongoDB database
            user_count = db.users.count_documents({})
            logger.info(f"MongoDB connection verified. Found {user_count} users.")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
            logger.info("Please run 'python initialize_db.py' to initialize the database")
            sys.exit(1)

def download_models():
    """Download YOLOv5 models if they don't exist"""
    import torch
    
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    model_files = {
        'yolov5s.pt': 'yolov5s',
        'yolov5m.pt': 'yolov5m',
        'yolov5l.pt': 'yolov5l'
    }
    
    for model_file, model_name in model_files.items():
        file_path = os.path.join(models_dir, model_file)
        if not os.path.exists(file_path):
            logger.info(f"Downloading {model_name} model...")
            try:
                model = torch.hub.load('ultralytics/yolov5', model_name)
                torch.save(model.state_dict(), file_path)
                logger.info(f"Downloaded {model_name} to {file_path}")
            except Exception as e:
                logger.error(f"Error downloading model {model_name}: {str(e)}")

def start_resource_monitor():
    """Start system resource monitoring in a background thread"""
    from app.utils.system_monitor import log_system_resources
    
    thread = threading.Thread(target=log_system_resources, kwargs={'interval': 60}, daemon=True)
    thread.start()
    logger.info("Started system resource monitoring")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SmartNVR - Network Video Recorder with AI')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the web server on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the web server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--no-cameras', action='store_true', help='Do not start camera processors')
    parser.add_argument('--download-models', action='store_true', help='Download YOLOv5 models')
    return parser.parse_args()

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    
    # Create required directories first
    for directory in ['logs', 'models', 'config', 'storage/recordings', 'storage/models', 'instance']:
        os.makedirs(directory, exist_ok=True)
    
    # Set up logging after directories are created
    logger = setup_logging()
    logger.info("Starting SmartNVR...")
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Download models if requested
    if args.download_models:
        download_models()
    
    # Initialize database and defaults
    initialize_database()
    
    # Start system resource monitoring
    start_resource_monitor()
    
    # Start camera processors if not disabled
    if not args.no_cameras:
        # Start cameras in a separate thread to not block the web server
        def start_cameras():
            time.sleep(2)  # Wait for app to initialize
            with app.app_context():
                camera_manager = CameraManager.get_instance()
                num_started = camera_manager.start_all_cameras()
                logger.info(f"Started {num_started} cameras")
        
        threading.Thread(target=start_cameras, daemon=True).start()
    else:
        logger.info("Camera processing disabled")
    
    # Configure API key for internal communication
    import secrets
    app.config['API_KEY'] = os.environ.get('API_KEY', secrets.token_hex(16))
    
    # Initialize scheduler for background tasks
    scheduler = get_scheduler()
    scheduler.start()
    logger.info("Initialized background task scheduler")
    
    # Start Flask web server
    logger.info(f"Starting web server on {args.host}:{args.port}...")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)