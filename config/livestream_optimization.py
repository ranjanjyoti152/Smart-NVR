"""
Smart-NVR Livestream Optimization Settings

This module provides settings and utilities for optimizing livestream performance.
These settings are applied at runtime to improve the performance of live video feeds.
"""

import os
import logging
import cv2

logger = logging.getLogger(__name__)

def apply_livestream_optimizations():
    """Apply optimizations for smoother livestreaming"""
    
    # OpenCV optimizations
    try:
        # Set OpenCV thread optimization parameters
        cv2.setNumThreads(4)  # Limit OpenCV threading to prevent oversubscription
        
        # Set FFMPEG parameters for better streaming
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1048576|stimeout;500000"
        
        logger.info("Applied OpenCV livestream optimization settings")
        return True
    except Exception as e:
        logger.error(f"Failed to apply livestream optimizations: {str(e)}")
        return False

# Performance tuning parameters
STREAM_SETTINGS = {
    # Frame processing settings
    'face_detection_interval': 5,  # Process every Nth frame for face detection
    'face_resize_factor': 0.5,     # Resize factor for face detection frames
    'max_faces_per_frame': 3,      # Maximum faces to process per frame
    
    # Queue sizes
    'frame_queue_size': 10,        # Size of frame queue for detection
    'face_queue_size': 5,          # Size of face detection queue
    
    # Stream parameters
    'rtsp_buffer_size': 3,         # RTSP buffer size
    'fps_target': 20,              # Target FPS for smooth streaming
    
    # Detection settings
    'draw_roi_on_stream': False,   # Don't draw ROIs on livestream for performance
}

def get_stream_settings():
    """Get the current stream settings dictionary"""
    return STREAM_SETTINGS

def update_stream_settings(new_settings):
    """Update stream settings with new values"""
    global STREAM_SETTINGS
    STREAM_SETTINGS.update(new_settings)
    logger.info(f"Updated stream settings: {new_settings}")