#!/usr/bin/env python3
"""
Simple database initialization script for SmartNVR
This script uses direct SQLite connections to create the database and tables
"""
import os
import sys
import sqlite3
import logging
import shutil
from datetime import datetime
from werkzeug.security import generate_password_hash

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
    """Main function to initialize the database"""
    # Create or ensure instance directory exists with full permissions
    instance_dir = os.path.join(os.getcwd(), 'instance')
    os.makedirs(instance_dir, exist_ok=True)
    os.chmod(instance_dir, 0o777)  # Full permissions
    
    # Path to database file
    db_path = os.path.join(instance_dir, 'smart_nvr.db')
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        logger.info(f"Removing existing database at {db_path}")
        os.remove(db_path)
        logger.info("Database file removed")
    
    # Create a new empty database file with full permissions
    with open(db_path, 'w') as f:
        pass
    os.chmod(db_path, 0o666)  # Read/write permissions
    
    logger.info(f"Creating new database at {db_path}")
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        logger.info("Successfully connected to the database")
        
        # Create User table
        logger.info("Creating User table")
        cursor.execute('''
        CREATE TABLE user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            api_key TEXT UNIQUE,
            is_admin BOOLEAN NOT NULL DEFAULT 0,
            is_active BOOLEAN NOT NULL DEFAULT 1,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
        ''')
        
        # Create AIModel table
        logger.info("Creating AIModel table")
        cursor.execute('''
        CREATE TABLE ai_model (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            file_path TEXT NOT NULL,
            description TEXT,
            is_default BOOLEAN NOT NULL DEFAULT 0,
            is_custom BOOLEAN NOT NULL DEFAULT 0,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create Camera table
        logger.info("Creating Camera table")
        cursor.execute('''
        CREATE TABLE camera (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rtsp_url TEXT NOT NULL,
            username TEXT,
            password TEXT,
            is_active BOOLEAN NOT NULL DEFAULT 1,
            recording_enabled BOOLEAN NOT NULL DEFAULT 1,
            detection_enabled BOOLEAN NOT NULL DEFAULT 1,
            model_id INTEGER,
            confidence_threshold REAL NOT NULL DEFAULT 0.45,
            location TEXT,
            status TEXT DEFAULT 'offline',
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES ai_model (id)
        )
        ''')
        
        # Create ROI table
        logger.info("Creating ROI table")
        cursor.execute('''
        CREATE TABLE roi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            coordinates TEXT NOT NULL,
            detection_classes TEXT,
            is_active BOOLEAN NOT NULL DEFAULT 1,
            email_notifications BOOLEAN NOT NULL DEFAULT 0,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (camera_id) REFERENCES camera (id)
        )
        ''')
        
        # Create Recording table
        logger.info("Creating Recording table")
        cursor.execute('''
        CREATE TABLE recording (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            thumbnail_path TEXT,
            timestamp TIMESTAMP NOT NULL,
            duration INTEGER,
            file_size INTEGER,
            recording_type TEXT NOT NULL DEFAULT 'continuous',
            is_flagged BOOLEAN NOT NULL DEFAULT 0,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (camera_id) REFERENCES camera (id)
        )
        ''')
        
        # Create Detection table
        logger.info("Creating Detection table")
        cursor.execute('''
        CREATE TABLE detection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER NOT NULL,
            recording_id INTEGER,
            roi_id INTEGER,
            timestamp TIMESTAMP NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox_x REAL NOT NULL,
            bbox_y REAL NOT NULL,
            bbox_width REAL NOT NULL,
            bbox_height REAL NOT NULL,
            image_path TEXT,
            video_path TEXT,
            notified BOOLEAN NOT NULL DEFAULT 0,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (camera_id) REFERENCES camera (id),
            FOREIGN KEY (recording_id) REFERENCES recording (id),
            FOREIGN KEY (roi_id) REFERENCES roi (id)
        )
        ''')
        
        # Insert default admin user
        logger.info("Creating default admin user: admin/admin")
        cursor.execute('''
        INSERT INTO user (username, email, password_hash, is_admin, created_at)
        VALUES (?, ?, ?, ?, ?)
        ''', ('admin', 'admin@example.com', hash_password('admin'), 1, datetime.now()))
        
        # Insert default YOLOv5 models
        logger.info("Creating default AI models")
        models = [
            ('YOLOv5s', 'models/yolov5s.pt', 'Small version of YOLOv5', 1, 0),
            ('YOLOv5m', 'models/yolov5m.pt', 'Medium version of YOLOv5', 0, 0),
            ('YOLOv5l', 'models/yolov5l.pt', 'Large version of YOLOv5', 0, 0)
        ]
        
        cursor.executemany('''
        INSERT INTO ai_model (name, file_path, description, is_default, is_custom)
        VALUES (?, ?, ?, ?, ?)
        ''', models)
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        logger.info("Database initialization complete!")
        logger.info("You can now run 'python run.py' to start the application")
        logger.info("Login with username: admin, password: admin")
        
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {e}")
        if os.path.exists(db_path):
            os.remove(db_path)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if os.path.exists(db_path):
            os.remove(db_path)
        sys.exit(1)

if __name__ == "__main__":
    main()