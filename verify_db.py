#!/usr/bin/env python3
"""
Database verification script for SmartNVR
This script verifies the database configuration and shows sample data
"""
import os
import sys
import sqlite3
import logging
from app import app, db
from app.models.user import User
from app.models.ai_model import AIModel

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    """Verify database configuration and access"""
    try:
        # Display database URI from configuration
        logger.info(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
        db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
        logger.info(f"Database path: {db_path}")
        logger.info(f"Database file exists: {os.path.exists(db_path)}")
        
        # Attempt to access database through SQLAlchemy
        with app.app_context():
            # Check user count
            user_count = User.query.count()
            logger.info(f"Number of users in database: {user_count}")
            
            # Show admin user
            admin = User.query.filter_by(username='admin').first()
            if admin:
                logger.info(f"Admin user found: {admin.username} (Email: {admin.email})")
            else:
                logger.warning("Admin user not found!")
            
            # Check AI models
            models = AIModel.query.all()
            logger.info(f"Number of AI models: {len(models)}")
            for model in models:
                logger.info(f"  - {model.name}: {model.description} (Default: {model.is_default})")
            
            logger.info("Database access verified successfully!")
    
    except Exception as e:
        logger.error(f"Error accessing database: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()