#!/usr/bin/env python
"""
Add email_notifications column to the ROI table
"""
import os
import sys
import sqlite3

# Add the current directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    """Add email_notifications column to the ROI table"""
    # Look for the database file in the instance directory
    db_path = os.path.join('instance', 'smart_nvr.db')
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        # Try alternative locations
        alternative_paths = [
            'smartnvr.db',
            os.path.join('data', 'smart_nvr.db')
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                db_path = alt_path
                print(f"Found database at {db_path}")
                break
        else:
            print("Database not found. Please specify the database path.")
            return False
    
    print(f"Updating database schema: {db_path}")
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if the column already exists
        cursor.execute("PRAGMA table_info(roi)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'email_notifications' not in columns:
            print("Adding email_notifications column to the ROI table...")
            cursor.execute("ALTER TABLE roi ADD COLUMN email_notifications BOOLEAN DEFAULT FALSE")
            conn.commit()
            print("Column added successfully.")
        else:
            print("email_notifications column already exists.")
            
        print("Migration completed successfully.")
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"Error updating database schema: {str(e)}")
        return False
        
    finally:
        conn.close()

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)