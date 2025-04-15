#!/usr/bin/env python3
"""
Migration script to add created_at column to the recording table
"""
import sqlite3
import os
from datetime import datetime

# Path to SQLite database file
DB_PATH = os.path.join('instance', 'smart_nvr.db')

def add_created_at_column():
    """Add created_at column to the recording table"""
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if column already exists
        cursor.execute("PRAGMA table_info(recording)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        if 'created_at' not in column_names:
            print("Adding created_at column to recording table...")
            # Add column to recording table
            cursor.execute("ALTER TABLE recording ADD COLUMN created_at TIMESTAMP")
            
            # Update all existing records to have a created_at value matching their timestamp
            cursor.execute("UPDATE recording SET created_at = timestamp")
            
            # Commit changes
            conn.commit()
            print("Column added successfully.")
        else:
            print("created_at column already exists in recording table.")
        
    except Exception as e:
        print(f"Error adding column: {str(e)}")
        conn.rollback()
    finally:
        # Close connection
        conn.close()

if __name__ == "__main__":
    add_created_at_column()
    print("Migration completed.")