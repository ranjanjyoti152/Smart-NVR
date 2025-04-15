from app import app, db
from app.models.recording import Recording
from app.models.detection import Detection
from app.models.camera import Camera
from datetime import datetime
import sqlite3

def main():
    """Check database for recordings and print results"""
    with app.app_context():
        # Check camera info
        camera = Camera.query.get(2)
        if camera:
            print(f"Camera 2 exists: {camera.name}")
        else:
            print("Camera 2 does not exist in database")
            
        # Check all recordings
        all_recordings = Recording.query.all()
        print(f"Total recordings in database: {len(all_recordings)}")
        
        # Check Camera 2 recordings
        recordings = Recording.query.filter_by(camera_id=2).all()
        print(f"Found {len(recordings)} total recordings for Camera 2")
        
        # Check April 15, 2025 recordings
        date_str = '2025-04-15'
        recordings_on_date = [r for r in recordings if r.timestamp.strftime('%Y-%m-%d') == date_str]
        print(f"Found {len(recordings_on_date)} recordings for Camera 2 on {date_str}")
        
        for r in recordings_on_date[:5]:
            print(f"Recording ID: {r.id}, Timestamp: {r.timestamp}, Duration: {r.duration}s")
            
        # Check if recordings exist for Camera 2 on ANY date
        dates = set()
        for r in recordings:
            dates.add(r.timestamp.strftime('%Y-%m-%d'))
        print(f"Camera 2 has recordings on these dates: {sorted(list(dates))}")
        
        # Check Camera 1 recordings on the date
        cam1_recordings = Recording.query.filter_by(camera_id=1).all()
        cam1_recordings_on_date = [r for r in cam1_recordings if r.timestamp.strftime('%Y-%m-%d') == date_str]
        print(f"Found {len(cam1_recordings_on_date)} recordings for Camera 1 on {date_str}")
        
        # Try direct SQLite query as a fallback
        print("\nAttempting direct SQLite query...")
        try:
            conn = sqlite3.connect('instance/smart_nvr.db')
            cursor = conn.cursor()
            cursor.execute(f"SELECT id, camera_id, timestamp FROM recording WHERE camera_id = 2 AND date(timestamp) = '{date_str}'")
            rows = cursor.fetchall()
            print(f"Direct SQLite query found {len(rows)} recordings")
            for row in rows[:5]:
                print(f"Recording ID: {row[0]}, Camera: {row[1]}, Timestamp: {row[2]}")
            
            # Check if camera 2 exists in the database
            cursor.execute("SELECT id, name FROM camera WHERE id = 2")
            cam = cursor.fetchone()
            if cam:
                print(f"SQLite confirms Camera 2 exists: {cam[1]}")
            else:
                print("SQLite confirms Camera 2 does not exist in database")
                
            # Get all dates for camera 2 recordings
            cursor.execute("SELECT DISTINCT date(timestamp) FROM recording WHERE camera_id = 2")
            dates = cursor.fetchall()
            print(f"Camera 2 has recordings on these dates according to SQLite: {[d[0] for d in dates]}")
            
            conn.close()
        except Exception as e:
            print(f"SQLite error: {e}")

if __name__ == '__main__':
    main()