from app import app, db
from app.models.camera import Camera
from app.models.recording import Recording
from datetime import datetime, timedelta
import os
import random

def main():
    """Add test data to the database"""
    with app.app_context():
        # Check if Camera2 exists
        camera = Camera.query.filter_by(id=2).first()
        
        if not camera:
            print("Creating Camera2...")
            camera = Camera(
                id=2,
                name="Camera2",
                rtsp_url="rtsp://example.com/camera2",
                username="admin",
                password="admin",
                is_active=True,
                recording_enabled=True,
                detection_enabled=True,
                confidence_threshold=0.5,
                model_id=1  # Assuming model ID 1 exists, or set to None if no models exist
            )
            db.session.add(camera)
            db.session.commit()
            print(f"Created Camera2 with ID: {camera.id}")
        else:
            print(f"Camera2 already exists: {camera.name}")
        
        # Add recordings for April 15, 2025
        date_str = '2025-04-15'
        base_date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Create storage directory if it doesn't exist
        storage_dir = f"storage/recordings/videos/2"
        os.makedirs(storage_dir, exist_ok=True)
        
        # Add recordings at different times throughout the day
        recordings_to_add = []
        for hour in range(8, 18):  # 8 AM to 6 PM
            for minute in [0, 15, 30, 45]:  # Every 15 minutes
                timestamp = base_date.replace(hour=hour, minute=minute)
                
                # Create a dummy file path (file doesn't need to actually exist for testing)
                file_path = f"{storage_dir}/camera2_{timestamp.strftime('%Y%m%d_%H%M%S')}.mp4"
                
                # Check if a recording at this timestamp already exists
                existing = Recording.query.filter_by(
                    camera_id=camera.id, 
                    timestamp=timestamp
                ).first()
                
                if not existing:
                    recording = Recording(
                        camera_id=camera.id,
                        file_path=file_path,
                        timestamp=timestamp,
                        duration=random.randint(30, 120),  # Random duration between 30-120 seconds
                        file_size=random.randint(1000000, 10000000),  # Random file size
                        recording_type='continuous',
                        is_flagged=False
                    )
                    recordings_to_add.append(recording)
        
        if recordings_to_add:
            db.session.add_all(recordings_to_add)
            db.session.commit()
            print(f"Added {len(recordings_to_add)} test recordings for Camera2 on {date_str}")
        else:
            print("No new recordings were added (may already exist)")
        
        # Verify recordings were added
        recordings = Recording.query.filter_by(camera_id=camera.id).all()
        recordings_on_date = [r for r in recordings if r.timestamp.strftime('%Y-%m-%d') == date_str]
        print(f"Camera2 now has {len(recordings_on_date)} recordings on {date_str}")

if __name__ == '__main__':
    main()