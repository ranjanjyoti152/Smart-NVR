from app import app, db
from app.models import Recording, Detection, Camera
from datetime import datetime

def main():
    """Check database for recordings and print results"""
    with app.app_context():
        # Check camera info
        camera = Camera.get_by_id("2")
        if camera:
            print(f"Camera 2 exists: {camera.name}")
        else:
            print("Camera 2 does not exist in database")
            
        # Check all recordings
        all_recordings = Recording.get_all()
        print(f"Total recordings in database: {len(all_recordings)}")
        
        # Check Camera 2 recordings
        recordings = Recording.get_by_camera("2")
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
            if isinstance(r.timestamp, datetime):  # Ensure timestamp is a datetime object
                dates.add(r.timestamp.strftime('%Y-%m-%d'))
        print(f"Camera 2 has recordings on these dates: {sorted(list(dates))}")
        
        # Check Camera 1 recordings on the date
        cam1_recordings = Recording.get_by_camera("1")
        cam1_recordings_on_date = [r for r in cam1_recordings 
                                if isinstance(r.timestamp, datetime) and r.timestamp.strftime('%Y-%m-%d') == date_str]
        print(f"Found {len(cam1_recordings_on_date)} recordings for Camera 1 on {date_str}")
        
        # Try direct MongoDB query as a fallback
        print("\nAttempting direct MongoDB query...")
        try:
            from datetime import datetime, timedelta
            
            # Parse date string to datetime objects for range query
            start_date = datetime.strptime(date_str, '%Y-%m-%d')
            end_date = start_date + timedelta(days=1)
            
            # Query recordings for camera 2 on the specified date
            recordings_cursor = db.recordings.find({
                'camera_id': "2",
                'timestamp': {
                    '$gte': start_date,
                    '$lt': end_date
                }
            })
            
            recordings_list = list(recordings_cursor)
            print(f"Direct MongoDB query found {len(recordings_list)} recordings")
            
            for recording in recordings_list[:5]:
                print(f"Recording ID: {recording['_id']}, Camera: {recording['camera_id']}, Timestamp: {recording['timestamp']}")
            
            # Check if camera 2 exists in the database
            camera_doc = db.cameras.find_one({'_id': '2'})
            if camera_doc:
                print(f"MongoDB confirms Camera 2 exists: {camera_doc.get('name')}")
            else:
                print("MongoDB confirms Camera 2 does not exist in database")
                
            # Get all dates for camera 2 recordings
            pipeline = [
                {'$match': {'camera_id': '2'}},
                {'$project': {
                    'date': {
                        '$dateToString': {
                            'format': '%Y-%m-%d',
                            'date': '$timestamp'
                        }
                    }
                }},
                {'$group': {'_id': '$date'}},
                {'$sort': {'_id': 1}}
            ]
            
            dates_cursor = db.recordings.aggregate(pipeline)
            dates_list = [doc['_id'] for doc in dates_cursor]
            print(f"Camera 2 has recordings on these dates according to MongoDB: {dates_list}")
            
        except Exception as e:
            print(f"MongoDB error: {e}")
            import traceback
            print(traceback.format_exc())

if __name__ == '__main__':
    main()