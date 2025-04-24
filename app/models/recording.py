"""
Recording model for video recordings
"""
from datetime import datetime, timedelta
from bson.objectid import ObjectId
from app import db

class Recording:
    """Recording model for video footage storage"""
    
    def __init__(self, recording_data):
        self._id = recording_data.get('_id')
        self.camera_id = recording_data.get('camera_id')
        self.file_path = recording_data.get('file_path')
        self.timestamp = recording_data.get('timestamp', datetime.utcnow())
        self.duration = recording_data.get('duration', 0)
        self.file_size = recording_data.get('file_size', 0)
        self.thumbnail_path = recording_data.get('thumbnail_path')
        self.recording_type = recording_data.get('recording_type', 'continuous')
        self.is_flagged = recording_data.get('is_flagged', False)
        self.created_at = recording_data.get('created_at', datetime.utcnow())
    
    def __repr__(self):
        return f'<Recording {self.id} from {self.timestamp}>'
    
    @property
    def id(self):
        """Return string representation of the ObjectId"""
        return str(self._id)
    
    @classmethod
    def get_by_id(cls, recording_id):
        """Get recording by ID"""
        try:
            recording_data = db.recordings.find_one({'_id': ObjectId(recording_id)})
            return Recording(recording_data) if recording_data else None
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving recording by ID {recording_id}: {str(e)}")
            return None
    
    @classmethod
    def get_by_camera(cls, camera_id):
        """Get recordings by camera ID"""
        try:
            recordings = db.recordings.find({'camera_id': str(camera_id)})
            return [Recording(recording) for recording in recordings]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving recordings for camera {camera_id}: {str(e)}")
            return []
    
    @classmethod
    def get_all(cls):
        """Get all recordings"""
        try:
            recordings = db.recordings.find()
            return [Recording(recording) for recording in recordings]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving all recordings: {str(e)}")
            return []
    
    @classmethod
    def get_by_date(cls, date_str, camera_id=None):
        """Get recordings by date"""
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            next_day = date + timedelta(days=1)
            
            query = {
                'timestamp': {
                    '$gte': date,
                    '$lt': next_day
                }
            }
            
            if camera_id:
                query['camera_id'] = str(camera_id)
                
            recordings = db.recordings.find(query).sort('timestamp', 1)
            return [Recording(recording) for recording in recordings]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving recordings by date {date_str}: {str(e)}")
            return []
    
    @classmethod
    def create(cls, camera_id, file_path, timestamp=None, duration=0, file_size=0, 
               thumbnail_path=None, recording_type='continuous', is_flagged=False):
        """Create a new recording"""
        try:
            recording_data = {
                'camera_id': str(camera_id),
                'file_path': file_path,
                'timestamp': timestamp or datetime.utcnow(),
                'duration': duration,
                'file_size': file_size,
                'thumbnail_path': thumbnail_path,
                'recording_type': recording_type,
                'is_flagged': is_flagged,
                'created_at': datetime.utcnow()
            }
            
            result = db.recordings.insert_one(recording_data)
            recording_data['_id'] = result.inserted_id
            return Recording(recording_data)
        except Exception as e:
            from app import app
            app.logger.error(f"Error creating recording: {str(e)}")
            return None
    
    def save(self):
        """Save recording changes to database"""
        try:
            db.recordings.update_one(
                {'_id': self._id},
                {'$set': {
                    'camera_id': self.camera_id,
                    'file_path': self.file_path,
                    'timestamp': self.timestamp,
                    'duration': self.duration,
                    'file_size': self.file_size,
                    'thumbnail_path': self.thumbnail_path,
                    'recording_type': self.recording_type,
                    'is_flagged': self.is_flagged
                }}
            )
            return True
        except Exception as e:
            from app import app
            app.logger.error(f"Error saving recording {self.id}: {str(e)}")
            return False
    
    def delete(self):
        """Delete the recording and its detections"""
        try:
            # Delete related detections
            db.detections.delete_many({'recording_id': str(self._id)})
            
            # Delete the recording
            db.recordings.delete_one({'_id': self._id})
            return True
        except Exception as e:
            from app import app
            app.logger.error(f"Error deleting recording {self.id}: {str(e)}")
            return False
    
    def get_detections(self):
        """Get all detections for this recording"""
        try:
            from app.models.detection import Detection
            detection_data = db.detections.find({'recording_id': str(self._id)})
            return [Detection(detection) for detection in detection_data]
        except Exception as e:
            from app import app
            app.logger.error(f"Error getting detections for recording {self.id}: {str(e)}")
            return []
    
    def get_detection_count(self):
        """Get detection count for this recording"""
        try:
            return db.detections.count_documents({'recording_id': str(self._id)})
        except Exception as e:
            from app import app
            app.logger.error(f"Error counting detections for recording {self.id}: {str(e)}")
            return 0
    
    def to_dict(self):
        """Convert recording to dictionary for API"""
        # Get detection count
        detection_count = db.detections.count_documents({'recording_id': str(self._id)})
        
        return {
            'id': str(self._id),
            'camera_id': self.camera_id,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'duration': self.duration,
            'file_size': self.file_size,
            'recording_type': self.recording_type,
            'is_flagged': self.is_flagged,
            'video_url': f'/api/recordings/{self.id}/video',
            'thumbnail_url': f'/api/recordings/{self.id}/thumbnail',
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'detection_count': detection_count,
        }
    
    # Add properties for backward compatibility
    @property
    def start_time(self):
        return self.timestamp
        
    @property
    def end_time(self):
        if self.timestamp and self.duration:
            return self.timestamp + timedelta(seconds=self.duration)
        return None
        
    @property
    def size_bytes(self):
        return self.file_size
        
    @property
    def has_detections(self):
        return self.get_detection_count() > 0