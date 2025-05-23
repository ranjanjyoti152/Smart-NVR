"""
Camera model for IP camera configuration
"""
from datetime import datetime
from bson.objectid import ObjectId
from app import db

class Camera:
    """Camera model for IP camera configuration"""
    
    def __init__(self, camera_data):
        self._id = camera_data.get('_id')
        self.name = camera_data.get('name')
        self.rtsp_url = camera_data.get('rtsp_url')
        self.username = camera_data.get('username')
        self.password = camera_data.get('password')
        self.is_active = camera_data.get('is_active', True)
        self.recording_enabled = camera_data.get('recording_enabled', True)
        self.detection_enabled = camera_data.get('detection_enabled', True)
        self.model_id = camera_data.get('model_id')
        self.confidence_threshold = camera_data.get('confidence_threshold', 0.5)
        self.created_at = camera_data.get('created_at', datetime.utcnow())
        self.face_recognition_enabled = camera_data.get('face_recognition_enabled', False)
        self.face_recognition_confidence = camera_data.get('face_recognition_confidence', 0.6)
    
    def __repr__(self):
        return f'<Camera {self.name}>'
    
    @property
    def id(self):
        """Return string representation of the ObjectId"""
        return str(self._id)
    
    # Add property for backward compatibility
    @property
    def enabled(self):
        return self.is_active
    
    @enabled.setter
    def enabled(self, value):
        self.is_active = value
    
    # Add property for backward compatibility
    @property
    def url(self):
        return self.rtsp_url
    
    @url.setter
    def url(self, value):
        self.rtsp_url = value
        
    # Add property for backward compatibility
    @property
    def ai_model_id(self):
        return self.model_id
    
    @ai_model_id.setter
    def ai_model_id(self, value):
        self.model_id = value
    
    @classmethod
    def get_by_id(cls, camera_id):
        """Get camera by ID"""
        try:
            camera_data = db.cameras.find_one({'_id': ObjectId(camera_id)})
            return Camera(camera_data) if camera_data else None
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving camera by ID {camera_id}: {str(e)}")
            return None
    
    @classmethod
    def get_all(cls):
        """Get all cameras"""
        try:
            return [Camera(camera) for camera in db.cameras.find()]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving all cameras: {str(e)}")
            return []
    
    @classmethod
    def get_active_cameras(cls):
        """Get all active cameras"""
        try:
            return [Camera(camera) for camera in db.cameras.find({'is_active': True})]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving active cameras: {str(e)}")
            return []
    
    @classmethod
    def create(cls, name, rtsp_url, username=None, password=None, model_id=None, 
               is_active=True, recording_enabled=True, detection_enabled=True,
               confidence_threshold=0.5, face_recognition_enabled=False, 
               face_recognition_confidence=0.6):
        """Create a new camera"""
        try:
            camera_data = {
                'name': name,
                'rtsp_url': rtsp_url,
                'username': username,
                'password': password,
                'is_active': is_active,
                'recording_enabled': recording_enabled,
                'detection_enabled': detection_enabled,
                'model_id': model_id,
                'confidence_threshold': confidence_threshold,
                'face_recognition_enabled': face_recognition_enabled,
                'face_recognition_confidence': face_recognition_confidence,
                'created_at': datetime.utcnow()
            }
            
            result = db.cameras.insert_one(camera_data)
            camera_data['_id'] = result.inserted_id
            return Camera(camera_data)
        except Exception as e:
            from app import app
            app.logger.error(f"Error creating camera {name}: {str(e)}")
            return None
    
    def save(self):
        """Save camera changes to database"""
        try:
            db.cameras.update_one(
                {'_id': self._id},
                {'$set': {
                    'name': self.name,
                    'rtsp_url': self.rtsp_url,
                    'username': self.username,
                    'password': self.password,
                    'is_active': self.is_active,
                    'recording_enabled': self.recording_enabled,
                    'detection_enabled': self.detection_enabled,
                    'model_id': self.model_id,
                    'confidence_threshold': self.confidence_threshold,
                    'face_recognition_enabled': self.face_recognition_enabled,
                    'face_recognition_confidence': self.face_recognition_confidence
                }}
            )
            return True
        except Exception as e:
            from app import app
            app.logger.error(f"Error saving camera {self.id}: {str(e)}")
            return False
    
    def delete(self):
        """Delete the camera from the database"""
        try:
            # Also delete related recordings, detections, and regions of interest
            db.recordings.delete_many({'camera_id': self.id})
            db.detections.delete_many({'camera_id': self.id})
            db.regions_of_interest.delete_many({'camera_id': self.id})
            db.cameras.delete_one({'_id': self._id})
            return True
        except Exception as e:
            from app import app
            app.logger.error(f"Error deleting camera {self.id}: {str(e)}")
            return False
    
    def get_recordings(self):
        """Get all recordings for this camera"""
        try:
            from app.models.recording import Recording
            recordings_data = db.recordings.find({'camera_id': self.id})
            return [Recording(recording) for recording in recordings_data]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving recordings for camera {self.id}: {str(e)}")
            return []
    
    def get_regions_of_interest(self):
        """Get all regions of interest for this camera"""
        try:
            from app.models.roi import ROI
            roi_data = db.regions_of_interest.find({'camera_id': self.id})
            return [ROI(roi) for roi in roi_data]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving ROIs for camera {self.id}: {str(e)}")
            return []
    
    def to_dict(self, include_credentials=False):
        """Convert camera to dictionary for API"""
        try:
            data = {
                'id': str(self._id),
                'name': self.name,
                'url': self.rtsp_url,
                'enabled': self.is_active,
                'recording_enabled': self.recording_enabled,
                'detection_enabled': self.detection_enabled,
                'model_id': self.model_id,
                'ai_model_id': self.model_id,  # Include both for compatibility
                'confidence_threshold': self.confidence_threshold,
                'stream_url': f'/api/cameras/{self.id}/stream',
                'snapshot_url': f'/api/cameras/{self.id}/snapshot',
                'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
                'face_recognition_enabled': self.face_recognition_enabled,
                'face_recognition_confidence': self.face_recognition_confidence,
            }
            
            if include_credentials:
                data['username'] = self.username
                data['password'] = self.password
                
            return data
        except Exception as e:
            from app import app
            app.logger.error(f"Error converting camera {self.id} to dict: {str(e)}")
            return {
                'id': str(self._id) if self._id else None,
                'name': self.name,
                'error': str(e)
            }