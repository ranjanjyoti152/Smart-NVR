"""
Detection model for object detections
"""
from datetime import datetime
from bson.objectid import ObjectId
from app import db

class Detection:
    """Detection model for object detections in video"""
    
    def __init__(self, detection_data):
        self._id = detection_data.get('_id')
        self.camera_id = detection_data.get('camera_id')
        self.recording_id = detection_data.get('recording_id')
        self.roi_id = detection_data.get('roi_id')
        self.timestamp = detection_data.get('timestamp', datetime.utcnow())
        self.class_name = detection_data.get('class_name')
        self.confidence = detection_data.get('confidence')
        self.bbox_x = detection_data.get('bbox_x')
        self.bbox_y = detection_data.get('bbox_y')
        self.bbox_width = detection_data.get('bbox_width')
        self.bbox_height = detection_data.get('bbox_height')
        self.image_path = detection_data.get('image_path')
        self.video_path = detection_data.get('video_path')
        self.notified = detection_data.get('notified', False)
        self.created_at = detection_data.get('created_at', datetime.utcnow())
    
    def __repr__(self):
        return f'<Detection {self.id} {self.class_name} at {self.timestamp}>'
    
    @property
    def id(self):
        """Return string representation of the ObjectId"""
        return str(self._id)
    
    @classmethod
    def get_by_id(cls, detection_id):
        """Get detection by ID"""
        try:
            detection_data = db.detections.find_one({'_id': ObjectId(detection_id)})
            return Detection(detection_data) if detection_data else None
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving detection by ID {detection_id}: {str(e)}")
            return None
    
    @classmethod
    def get_by_camera(cls, camera_id, limit=100, skip=0):
        """Get detections by camera ID with pagination"""
        try:
            detections = db.detections.find({'camera_id': str(camera_id)}).sort('timestamp', -1).skip(skip).limit(limit)
            return [Detection(detection) for detection in detections]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving detections for camera {camera_id}: {str(e)}")
            return []
    
    @classmethod
    def get_by_recording(cls, recording_id):
        """Get detections by recording ID"""
        try:
            detections = db.detections.find({'recording_id': str(recording_id)}).sort('timestamp', 1)
            return [Detection(detection) for detection in detections]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving detections for recording {recording_id}: {str(e)}")
            return []
    
    @classmethod
    def get_by_roi(cls, roi_id, limit=100, skip=0):
        """Get detections by ROI ID with pagination"""
        try:
            detections = db.detections.find({'roi_id': str(roi_id)}).sort('timestamp', -1).skip(skip).limit(limit)
            return [Detection(detection) for detection in detections]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving detections for ROI {roi_id}: {str(e)}")
            return []
    
    @classmethod
    def get_by_class(cls, class_name, camera_id=None, limit=100, skip=0):
        """Get detections by class name with optional camera filter"""
        try:
            query = {'class_name': class_name}
            if camera_id:
                query['camera_id'] = str(camera_id)
            detections = db.detections.find(query).sort('timestamp', -1).skip(skip).limit(limit)
            return [Detection(detection) for detection in detections]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving detections for class {class_name}: {str(e)}")
            return []
    
    @classmethod
    def get_by_date_range(cls, start_date, end_date, camera_id=None, limit=100, skip=0):
        """Get detections by date range with optional camera filter"""
        try:
            query = {
                'timestamp': {
                    '$gte': start_date,
                    '$lt': end_date
                }
            }
            if camera_id:
                query['camera_id'] = str(camera_id)
            detections = db.detections.find(query).sort('timestamp', -1).skip(skip).limit(limit)
            return [Detection(detection) for detection in detections]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving detections by date range: {str(e)}")
            return []
    
    @classmethod
    def count_by_camera(cls, camera_id):
        """Count detections for a camera"""
        try:
            return db.detections.count_documents({'camera_id': str(camera_id)})
        except Exception as e:
            from app import app
            app.logger.error(f"Error counting detections for camera {camera_id}: {str(e)}")
            return 0
    
    @classmethod
    def count_by_recording(cls, recording_id):
        """Count detections for a recording"""
        try:
            return db.detections.count_documents({'recording_id': str(recording_id)})
        except Exception as e:
            from app import app
            app.logger.error(f"Error counting detections for recording {recording_id}: {str(e)}")
            return 0
    
    @classmethod
    def create(cls, camera_id, class_name, confidence, bbox_x, bbox_y, bbox_width, bbox_height, 
               timestamp=None, recording_id=None, roi_id=None, image_path=None, video_path=None):
        """Create a new detection"""
        try:
            detection_data = {
                'camera_id': str(camera_id),
                'class_name': class_name,
                'confidence': float(confidence),
                'bbox_x': float(bbox_x),
                'bbox_y': float(bbox_y),
                'bbox_width': float(bbox_width),
                'bbox_height': float(bbox_height),
                'timestamp': timestamp or datetime.utcnow(),
                'notified': False,
                'created_at': datetime.utcnow()
            }
            
            if recording_id:
                detection_data['recording_id'] = str(recording_id)
            
            if roi_id:
                detection_data['roi_id'] = str(roi_id)
            
            if image_path:
                detection_data['image_path'] = image_path
            
            if video_path:
                detection_data['video_path'] = video_path
            
            result = db.detections.insert_one(detection_data)
            detection_data['_id'] = result.inserted_id
            return Detection(detection_data)
        except Exception as e:
            from app import app
            app.logger.error(f"Error creating detection: {str(e)}")
            return None
    
    def save(self):
        """Save detection changes to database"""
        try:
            db.detections.update_one(
                {'_id': self._id},
                {'$set': {
                    'camera_id': self.camera_id,
                    'recording_id': self.recording_id,
                    'roi_id': self.roi_id,
                    'timestamp': self.timestamp,
                    'class_name': self.class_name,
                    'confidence': self.confidence,
                    'bbox_x': self.bbox_x,
                    'bbox_y': self.bbox_y,
                    'bbox_width': self.bbox_width,
                    'bbox_height': self.bbox_height,
                    'image_path': self.image_path,
                    'video_path': self.video_path,
                    'notified': self.notified
                }}
            )
            return True
        except Exception as e:
            from app import app
            app.logger.error(f"Error saving detection {self.id}: {str(e)}")
            return False
    
    def delete(self):
        """Delete the detection"""
        try:
            db.detections.delete_one({'_id': self._id})
            return True
        except Exception as e:
            from app import app
            app.logger.error(f"Error deleting detection {self.id}: {str(e)}")
            return False
    
    def mark_notified(self):
        """Mark detection as notified"""
        try:
            self.notified = True
            db.detections.update_one({'_id': self._id}, {'$set': {'notified': True}})
            return True
        except Exception as e:
            from app import app
            app.logger.error(f"Error marking detection {self.id} as notified: {str(e)}")
            return False
    
    def to_dict(self):
        """Convert detection to dictionary for API"""
        return {
            'id': str(self._id),
            'camera_id': self.camera_id,
            'recording_id': self.recording_id,
            'roi_id': self.roi_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': [self.bbox_x, self.bbox_y, self.bbox_width, self.bbox_height],
            'image_path': self.image_path,
            'video_path': self.video_path,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'notified': self.notified
        }