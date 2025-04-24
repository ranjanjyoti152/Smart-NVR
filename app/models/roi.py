"""
Region of Interest (ROI) model
"""
from datetime import datetime
import json
from bson.objectid import ObjectId
from app import db

class ROI:
    """Region of Interest model for defining detection areas in cameras"""
    
    def __init__(self, roi_data):
        self._id = roi_data.get('_id')
        self.camera_id = roi_data.get('camera_id')
        self.name = roi_data.get('name')
        self.coordinates = roi_data.get('coordinates')  # Array of points [[x1,y1], [x2,y2], ...]
        self.detection_classes = roi_data.get('detection_classes')  # Array of class names
        self.is_active = roi_data.get('is_active', True)
        self.email_notifications = roi_data.get('email_notifications', False)
        self.created_at = roi_data.get('created_at', datetime.utcnow())
    
    def __repr__(self):
        return f'<ROI {self.name}>'
    
    @property
    def id(self):
        """Return string representation of the ObjectId"""
        return str(self._id)
    
    @classmethod
    def get_by_id(cls, roi_id):
        """Get ROI by ID"""
        try:
            roi_data = db.regions_of_interest.find_one({'_id': ObjectId(roi_id)})
            return ROI(roi_data) if roi_data else None
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving ROI by ID {roi_id}: {str(e)}")
            return None
    
    @classmethod
    def get_by_camera(cls, camera_id):
        """Get all ROIs for a camera"""
        try:
            rois = db.regions_of_interest.find({'camera_id': str(camera_id)})
            return [ROI(roi) for roi in rois]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving ROIs for camera {camera_id}: {str(e)}")
            return []
    
    @classmethod
    def get_all(cls):
        """Get all ROIs"""
        try:
            roi_list = list(db.regions_of_interest.find())
            return [ROI(roi) for roi in roi_list]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving all ROIs: {str(e)}")
            return []
    
    @classmethod
    def create(cls, camera_id, name, coordinates, detection_classes=None, is_active=True, email_notifications=False):
        """Create a new ROI"""
        try:
            # Ensure coordinates is properly formatted
            if isinstance(coordinates, str):
                try:
                    coordinates = json.loads(coordinates)
                except:
                    coordinates = []
                    
            # Ensure detection_classes is properly formatted
            if isinstance(detection_classes, str):
                try:
                    detection_classes = json.loads(detection_classes)
                except:
                    detection_classes = []
            
            roi_data = {
                'camera_id': str(camera_id),
                'name': name,
                'coordinates': coordinates,  # Store as native array
                'detection_classes': detection_classes or [],  # Store as native array
                'is_active': is_active,
                'email_notifications': email_notifications,
                'created_at': datetime.utcnow()
            }
            
            result = db.regions_of_interest.insert_one(roi_data)
            roi_data['_id'] = result.inserted_id
            return ROI(roi_data)
        except Exception as e:
            from app import app
            app.logger.error(f"Error creating ROI '{name}' for camera {camera_id}: {str(e)}")
            return None
    
    def save(self):
        """Save ROI changes to database"""
        db.regions_of_interest.update_one(
            {'_id': self._id},
            {'$set': {
                'camera_id': self.camera_id,
                'name': self.name,
                'coordinates': self.coordinates,
                'detection_classes': self.detection_classes,
                'is_active': self.is_active,
                'email_notifications': self.email_notifications
            }}
        )
    
    def delete(self):
        """Delete the ROI and its associated detections"""
        # Delete associated detections
        db.detections.delete_many({'roi_id': str(self._id)})
        
        # Delete the ROI
        db.regions_of_interest.delete_one({'_id': self._id})
    
    def get_detections(self):
        """Get all detections for this ROI"""
        from app.models.detection import Detection
        detection_data = db.detections.find({'roi_id': str(self._id)})
        return [Detection(detection) for detection in detection_data]
    
    # Add properties for backward compatibility
    @property
    def description(self):
        return None
    
    @property
    def points(self):
        return self.coordinates
        
    @property
    def active(self):
        return self.is_active
        
    @property
    def color(self):
        return "#FF0000"  # Default color
    
    def to_dict(self):
        """Convert ROI to dictionary for API and frontend"""
        # Ensure coordinates is a list
        coordinates = self.coordinates
        if isinstance(coordinates, str):
            try:
                coordinates = json.loads(coordinates)
            except:
                coordinates = []
        
        # Ensure detection_classes is a list
        detection_classes = self.detection_classes
        if isinstance(detection_classes, str):
            try:
                detection_classes = json.loads(detection_classes)
            except:
                detection_classes = []
        
        return {
            'id': str(self._id),
            'camera_id': self.camera_id,
            'name': self.name,
            'coordinates': coordinates,
            'points': coordinates,  # For backward compatibility
            'is_active': self.is_active,
            'active': self.is_active,  # For backward compatibility
            'detection_classes': detection_classes,
            'email_notifications': self.email_notifications,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at
        }