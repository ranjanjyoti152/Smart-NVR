"""
Face model for facial recognition
"""
from datetime import datetime
from bson.objectid import ObjectId
from app import db
import os

class Face:
    """Face model for storing detected faces and recognition data"""
    
    def __init__(self, face_data):
        self._id = face_data.get('_id')
        self.camera_id = face_data.get('camera_id')
        self.image_path = face_data.get('image_path')
        self.name = face_data.get('name', 'Unknown')
        self.detected_at = face_data.get('detected_at', datetime.utcnow())
        self.last_seen_at = face_data.get('last_seen_at', self.detected_at)
        self.face_encoding = face_data.get('face_encoding')
        self.created_at = face_data.get('created_at', datetime.utcnow())
    
    def __repr__(self):
        return f'<Face {self.id} - {self.name}>'
    
    @property
    def id(self):
        """Return string representation of the ObjectId"""
        return str(self._id)
    
    @classmethod
    def get_by_id(cls, face_id):
        """Get face by ID"""
        try:
            face_data = db.faces.find_one({'_id': ObjectId(face_id)})
            return Face(face_data) if face_data else None
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving face by ID {face_id}: {str(e)}")
            return None
    
    @classmethod
    def get_by_camera(cls, camera_id):
        """Get faces detected by a specific camera"""
        try:
            faces = db.faces.find({'camera_id': str(camera_id)})
            return [Face(face) for face in faces]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving faces for camera {camera_id}: {str(e)}")
            return []
    
    @classmethod
    def get_all(cls):
        """Get all faces"""
        try:
            faces = db.faces.find()
            return [Face(face) for face in faces]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving all faces: {str(e)}")
            return []
    
    @classmethod
    def get_by_name(cls, name):
        """Get faces by name"""
        try:
            faces = db.faces.find({'name': name})
            return [Face(face) for face in faces]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving faces with name {name}: {str(e)}")
            return []
    
    @classmethod
    def get_recognized(cls):
        """Get all faces that have been named (not unknown)"""
        try:
            faces = db.faces.find({'name': {'$ne': 'Unknown'}})
            return [Face(face) for face in faces]
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving recognized faces: {str(e)}")
            return []
    
    @classmethod
    def count(cls, query=None):
        """Count faces matching the given query"""
        try:
            return db.faces.count_documents(query or {})
        except Exception as e:
            from app import app
            app.logger.error(f"Error counting faces: {str(e)}")
            return 0
    
    @classmethod
    def create(cls, camera_id, image_path, face_encoding=None, name="Unknown"):
        """Create a new face entry"""
        try:
            face_data = {
                'camera_id': str(camera_id),
                'image_path': image_path,
                'name': name,
                'face_encoding': face_encoding,
                'detected_at': datetime.utcnow(),
                'last_seen_at': datetime.utcnow(),
                'created_at': datetime.utcnow()
            }
            
            result = db.faces.insert_one(face_data)
            face_data['_id'] = result.inserted_id
            return Face(face_data)
        except Exception as e:
            from app import app
            app.logger.error(f"Error creating face: {str(e)}")
            return None
    
    def update_name(self, name):
        """Update the name of a face"""
        try:
            self.name = name
            db.faces.update_one(
                {'_id': self._id},
                {'$set': {'name': name}}
            )
            return True
        except Exception as e:
            from app import app
            app.logger.error(f"Error updating face name {self.id}: {str(e)}")
            return False
    
    def update_last_seen(self):
        """Update the last seen timestamp"""
        try:
            self.last_seen_at = datetime.utcnow()
            db.faces.update_one(
                {'_id': self._id},
                {'$set': {'last_seen_at': self.last_seen_at}}
            )
            return True
        except Exception as e:
            from app import app
            app.logger.error(f"Error updating face last seen {self.id}: {str(e)}")
            return False
    
    def save(self):
        """Save face changes to database"""
        try:
            db.faces.update_one(
                {'_id': self._id},
                {'$set': {
                    'camera_id': self.camera_id,
                    'image_path': self.image_path,
                    'name': self.name,
                    'face_encoding': self.face_encoding,
                    'detected_at': self.detected_at,
                    'last_seen_at': self.last_seen_at
                }}
            )
            return True
        except Exception as e:
            from app import app
            app.logger.error(f"Error saving face {self.id}: {str(e)}")
            return False
    
    def delete(self):
        """Delete the face from the database and remove the image file"""
        try:
            # First try to delete the image file
            if self.image_path and os.path.exists(self.image_path):
                os.remove(self.image_path)
                
            # Then delete from database
            db.faces.delete_one({'_id': self._id})
            return True
        except Exception as e:
            from app import app
            app.logger.error(f"Error deleting face {self.id}: {str(e)}")
            return False
    
    def to_dict(self):
        """Convert face to dictionary for API"""
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'image_path': self.image_path,
            'name': self.name,
            'detected_at': self.detected_at.isoformat() if isinstance(self.detected_at, datetime) else self.detected_at,
            'last_seen_at': self.last_seen_at.isoformat() if isinstance(self.last_seen_at, datetime) else self.last_seen_at,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at
        }