"""
AI Model for object detection
"""
from datetime import datetime
import os
from bson.objectid import ObjectId
from app import db

class AIModel:
    """AI Model for object detection tasks"""
    
    def __init__(self, model_data):
        self._id = model_data.get('_id')
        self.name = model_data.get('name')
        self.file_path = model_data.get('file_path')
        self.description = model_data.get('description')
        self.is_default = model_data.get('is_default', False)
        self.is_custom = model_data.get('is_custom', False)
        self.created_at = model_data.get('created_at', datetime.utcnow())
    
    def __repr__(self):
        return f'<AIModel {self.name}>'
    
    @property
    def id(self):
        """Return string representation of the ObjectId"""
        return str(self._id)
    
    @classmethod
    def get_by_id(cls, model_id):
        """Get model by ID"""
        try:
            model_data = db.ai_models.find_one({'_id': ObjectId(model_id)})
            return AIModel(model_data) if model_data else None
        except:
            return None
    
    @classmethod
    def get_by_name(cls, name):
        """Get model by name"""
        model_data = db.ai_models.find_one({'name': name})
        return AIModel(model_data) if model_data else None
    
    @classmethod
    def get_all(cls):
        """Get all models"""
        return [AIModel(model) for model in db.ai_models.find()]
    
    @classmethod
    def get_default_model(cls):
        """Get the default model"""
        model_data = db.ai_models.find_one({'is_default': True})
        return AIModel(model_data) if model_data else None
    
    @classmethod
    def create(cls, name, file_path, description=None, is_default=False, is_custom=False):
        """Create a new model"""
        # If setting as default, clear other defaults
        if is_default:
            db.ai_models.update_many({'is_default': True}, {'$set': {'is_default': False}})
        
        model_data = {
            'name': name,
            'file_path': file_path,
            'description': description,
            'is_default': is_default,
            'is_custom': is_custom,
            'created_at': datetime.utcnow()
        }
        
        result = db.ai_models.insert_one(model_data)
        model_data['_id'] = result.inserted_id
        return AIModel(model_data)
    
    def exists(self):
        """Check if the model file exists"""
        return os.path.exists(self.file_path)
    
    def set_as_default(self):
        """Set this model as the default"""
        # Clear other defaults
        db.ai_models.update_many({'is_default': True}, {'$set': {'is_default': False}})
        
        # Set this one as default
        self.is_default = True
        db.ai_models.update_one({'_id': self._id}, {'$set': {'is_default': True}})
    
    def save(self):
        """Save model changes to database"""
        model_data = self.to_dict()
        model_data['_id'] = self._id
        db.ai_models.update_one({'_id': self._id}, {'$set': model_data})
    
    def delete(self):
        """Delete the model from the database"""
        db.ai_models.delete_one({'_id': self._id})
    
    def to_dict(self):
        """Convert model to dictionary for API"""
        # Get camera count (cameras using this model)
        camera_count = db.cameras.count_documents({'model_id': str(self._id)})
        
        return {
            'id': str(self._id),
            'name': self.name,
            'file_path': self.file_path,
            'description': self.description,
            'is_default': self.is_default,
            'is_custom': self.is_custom,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'camera_count': camera_count
        }