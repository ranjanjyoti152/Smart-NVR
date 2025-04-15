"""
AI Model for object detection
"""
from datetime import datetime
import os
from app import db

class AIModel(db.Model):
    """AI Model for object detection tasks"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    is_default = db.Column(db.Boolean, default=False)
    is_custom = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    cameras = db.relationship('Camera', backref='ai_model', lazy=True)
    
    def __repr__(self):
        return f'<AIModel {self.name}>'
    
    def exists(self):
        """Check if the model file exists"""
        return os.path.exists(self.file_path)
    
    def to_dict(self):
        """Convert model to dictionary for API"""
        return {
            'id': self.id,
            'name': self.name,
            'file_path': self.file_path,
            'description': self.description,
            'is_default': self.is_default,
            'is_custom': self.is_custom,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'camera_count': len(self.cameras) if self.cameras else 0
        }