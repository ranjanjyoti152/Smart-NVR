"""
Camera model for IP camera configuration
"""
from datetime import datetime
from app import db

class Camera(db.Model):
    """Camera model for IP camera configuration"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    rtsp_url = db.Column(db.String(255), nullable=False)
    username = db.Column(db.String(100))
    password = db.Column(db.String(100))
    is_active = db.Column(db.Boolean, default=True)
    recording_enabled = db.Column(db.Boolean, default=True)
    detection_enabled = db.Column(db.Boolean, default=True)
    model_id = db.Column(db.Integer, db.ForeignKey('ai_model.id'))
    confidence_threshold = db.Column(db.Float, default=0.5)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Define relationships
    recordings = db.relationship('Recording', backref='camera', lazy=True, cascade="all, delete-orphan")
    regions_of_interest = db.relationship('ROI', backref='camera', lazy=True, cascade="all, delete-orphan")
    
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
    
    def __repr__(self):
        return f'<Camera {self.name}>'
    
    def to_dict(self, include_credentials=False):
        """Convert camera to dictionary for API"""
        data = {
            'id': self.id,
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
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
        
        if include_credentials:
            data['username'] = self.username
            data['password'] = self.password
            
        return data