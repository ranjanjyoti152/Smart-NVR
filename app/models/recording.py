"""
Recording model for video recordings
"""
from datetime import datetime
from app import db

class Recording(db.Model):
    """Recording model for video footage storage"""
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)  # Use timestamp instead of start_time
    duration = db.Column(db.Float, default=0)  # Duration in seconds
    file_size = db.Column(db.Integer, default=0)  # File size in bytes (renamed from size_bytes)
    thumbnail_path = db.Column(db.String(255))  # Path to thumbnail image
    recording_type = db.Column(db.String(20), default='continuous')  # Type: continuous, motion, manual, etc.
    is_flagged = db.Column(db.Boolean, default=False)  # User-flagged importance
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships - use a unique backref name
    detections = db.relationship('Detection', backref='recording_parent', lazy='dynamic',
                            foreign_keys='Detection.recording_id', cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Recording {self.id} from {self.timestamp}>'
    
    def to_dict(self):
        """Convert recording to dictionary for API"""
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'duration': self.duration,
            'file_size': self.file_size,
            'recording_type': self.recording_type,
            'is_flagged': self.is_flagged,
            'video_url': f'/api/recordings/{self.id}/video',
            'thumbnail_url': f'/api/recordings/{self.id}/thumbnail',
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'detection_count': self.detections.count() if self.detections else 0,
        }
    
    # Add properties for backward compatibility
    @property
    def start_time(self):
        return self.timestamp
        
    @property
    def end_time(self):
        if self.timestamp and self.duration:
            from datetime import timedelta
            return self.timestamp + timedelta(seconds=self.duration)
        return None
        
    @property
    def size_bytes(self):
        return self.file_size
        
    @property
    def has_detections(self):
        return self.detections.count() > 0