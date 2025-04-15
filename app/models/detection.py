"""
Detection model for object detections
"""
from datetime import datetime
from app import db

class Detection(db.Model):
    """Detection model for object detections in video"""
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=False)
    recording_id = db.Column(db.Integer, db.ForeignKey('recording.id'), nullable=True)
    roi_id = db.Column(db.Integer, db.ForeignKey('roi.id'), nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    class_name = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    bbox_x = db.Column(db.Float, nullable=False)
    bbox_y = db.Column(db.Float, nullable=False)
    bbox_width = db.Column(db.Float, nullable=False)
    bbox_height = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(255))
    video_path = db.Column(db.String(255))
    notified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Define camera relationship only, leave recording and ROI for their respective models
    camera = db.relationship('Camera', backref=db.backref('detections', lazy='dynamic'))
    # We don't define the relationships to recording and roi here to avoid conflicts
    
    def __repr__(self):
        return f'<Detection {self.id} {self.class_name} at {self.timestamp}>'
    
    def to_dict(self):
        """Convert detection to dictionary for API"""
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'recording_id': self.recording_id, 
            'roi_id': self.roi_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': [self.bbox_x, self.bbox_y, self.bbox_width, self.bbox_height],
            'image_path': self.image_path,
            'video_path': self.video_path,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'notified': self.notified
        }