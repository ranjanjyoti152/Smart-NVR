"""
Region of Interest (ROI) model
"""
from app import db

class ROI(db.Model):
    """Region of Interest model for defining detection areas in cameras"""
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    coordinates = db.Column(db.Text, nullable=False)  # JSON string of polygon points [[x1,y1], [x2,y2], ...]
    detection_classes = db.Column(db.Text)  # JSON string of class names
    is_active = db.Column(db.Boolean, default=True)
    email_notifications = db.Column(db.Boolean, default=False)  # Enable email notifications for this ROI
    
    # Define relationship with Detection
    detections = db.relationship('Detection', backref='detection_roi', lazy='dynamic', 
                              foreign_keys='Detection.roi_id')
    
    def __repr__(self):
        return f'<ROI {self.name}>'
    
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
        import json
        
        # Handle coordinates properly for frontend compatibility
        coordinates = self.coordinates
        if isinstance(coordinates, str):
            try:
                coordinates = json.loads(coordinates)
            except:
                coordinates = []
        
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'name': self.name,
            'coordinates': coordinates,
            'points': coordinates,  # For backward compatibility
            'is_active': self.is_active,
            'active': self.is_active,  # For backward compatibility
            'detection_classes': json.loads(self.detection_classes) if isinstance(self.detection_classes, str) and self.detection_classes else [],
            'email_notifications': self.email_notifications
        }