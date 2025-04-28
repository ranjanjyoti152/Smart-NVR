"""
User model for authentication
"""
from datetime import datetime
import secrets
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from bson.objectid import ObjectId
from app import db

class User(UserMixin):
    """User model for authentication"""
    
    def __init__(self, user_data):
        if user_data is None:
            raise ValueError("User data cannot be None")
            
        # Assign properties from user_data
        self._id = user_data.get('_id')
        self.username = user_data.get('username')
        self.email = user_data.get('email')
        self.password_hash = user_data.get('password_hash')
        self.api_key = user_data.get('api_key')
        # Use internal variables directly to avoid property setter issues during initialization
        self._is_admin = user_data.get('is_admin', False)
        self._is_active = user_data.get('is_active', True)
        
        # Handle last_login as datetime (convert from string if needed)
        last_login = user_data.get('last_login')
        if last_login and isinstance(last_login, str):
            try:
                # Try to parse the ISO format string
                self.last_login = datetime.fromisoformat(last_login)
            except (ValueError, TypeError):
                # If we can't parse it, use None
                self.last_login = None
        else:
            self.last_login = last_login
        
        # Handle created_at as datetime (convert from string if needed)
        created_at = user_data.get('created_at', datetime.utcnow())
        if created_at and isinstance(created_at, str):
            try:
                # Try to parse the ISO format string
                self.created_at = datetime.fromisoformat(created_at)
            except (ValueError, TypeError):
                # If we can't parse it, use current time
                self.created_at = datetime.utcnow()
        else:
            self.created_at = created_at
            
        # Initialize user preferences
        self.preferences = user_data.get('preferences', {})
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    @property
    def id(self):
        """Return the string representation of the ObjectId as the user ID"""
        return str(self._id)
    
    @property
    def is_active(self):
        """Required by Flask-Login, check if user account is active"""
        return self._is_active
    
    @is_active.setter
    def is_active(self, value):
        """Set is_active status"""
        self._is_active = value
        
    @property
    def is_admin(self):
        """Check if user has admin role"""
        return self._is_admin
        
    @is_admin.setter
    def is_admin(self, value):
        """Set admin status"""
        self._is_admin = value
    
    def get_id(self):
        """Required by Flask-Login, return the user ID as a string"""
        return str(self._id)
    
    @classmethod
    def get_by_id(cls, user_id):
        """Get user by ID"""
        try:
            user_data = db.users.find_one({'_id': ObjectId(user_id)})
            return User(user_data) if user_data else None
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving user by ID {user_id}: {str(e)}")
            return None
    
    @classmethod
    def get_by_username(cls, username):
        """Get user by username"""
        try:
            user_data = db.users.find_one({'username': username})
            return User(user_data) if user_data else None
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving user by username {username}: {str(e)}")
            return None
    
    @classmethod
    def get_by_email(cls, email):
        """Get user by email"""
        try:
            user_data = db.users.find_one({'email': email})
            return User(user_data) if user_data else None
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving user by email {email}: {str(e)}")
            return None
    
    @classmethod
    def get_by_api_key(cls, api_key):
        """Get user by API key"""
        try:
            user_data = db.users.find_one({'api_key': api_key})
            return User(user_data) if user_data else None
        except Exception as e:
            from app import app
            app.logger.error(f"Error retrieving user by API key: {str(e)}")
            return None
    
    @classmethod
    def create(cls, username, email, password, is_admin=False):
        """Create a new user"""
        user_data = {
            'username': username,
            'email': email,
            'password_hash': generate_password_hash(password),
            'is_admin': is_admin,
            'is_active': True,
            'created_at': datetime.utcnow(),
            'preferences': {
                'email_notifications': False,
                'push_notifications': False
            }
        }
        result = db.users.insert_one(user_data)
        user_data['_id'] = result.inserted_id
        return User(user_data)
    
    @classmethod
    def get_all(cls):
        """Get all users from the database"""
        users_data = db.users.find()
        return [User(user_data) for user_data in users_data]
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
        db.users.update_one({'_id': self._id}, {'$set': {'password_hash': self.password_hash}})
    
    def check_password(self, password):
        """Check password"""
        return check_password_hash(self.password_hash, password)
    
    def generate_api_key(self):
        """Generate API key for user"""
        self.api_key = secrets.token_urlsafe(32)
        db.users.update_one({'_id': self._id}, {'$set': {'api_key': self.api_key}})
        return self.api_key
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        db.users.update_one({'_id': self._id}, {'$set': {'last_login': self.last_login}})
    
    def save(self):
        """Save user changes to database"""
        user_data = self.to_dict(include_email=True, include_api_key=True, include_preferences=True)
        user_data['_id'] = self._id
        db.users.update_one({'_id': self._id}, {'$set': user_data})
    
    def get_preference(self, key, default=None):
        """Get user preference with fallback to default"""
        if not hasattr(self, 'preferences') or self.preferences is None:
            return default
        return self.preferences.get(key, default)
    
    def set_preference(self, key, value):
        """Set a user preference"""
        if not hasattr(self, 'preferences') or self.preferences is None:
            self.preferences = {}
        self.preferences[key] = value
        db.users.update_one({'_id': self._id}, {'$set': {f'preferences.{key}': value}})
    
    def to_dict(self, include_email=False, include_api_key=False, include_preferences=False):
        """Convert user to dictionary for API"""
        data = {
            'id': str(self._id),
            'username': self.username,
            'is_admin': self.is_admin,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'last_login': self.last_login.isoformat() if isinstance(self.last_login, datetime) else self.last_login,
        }
        
        if include_email:
            data['email'] = self.email
            
        if include_api_key and self.api_key:
            data['api_key'] = self.api_key
            
        if include_preferences and hasattr(self, 'preferences'):
            data['preferences'] = self.preferences
            
        return data