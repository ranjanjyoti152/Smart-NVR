"""
Utilities package for SmartNVR application
"""
from .face_recognition_service import generate_face_encoding, store_person, recognize_faces

__all__ = [
    'generate_face_encoding',
    'store_person',
    'recognize_faces'
]