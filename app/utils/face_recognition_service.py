"""
Face recognition service using the face_recognition library.
Provides functions to generate encodings, store persons, and recognize faces in frames.
"""
import face_recognition
import numpy as np
import json # Using json to parse stringified list of floats
from app.models.person import Person # Assuming app.models.person.Person exists
# from app import db # Not directly needed as Person model handles DB interaction
# from app import app # For app.logger, if available and needed for more robust logging

# Placeholder for logging if app.logger is not easily accessible
def _log_error(message):
    print(f"ERROR: FaceRecognitionService: {message}")

def _log_info(message):
    print(f"INFO: FaceRecognitionService: {message}")

def generate_face_encoding(image_path: str) -> list[float] | None:
    """
    Loads an image, generates face encoding for the first face found.

    Args:
        image_path (str): Path to the image file.

    Returns:
        list[float] | None: The face encoding as a list of floats, or None if error.
    """
    try:
        _log_info(f"Loading image from path: {image_path}")
        image = face_recognition.load_image_file(image_path)
        _log_info(f"Generating face encodings for: {image_path}")
        encodings = face_recognition.face_encodings(image)
        if encodings:
            # Assume one face per reference image, return the first one
            _log_info(f"Found {len(encodings)} face(s) in {image_path}. Using the first one.")
            return encodings[0].tolist()
        else:
            _log_info(f"No faces found in image: {image_path}")
            return None
    except FileNotFoundError:
        _log_error(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        _log_error(f"Error generating face encoding for {image_path}: {str(e)}")
        return None

def store_person(name: str, image_path: str, description: str | None = None) -> Person | None:
    """
    Generates face encoding for an image and stores a new Person in the database.

    Args:
        name (str): Name of the person.
        image_path (str): Path to the image file for the person.
        description (str | None, optional): Description for the person. Defaults to None.

    Returns:
        Person | None: The created Person object, or None if failed.
    """
    _log_info(f"Attempting to store person: {name} with image: {image_path}")
    encoding = generate_face_encoding(image_path)
    if encoding:
        try:
            # Convert encoding (list of floats) to string for storage
            # The Person model expects a string for face_encoding
            encoding_str = json.dumps(encoding)
            
            _log_info(f"Generated encoding string for {name}. Attempting to create Person in DB.")
            # Create Person object using Person.create class method
            person = Person.create(
                name=name,
                image_path=image_path,
                face_encoding=encoding_str, # Pass the string representation
                description=description,
                is_active=True # Default to active
            )
            if person:
                _log_info(f"Person '{name}' stored successfully with ID: {person._id}")
                return person
            else:
                _log_error(f"Failed to create person '{name}' in database (Person.create returned None).")
                return None
        except Exception as e:
            # This could be a DB error from Person.create or json.dumps error
            _log_error(f"Error storing person '{name}': {str(e)}")
            return None
    else:
        _log_error(f"Could not generate face encoding for '{name}' from image {image_path}. Person not stored.")
        return None

def recognize_faces(frame: np.ndarray, known_persons: list[Person], tolerance: float = 0.6) -> list[dict]:
    """
    Recognizes known faces in a given video frame.

    Args:
        frame (np.ndarray): The video frame (as a NumPy array from OpenCV).
        known_persons (list[Person]): A list of Person objects to recognize against.
        tolerance (float, optional): How much distance between faces to consider it a match.
                                     Lower is stricter. Defaults to 0.6.

    Returns:
        list[dict]: A list of dictionaries, each containing info about a recognized person.
                    Each dict includes 'person_id', 'person_name', 'bbox', and 'confidence'.
    """
    if not known_persons:
        _log_info("No known persons provided for recognition.")
        return []

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    _log_info("Detecting face locations in the current frame.")
    face_locations = face_recognition.face_locations(rgb_frame)
    if not face_locations:
        _log_info("No faces found in the current frame.")
        return []
    
    _log_info(f"Found {len(face_locations)} face(s) in the frame. Generating encodings.")
    # Specify model="small" for faster processing if needed, but less accurate
    # face_encodings_in_frame = face_recognition.face_encodings(rgb_frame, face_locations, model="small")
    face_encodings_in_frame = face_recognition.face_encodings(rgb_frame, face_locations)

    known_face_encodings_np = []
    person_map = [] # To map back from encoding index to Person object

    _log_info("Processing known persons for comparison.")
    for person in known_persons:
        if person.face_encoding and person.is_active:
            try:
                # Stored as string, parse back to list of floats, then to numpy array
                encoding_list = json.loads(person.face_encoding)
                known_face_encodings_np.append(np.array(encoding_list))
                person_map.append(person)
            except json.JSONDecodeError:
                _log_error(f"Could not parse face encoding for person ID {person._id} ('{person.name}'). Invalid JSON: {person.face_encoding}")
            except Exception as e:
                _log_error(f"Error processing encoding for person {person.name} (ID: {person._id}): {e}")
        elif not person.is_active:
            _log_info(f"Skipping inactive person: {person.name} (ID: {person._id})")
        elif not person.face_encoding:
            _log_info(f"Skipping person with no face encoding: {person.name} (ID: {person._id})")

    if not known_face_encodings_np:
        _log_info("No valid known face encodings to compare against after processing.")
        return []

    recognized_people_list = []
    _log_info(f"Comparing {len(face_encodings_in_frame)} detected faces against {len(known_face_encodings_np)} known encodings.")

    for i, current_face_encoding_frame in enumerate(face_encodings_in_frame):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings_np, current_face_encoding_frame, tolerance=tolerance)
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings_np, current_face_encoding_frame)
        
        if len(face_distances) == 0: # Should not happen if known_face_encodings_np is not empty
            continue

        best_match_index = np.argmin(face_distances) # Finds the index of the smallest distance

        if matches[best_match_index]: # If the best match (smallest distance) is within tolerance
            matched_person_obj = person_map[best_match_index]
            current_face_location = face_locations[i] # Get the bbox for this face
            
            # (top, right, bottom, left) - this is the format from face_recognition.face_locations
            top, right, bottom, left = current_face_location

            # Confidence: 1.0 means a perfect match (distance 0).
            # Distance is a measure of dissimilarity, so 1.0 - distance can be a simple confidence score.
            # Clamp confidence between 0 and 1, as distance can sometimes slightly exceed 1.0.
            confidence = max(0.0, min(1.0, 1.0 - face_distances[best_match_index]))

            # The Person model's id property returns str(self._id)
            # If using self._id directly, ensure it's converted to string.
            # The Person model's to_dict() also stringifies _id.
            # For consistency with Detection model that has `person_id` as string:
            person_id_str = str(matched_person_obj._id)


            recognized_people_list.append({
                'person_id': person_id_str, 
                'person_name': matched_person_obj.name,
                'bbox': [top, right, bottom, left], # Standard face_locations format
                'confidence': confidence 
            })
            _log_info(f"Recognized {matched_person_obj.name} (ID: {person_id_str}) with confidence {confidence:.2f} at location: {[top, right, bottom, left]}")
        else:
            _log_info(f"No match found for face at index {i} with best distance {face_distances[best_match_index]:.2f} (tolerance: {tolerance})")


    _log_info(f"Recognition complete. Found {len(recognized_people_list)} known person(s) in the frame.")
    return recognized_people_list
