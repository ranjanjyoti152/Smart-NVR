"""
Defines the Person model for interacting with the 'recognized_persons' MongoDB collection.
"""
from datetime import datetime, timezone
from bson.objectid import ObjectId
from pymongo.errors import PyMongoError # For more specific error handling if needed
from app import db

class Person:
    """
    Represents a person with attributes stored in MongoDB.
    Interacts with the 'recognized_persons' collection.
    """
    collection_name = "recognized_persons"

    def __init__(self, data):
        """
        Initializes a Person instance from a dictionary.

        Args:
            data (dict): A dictionary containing person attributes.
                         Expected keys: 'name', 'image_path', 'face_encoding'.
                         Optional keys: '_id', 'description', 'is_active',
                                        'created_at', 'updated_at'.
        """
        self._id = data.get('_id')
        self.name = data.get('name')
        self.description = data.get('description')
        self.face_encoding = data.get('face_encoding') # Stored as string
        self.image_path = data.get('image_path')
        self.is_active = data.get('is_active', True)
        
        created_at_data = data.get('created_at')
        self.created_at = created_at_data if isinstance(created_at_data, datetime) \
            else datetime.now(timezone.utc)

        updated_at_data = data.get('updated_at')
        self.updated_at = updated_at_data if isinstance(updated_at_data, datetime) \
            else datetime.now(timezone.utc)


    def save(self):
        """
        Inserts a new person document or updates an existing one in the
        'recognized_persons' collection. Updates 'updated_at' automatically.

        Returns:
            ObjectId or None: The _id of the saved document, or None if save failed.
        """
        collection = db[self.collection_name]
        self.updated_at = datetime.now(timezone.utc)
        
        doc_data = self.to_dict(exclude_id_str=True) # Use internal representation
        # Remove _id from doc_data if it's None (for inserts)
        if doc_data.get('_id') is None:
            doc_data.pop('_id', None)

        try:
            if self._id:
                # Update existing document
                result = collection.update_one({'_id': self._id}, {'$set': doc_data})
                if result.modified_count == 1:
                    return self._id
                # If not found to update, it's an issue or was deleted.
                # Consider if we should upsert or handle as error.
                # For now, if modified_count is 0, assume it wasn't found or data was same.
                # To be safer, one might check result.matched_count.
                return self._id if result.matched_count == 1 else None
            else:
                # Insert new document
                if 'created_at' not in doc_data or not isinstance(doc_data['created_at'], datetime):
                    doc_data['created_at'] = self.created_at # Ensure it's set
                
                result = collection.insert_one(doc_data)
                self._id = result.inserted_id
                return self._id
        except PyMongoError as e:
            # Log error e
            print(f"MongoDB Error in save: {e}")
            return None

    def delete(self):
        """
        Deletes the person document from the collection.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        if not self._id:
            return False
        collection = db[self.collection_name]
        try:
            result = collection.delete_one({'_id': self._id})
            return result.deleted_count == 1
        except PyMongoError as e:
            print(f"MongoDB Error in delete: {e}")
            return False

    def to_dict(self, exclude_id_str=False):
        """
        Returns a dictionary representation of the person.
        Converts _id to string and datetime objects to ISO format strings.

        Args:
            exclude_id_str (bool): If True, _id is returned as ObjectId, not string.
                                   Useful for internal representation for saving.

        Returns:
            dict: A dictionary representation of the Person instance.
        """
        data = {
            'name': self.name,
            'description': self.description,
            'face_encoding': self.face_encoding,
            'image_path': self.image_path,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'updated_at': self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
        }
        if self._id:
            data['_id'] = str(self._id) if not exclude_id_str else self._id
        
        # For saving, we want actual datetime objects, not ISO strings
        if exclude_id_str: # implies internal use for saving
            data['created_at'] = self.created_at
            data['updated_at'] = self.updated_at

        return data

    @classmethod
    def create(cls, name, image_path, face_encoding, description=None, is_active=True):
        """
        Creates a new Person instance, saves it to the database, and returns the instance.
        'created_at' and 'updated_at' are set automatically.

        Args:
            name (str): The name of the person.
            image_path (str): The path to the person's image.
            face_encoding (str): The string representation of the face encoding.
            description (str, optional): A description of the person. Defaults to None.
            is_active (bool, optional): Whether the person is active. Defaults to True.

        Returns:
            Person: The created Person instance, or None if creation failed.
        """
        now = datetime.now(timezone.utc)
        data = {
            'name': name,
            'image_path': image_path,
            'face_encoding': face_encoding, # Assumed to be string
            'description': description,
            'is_active': is_active,
            'created_at': now,
            'updated_at': now
        }
        person = cls(data)
        if person.save():
            return person
        return None

    @classmethod
    def get_by_id(cls, person_id):
        """
        Retrieves a person by their _id.

        Args:
            person_id (str or ObjectId): The ID of the person to retrieve.

        Returns:
            Person: The Person instance if found, otherwise None.
        """
        collection = db[cls.collection_name]
        try:
            if isinstance(person_id, str):
                person_id = ObjectId(person_id)
        except Exception: # Invalid ObjectId format
            return None
        
        try:
            doc = collection.find_one({'_id': person_id})
            if doc:
                return cls(doc)
            return None
        except PyMongoError as e:
            print(f"MongoDB Error in get_by_id: {e}")
            return None

    @classmethod
    def get_by_name(cls, name):
        """
        Retrieves a person by their name.
        Note: Names might not be unique. This method returns the first match.

        Args:
            name (str): The name of the person to retrieve.

        Returns:
            Person: The Person instance if found, otherwise None.
        """
        collection = db[cls.collection_name]
        try:
            doc = collection.find_one({'name': name})
            if doc:
                return cls(doc)
            return None
        except PyMongoError as e:
            print(f"MongoDB Error in get_by_name: {e}")
            return None

    @classmethod
    def get_all(cls, page=1, per_page=20):
        """
        Retrieves all persons, with pagination.

        Args:
            page (int, optional): The page number to retrieve. Defaults to 1.
            per_page (int, optional): The number of persons per page. Defaults to 20.

        Returns:
            list: A list of Person instances. Returns empty list on error or no data.
        """
        collection = db[cls.collection_name]
        persons_list = []
        try:
            skip_count = (page - 1) * per_page
            cursor = collection.find().skip(skip_count).limit(per_page)
            for doc in cursor:
                persons_list.append(cls(doc))
            return persons_list
        except PyMongoError as e:
            print(f"MongoDB Error in get_all: {e}")
            return [] # Return empty list on error

    @classmethod
    def update(cls, person_id, update_data):
        """
        Updates specific fields of a person. 'updated_at' is set automatically.
        The `update_data` should not contain `_id`, `created_at`.
        `updated_at` will be overridden.

        Args:
            person_id (str or ObjectId): The ID of the person to update.
            update_data (dict): A dictionary of fields to update.

        Returns:
            Person: The updated Person instance, or None if the person was not found or update failed.
        """
        collection = db[cls.collection_name]
        
        try:
            if isinstance(person_id, str):
                obj_id = ObjectId(person_id)
            elif isinstance(person_id, ObjectId):
                obj_id = person_id
            else:
                return None # Invalid person_id type
        except Exception: # Invalid ObjectId format
            return None

        if not update_data:
            # Optionally, retrieve and return the person if no data to update
            return cls.get_by_id(obj_id) 

        update_doc = {'$set': update_data.copy()} # Work on a copy
        update_doc['$set']['updated_at'] = datetime.now(timezone.utc)

        # Ensure not to update immutable fields directly via this method
        for field in ['_id', 'created_at']:
            if field in update_doc['$set']:
                del update_doc['$set'][field]
        
        try:
            result = collection.update_one({'_id': obj_id}, update_doc)
            if result.matched_count == 1:
                return cls.get_by_id(obj_id) # Fetch the updated document
            return None # Not found or not updated
        except PyMongoError as e:
            print(f"MongoDB Error in update: {e}")
            return None

    @classmethod
    def find_by_face_encoding(cls, face_encoding_str, tolerance=0.6):
        """
        Finds persons whose face encodings are similar to the given one.
        This is a placeholder and requires actual face comparison logic.
        Currently, it would only find exact string matches for 'face_encoding'.
        
        NOTE: Proper implementation requires loading encodings and using a library
              like face_recognition.compare_faces. This method as-is will not
              perform true "similarity" search based on the string representation
              unless the string representation itself is cleverly designed for matching.
              For now, it's a simple equality check on the string.

        Args:
            face_encoding_str (str): The string representation of the face encoding to search for.
            tolerance (float): Tolerance for face comparison (not used in this placeholder).

        Returns:
            list: A list of Person instances matching the encoding (by string equality).
        """
        collection = db[cls.collection_name]
        persons_list = []
        try:
            # This is a naive search. Real implementation needs to fetch encodings,
            # convert them to numerical arrays, and compare.
            cursor = collection.find({'face_encoding': face_encoding_str})
            for doc in cursor:
                persons_list.append(cls(doc))
            return persons_list
        except PyMongoError as e:
            print(f"MongoDB Error in find_by_face_encoding: {e}")
            return []

    @classmethod
    def count(cls):
        """
        Counts the total number of persons in the collection.

        Returns:
            int: The total number of persons, or 0 if an error occurs.
        """
        collection = db[cls.collection_name]
        try:
            return collection.count_documents({})
        except PyMongoError as e:
            print(f"MongoDB Error in count: {e}")
            return 0
