import numpy as np
import tensorflow as tf
import keras
import cv2
import os
import pickle
import json
from deepface import DeepFace
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess
from livenessmodel import get_liveness_model

# --- This function is required for your embedder model ---
def l2_normalize_tf(v):
    return tf.math.l2_normalize(v, axis=1)

class FaceRecognitionBackend:
    def __init__(self):
        # --- 1. DEFINE FILE PATHS ---
        model_dir = "models/"
        db_dir = "database/"
        
        # This buffer will hold the frames for the liveness model
        self.input_vid = []

        # --- 2. LOAD YOUR "IDENTIFIER" MODEL (from v2.ipynb) ---
        embedder_path = os.path.join(model_dir, "metric_learning_embedding_model_cosine.keras")
        try:
            self.embedder_model = keras.models.load_model(
                embedder_path,
                safe_mode=False,
                compile=False,
                custom_objects={"l2_normalize_tf": l2_normalize_tf}
            )
            print(f"Loaded Identifier model from {embedder_path}")
        except Exception as e:
            print(f"Error loading embedder model: {str(e)}")
            raise

        # liveness detection model
        liveness_weights_path = "models/model (1).h5" 
        try:
            self.liveness_model = get_liveness_model()
            self.liveness_model.load_weights(liveness_weights_path)
            print(f"Loaded Liveness model from {liveness_weights_path}")
        except Exception as e:
            print(f"WARNING: Could not load liveness model from {liveness_weights_path}. Error: {e}. Spoof detection will be skipped.")
            self.liveness_model = None

        # emotion detection model
        # emotion_path = os.path.join(model_dir, "face_model.h5")
        # try:
        #     self.emotion_model = keras.models.load_model(emotion_path, safe_mode=False, compile=False)
        #     self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        #     print(f"Loaded Analyzer model from {emotion_path}")
        # except Exception as e:
        #     print(f"WARNING: Could not load emotion model from {emotion_path}. Error: {e}. Emotion detection will be skipped.")
        #     self.emotion_model = None

        # --- 5. LOAD YOUR DATABASES ---
        self.face_db_path = os.path.join(db_dir, "face_database.pkl")
        self.employee_db = self.load_employee_database()
        
        self.info_db_path = os.path.join(db_dir, "employees.json")
        self.employee_info_db = self.load_employee_info()
        
        # This is the "magic number" from the final cell of your v2.ipynb
        self.VERIFICATION_THRESHOLD = 0.75

        # --- 6. LOAD OPENCV'S FAST FACE DETECTOR ---
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def load_employee_database(self):
        """Loads the .pkl file containing employee IDs and face embeddings."""
        if os.path.exists(self.face_db_path):
            try:
                with open(self.face_db_path, "rb") as f:
                    database = pickle.load(f)
                    print(f"Loaded face database with {len(database)} employees.")
                    return database
            except Exception as e:
                print(f"Error loading face_database.pkl: {e}. Creating new empty database.")
                return {}
        else:
            print(f"Warning: No face database file found at {self.face_db_path}. Creating new empty database.")
            return {}

    def load_employee_info(self):
        """Loads the .json file containing employee names, departments, etc."""
        if os.path.exists(self.info_db_path):
            with open(self.info_db_path, "r") as f:
                database = json.load(f)
                print(f"Loaded employee info database with {len(database)} employees.")
                return database
        else:
            print(f"Warning: No info database file found at {self.info_db_path}.")
            return {}
        
    def _save_pickle_db(self, path, data):
        """Helper to save a .pkl file."""
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Error saving {path}: {e}")
            return False

    def _save_json_db(self, path, data):
        """Helper to save a .json file."""
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving {path}: {e}")
            return False
            
    def _preprocess_image(self, face_roi, target_size, model_type):
        """A helper function to preprocess a face for a specific model."""
        try:
            if model_type == "verification":
                # Preprocessing for *your* ResNet50 model
                image = cv2.resize(face_roi, target_size)
                image = resnet50_preprocess(image)
            
            # elif model_type == "emotion":
            #    # 1. 'face_roi' is (h, w, 3) COLOR. Resize it first.
            #     image = cv2.resize(face_roi, target_size) # Becomes (48, 48, 3)
                
            #     # 2. Convert to grayscale.
            #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Becomes (48, 48)

            #     # 3. Normalize it.
            #     image = image / 255.0 
                
            #     # 4. Add the channel dimension: (48, 48) -> (48, 48, 1)
            #     image = np.expand_dims(image, axis=-1)
                
            # Add the "batch" dimension: (48, 48, 1) -> (1, 48, 48, 1)
            return np.expand_dims(image, axis=0)

        except Exception as e:
            print(f"Error preprocessing image for {model_type}: {e}")
            return None

    def _search_database(self, trial_embedding):
        """Compares a trial embedding to all master embeddings in the database."""
        best_match_id = None
        best_match_score = -1.0  # Use -1 for similarity (higher is better)

        for employee_id, master_embedding in self.employee_db.items():
            # Calculate Cosine Similarity
            similarity = np.dot(trial_embedding, master_embedding)
            
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_id = employee_id
                
        return best_match_id, best_match_score
    
    # ADD THIS NEW FUNCTION TO THE CLASS
    def update_liveness_buffer(self, frame):
        """Called by the GUI to continuously fill the liveness buffer."""
        if self.liveness_model:
            # 1. Preprocess the frame
            liveimg = cv2.resize(frame, (100, 100))
            liveimg = cv2.cvtColor(liveimg, cv2.COLOR_BGR2GRAY)

            # 2. Add to the 24-frame buffer
            self.input_vid.append(liveimg)

            # 3. Keep buffer at 24 frames
            if len(self.input_vid) > 24:
                self.input_vid = self.input_vid[-24:]

    def verify_employee(self, frame):
        """
        Main 3-stage verification pipeline called by the GUI.
        """
        
        # --- PIPELINE STEP 1: Liveness "Gatekeeper" ---
        if self.liveness_model:
            # 1. Check if buffer is full (it should be)
            if len(self.input_vid) < 24:
                return {'success': False, 'message': 'Initializing liveness check... Please wait.'}

            # 2. If buffer is full, run prediction
            else:
                inp = np.array([self.input_vid]) # Use the buffer we built in the background # Keep buffer at 24 frames
                inp = inp / 255.0  # Normalize
                inp = inp.reshape(1, 24, 100, 100, 1) # Reshape for Conv3D
                
                pred = self.liveness_model.predict(inp, verbose=0)
                
                # 5. Check the "real" score (pred[0][0])
                if pred[0][0] < 0.50: 
                    # If spoof, reset buffer and stop
                    # self.input_vid = [] 
                    return {'success': False, 'message': 'SPOOF DETECTED'}

        # --- If Liveness Passes, continue to Face Detection ---
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60)
        )
        
        if len(faces) == 0:
            return {'success': False, 'message': 'No face detected'}
        
        if len(faces) > 1:
            return {'success': False, 'message': 'Multiple faces detected. Please ensure only one person is in frame.'}
        
        (x, y, w, h) = faces[0]
        
        face_roi_color = frame[y:y+h, x:x+w]
        face_roi_gray = gray_frame[y:y+h, x:x+w]
        
        # --- PIPELINE STEP 2: Emotion "Analyzer" (using DeepFace) ---
        emotion_label = "N/A"
        try:
            # DeepFace analyzes the BGR face_roi directly.
            # It returns a list of faces; we take the first one.
            analysis = DeepFace.analyze(
                face_roi_color, 
                actions=['emotion'], 
                enforce_detection=False # We already detected the face
            )
            emotion_label = analysis[0]['dominant_emotion']
        except Exception as e:
            # This can happen if the face is too small or DeepFace fails
            print(f"DeepFace emotion error: {e}")
            emotion_label = "Unknown"

        # --- PIPELINE STEP 3: Verification "Identifier" ---
        ver_face = self._preprocess_image(face_roi_color, (224, 224), "verification")
        
        if ver_face is None:
            return {'success': False, 'message': 'Verification preprocessing error.'}
            
        trial_embedding = self.embedder_model.predict(ver_face, verbose=0)[0]
        
        # --- PIPELINE STEP 4: Search the Database ---
        best_id, best_score = self._search_database(trial_embedding)
        
        # --- PIPELINE STEP 5: Make the Final Decision ---
        if best_score > self.VERIFICATION_THRESHOLD:
            # We have a match!
            employee_data = self.employee_info_db.get(best_id)
            if employee_data:
                employee_name = employee_data.get("name", "Unknown")
                employee_dept = employee_data.get("department", "N/A")
            else:
                employee_name = "Unknown Employee"
                employee_dept = "N/A"
                
            return {
                'success': True,
                'employee': {'id': best_id, 'name': employee_name, 'department': employee_dept, 'emotion': emotion_label},
                'message': f"Welcome, {employee_name}!"
            }
        else:
            # No match
            return {
                'success': False,
                'employee': {'emotion': emotion_label}, # Still return the emotion
                'message': 'User Not Recognized'
            }
            
    def enroll_employee(self, frame, emp_id, emp_name, emp_dept):
        """
        Enrolls a new employee from a single frame.
        """
        # 1. Find face in the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60)
        )

        # 2. Validate face
        if len(faces) == 0:
            return {'success': False, 'message': 'Enrollment failed: No face detected.'}
        if len(faces) > 1:
            return {'success': False, 'message': 'Enrollment failed: Multiple faces detected.'}

        (x, y, w, h) = faces[0]
        face_roi_color = frame[y:y+h, x:x+w]

        # 3. Create the face "fingerprint" (vector)
        ver_face = self._preprocess_image(face_roi_color, (224, 224), "verification")
        if ver_face is None:
            return {'success': False, 'message': 'Enrollment failed: Could not process face.'}

        trial_embedding = self.embedder_model.predict(ver_face, verbose=0)[0]

        # 4. Update and save the Info DB (JSON)
        info_db = self.load_employee_info()
        info_db[emp_id] = {
            "name": emp_name,
            "department": emp_dept
        }
        if not self._save_json_db(self.info_db_path, info_db):
            return {'success': False, 'message': 'Enrollment failed: Could not save info database.'}

        # 5. Update and save the Face DB (PKL)
        face_db = self.load_employee_database()
        face_db[emp_id] = trial_embedding
        if not self._save_pickle_db(self.face_db_path, face_db):
            return {'success': False, 'message': 'Enrollment failed: Could not save face database.'}

        # 6. CRITICAL: Reload the in-memory databases
        # This makes the app recognize the new person immediately
        self.employee_db = self.load_employee_database()
        self.employee_info_db = self.load_employee_info()

        print(f"Enrollment successful for {emp_id}")
        return {'success': True, 'message': f'Successfully enrolled {emp_name} ({emp_id})'}