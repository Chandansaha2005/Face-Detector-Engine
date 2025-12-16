import cv2
import numpy as np
import mediapipe as mp
import math
import pickle
from datetime import datetime

class MediaPipeFaceSystem:
    """Complete face detection and recognition using MediaPipe"""
    
    def __init__(self):
        print("Initializing MediaPipe Face System...")
        
        # Initialize MediaPipe modules
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Face Detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0=short range, 1=long range
            min_detection_confidence=0.5
        )
        
        # Face Mesh for landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Database for known faces
        self.known_faces = {}  # person_id -> {'embedding': [], 'metadata': {}}
        self.face_embeddings = []
        self.face_labels = []
        
        print("✓ MediaPipe initialized successfully")
        print("  - Face Detection: Ready")
        print("  - Face Landmarks: Ready")
    
    def detect_faces(self, image):
        """Detect faces in image using MediaPipe"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process for face detection
        results = self.face_detection.process(rgb_image)
        
        faces = []
        
        if results.detections:
            for idx, detection in enumerate(results.detections):
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                box_width = int(bboxC.width * w)
                box_height = int(bboxC.height * h)
                
                # Adjust bounding box to be larger
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x1 + box_width + 2*padding)
                y2 = min(h, y1 + box_height + 2*padding)
                
                # Extract face ROI
                face_roi = image[y1:y2, x1:x2]
                
                if face_roi.size > 0:
                    # Get face landmarks for this face
                    landmarks = self.extract_landmarks(face_roi)
                    
                    faces.append({
                        'id': idx,
                        'bbox': (x1, y1, x2, y2),
                        'face_image': face_roi,
                        'confidence': float(detection.score[0]),
                        'landmarks': landmarks,
                        'embedding': self.create_embedding(face_roi, landmarks)
                    })
        
        return faces
    
    def extract_landmarks(self, face_image):
        """Extract facial landmarks"""
        try:
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_face)
            
            if results.multi_face_landmarks:
                landmarks = []
                for landmark in results.multi_face_landmarks[0].landmark:
                    # Convert to pixel coordinates relative to face ROI
                    h, w = face_image.shape[:2]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append((x, y))
                
                return landmarks
            
        except:
            pass
        
        return None
    
    def create_embedding(self, face_image, landmarks=None):
        """Create embedding vector from face"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Resize to standard size
            gray = cv2.resize(gray, (100, 100))
            
            # Apply histogram equalization
            gray = cv2.equalizeHist(gray)
            
            # Flatten and normalize
            embedding = gray.flatten().astype('float32') / 255.0
            
            # Add landmarks if available
            if landmarks and len(landmarks) > 0:
                # Normalize landmarks
                landmarks_vec = np.array(landmarks).flatten()
                landmarks_vec = landmarks_vec / np.linalg.norm(landmarks_vec)
                # Combine with image features
                embedding = np.concatenate([embedding, landmarks_vec[:100]])
            
            # Normalize final embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return None
    
    def add_known_face(self, person_id, name, face_image):
        """Add a known face to database"""
        try:
            # Detect face in the image
            faces = self.detect_faces(face_image)
            
            if len(faces) == 0:
                print("✗ No face detected in the image")
                return False
            
            # Use the first detected face
            face_data = faces[0]
            
            # Store in database
            self.known_faces[person_id] = {
                'embedding': face_data['embedding'],
                'metadata': {
                    'name': name,
                    'added_at': datetime.now().isoformat(),
                    'face_size': face_data['face_image'].shape
                }
            }
            
            # For quick search
            self.face_embeddings.append(face_data['embedding'])
            self.face_labels.append(person_id)
            
            print(f"✓ Added face: {name} (ID: {person_id})")
            print(f"  Embedding size: {len(face_data['embedding'])}")
            
            return True
            
        except Exception as e:
            print(f"Error adding face: {e}")
            return False
    
    def recognize_face(self, query_embedding, threshold=0.6):
        """Recognize face from embedding"""
        if len(self.face_embeddings) == 0 or query_embedding is None:
            return None, 0.0
        
        best_match = None
        best_similarity = 0
        
        for idx, known_embedding in enumerate(self.face_embeddings):
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, known_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = self.face_labels[idx]
        
        # Apply threshold
        if best_similarity >= threshold:
            return best_match, best_similarity
        
        return None, best_similarity
    
    def compare_faces(self, embedding1, embedding2):
        """Compare two face embeddings"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2)
        distance = 1 - similarity
        
        return similarity, distance
    
    def draw_detection(self, image, face_data, recognized_name=None, similarity=0.0):
        """Draw detection results on image"""
        x1, y1, x2, y2 = face_data['bbox']
        
        # Choose color based on recognition
        if recognized_name and similarity > 0.7:
            # Known person - Green
            color = (0, 255, 0)
            thickness = 3
            label = f"{recognized_name} ({similarity:.2f})"
            
            # Draw "FOUND" above the box
            cv2.putText(image, "FOUND", (x1, y1 - 40),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
            
        elif recognized_name and similarity > 0.6:
            # Possible match - Yellow
            color = (0, 255, 255)
            thickness = 2
            label = f"Possible: {recognized_name} ({similarity:.2f})"
            
        else:
            # Unknown - Red
            color = (0, 0, 255)
            thickness = 2
            label = "Unknown"
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, 
                     (x1, y2 - label_size[1] - 10),
                     (x1 + label_size[0], y2),
                     color, cv2.FILLED)
        
        # Draw label
        cv2.putText(image, label, (x1, y2 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def save_to_file(self, filename='face_database.pkl'):
        """Save known faces to file"""
        try:
            data = {
                'known_faces': self.known_faces,
                'embeddings': self.face_embeddings,
                'labels': self.face_labels
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"✓ Saved {len(self.known_faces)} faces to {filename}")
            
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def load_from_file(self, filename='face_database.pkl'):
        """Load known faces from file"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.known_faces = data['known_faces']
            self.face_embeddings = data['embeddings']
            self.face_labels = data['labels']
            
            print(f"✓ Loaded {len(self.known_faces)} faces from {filename}")
            
        except FileNotFoundError:
            print(f"ℹ️  No existing database found. Starting fresh.")
        except Exception as e:
            print(f"Error loading database: {e}")