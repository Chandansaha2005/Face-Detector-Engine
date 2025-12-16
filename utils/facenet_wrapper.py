import tensorflow as tf
import numpy as np
import cv2
from mtcnn import MTCNN
import os

class FaceNetRecognizer:
    """FaceNet-based face recognition system"""
    
    def __init__(self, model_path='facenet_model/facenet_keras.h5'):
        print("Loading FaceNet model...")
        
        # Load FaceNet model
        self.model = tf.keras.models.load_model(model_path)
        print(f"✓ FaceNet model loaded from {model_path}")
        
        # Initialize MTCNN for face detection
        self.detector = MTCNN()
        
        # Input shape required by FaceNet
        self.input_shape = (160, 160)
        
    def detect_faces(self, image):
        """Detect faces using MTCNN"""
        # Convert BGR to RGB (MTCNN expects RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detections = self.detector.detect_faces(rgb_image)
        
        faces = []
        for detection in detections:
            # Get bounding box
            x, y, width, height = detection['box']
            
            # Ensure coordinates are positive
            x = max(0, x)
            y = max(0, y)
            
            # Extract face
            face = rgb_image[y:y+height, x:x+width]
            
            # Get face landmarks (optional, for alignment)
            keypoints = detection['keypoints']
            
            faces.append({
                'bbox': (x, y, x+width, y+height),  # (x1, y1, x2, y2)
                'face': face,
                'confidence': detection['confidence'],
                'keypoints': keypoints
            })
        
        return faces
    
    def preprocess_face(self, face_image):
        """Preprocess face for FaceNet"""
        # Resize to 160x160
        face_resized = cv2.resize(face_image, self.input_shape)
        
        # Convert to float32
        face_float = face_resized.astype('float32')
        
        # Normalize pixel values (as done in FaceNet training)
        face_normalized = (face_float - 127.5) / 127.5
        
        # Add batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def get_embedding(self, face_image):
        """Get 128-D embedding from face image"""
        # Preprocess
        processed_face = self.preprocess_face(face_image)
        
        # Get embedding
        embedding = self.model.predict(processed_face, verbose=0)[0]
        
        # Normalize embedding (unit vector)
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def get_embeddings_from_image(self, image_path):
        """Get embeddings from an image file"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image: {image_path}")
            return []
        
        # Detect faces
        faces = self.detect_faces(image)
        
        embeddings = []
        for face_data in faces:
            # Get embedding for each face
            embedding = self.get_embedding(face_data['face'])
            
            embeddings.append({
                'embedding': embedding,
                'bbox': face_data['bbox'],
                'confidence': face_data['confidence']
            })
        
        return embeddings
    
    def compare_faces(self, embedding1, embedding2, threshold=0.6):
        """
        Compare two face embeddings
        Returns: (is_match, distance)
        """
        # Calculate Euclidean distance
        distance = np.linalg.norm(embedding1 - embedding2)
        
        # Convert distance to similarity score (0-1)
        similarity = 1 - min(distance / 2.0, 1.0)
        
        # Determine if it's a match
        is_match = distance < threshold
        
        return is_match, similarity, distance
    
    def find_best_match(self, query_embedding, database_embeddings):
        """Find best match in database"""
        best_match = None
        best_similarity = 0
        best_distance = float('inf')
        
        for person_id, ref_embedding in database_embeddings.items():
            is_match, similarity, distance = self.compare_faces(
                query_embedding, ref_embedding
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_distance = distance
                best_match = person_id
        
        return best_match, best_similarity, best_distance