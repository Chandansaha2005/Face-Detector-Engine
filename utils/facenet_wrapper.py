import tensorflow as tf
import numpy as np
import cv2
import os

class FaceNetRecognizer:
    """Simplified face recognition with OpenFace"""
    
    def __init__(self, model_path='openface_model/nn4.small2.v1.h5'):
        print("Loading OpenFace model...")
        
        try:
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            print(f"✓ OpenFace model loaded: {model_path}")
            self.model_loaded = True
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            print("Using simple face detection only")
            self.model_loaded = False
            self.model = None
        
        # Use OpenCV's face detector as fallback
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Input size for OpenFace
        self.input_shape = (96, 96)
    
    def detect_faces_opencv(self, image):
        """Detect faces using OpenCV Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                'bbox': (x, y, x+w, y+h),
                'face': image[y:y+h, x:x+w]
            })
        
        return results
    
    def preprocess_face(self, face_image):
        """Preprocess face for OpenFace model"""
        # Resize to 96x96 (OpenFace requirement)
        face_resized = cv2.resize(face_image, self.input_shape)
        
        # Convert to float32
        face_float = face_resized.astype('float32')
        
        # Normalize to [0, 1]
        face_normalized = face_float / 255.0
        
        # Add batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def get_embedding(self, face_image):
        """Get embedding from face"""
        if not self.model_loaded:
            return None
        
        try:
            processed = self.preprocess_face(face_image)
            embedding = self.model.predict(processed, verbose=0)[0]
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            return embedding
        except:
            return None
    
    def compare_faces(self, embedding1, embedding2, threshold=0.5):
        """Compare two embeddings"""
        if embedding1 is None or embedding2 is None:
            return False, 0.0, 1.0
        
        distance = np.linalg.norm(embedding1 - embedding2)
        similarity = 1 - min(distance / 2.0, 1.0)
        is_match = distance < threshold
        
        return is_match, similarity, distance