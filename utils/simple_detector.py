import cv2
import numpy as np
import mediapipe as mp

class SimpleFaceFinder:
    """Simple face finder for video processing"""
    
    def __init__(self):
        print("Initializing Simple Face Finder...")
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for long-range (CCTV)
            min_detection_confidence=0.5
        )
        
        print("✓ Face detector ready")
    
    def extract_face_from_photo(self, photo_path):
        """Extract face encoding from a single photo"""
        print(f"Processing reference photo: {photo_path}")
        
        # Read image
        image = cv2.imread(photo_path)
        if image is None:
            print(f"✗ Cannot read image: {photo_path}")
            return None
        
        # Detect faces
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            print("✗ No face found in reference photo")
            return None
        
        # Use the first (largest) face
        face_data = faces[0]
        print(f"✓ Found face in reference photo")
        print(f"  Face size: {face_data['face'].shape}")
        
        # Create simple signature
        signature = self.create_signature(face_data['face'])
        
        return {
            'signature': signature,
            'face_image': face_data['face'],
            'bbox': face_data['bbox']
        }
    
    def detect_faces(self, frame):
        """Detect faces in a frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process for face detection
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                box_width = int(bboxC.width * w)
                box_height = int(bboxC.height * h)
                
                # Add padding
                padding = 15
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x1 + box_width + 2*padding)
                y2 = min(h, y1 + box_height + 2*padding)
                
                # Extract face
                face_roi = frame[y1:y2, x1:x2]
                
                if face_roi.size > 0:
                    faces.append({
                        'bbox': (x1, y1, x2, y2),
                        'face': face_roi,
                        'confidence': float(detection.score[0])
                    })
        
        return faces
    
    def create_signature(self, face_image):
        """Create a simple signature for face matching"""
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        gray = cv2.resize(gray, (100, 100))
        
        # Apply histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        # Flatten and normalize
        signature = gray.flatten().astype('float32') / 255.0
        
        # Normalize to unit vector
        norm = np.linalg.norm(signature)
        if norm > 0:
            signature = signature / norm
        
        return signature
    
    def compare_faces(self, sig1, sig2):
        """Compare two face signatures (returns percentage in -100..100)"""
        if sig1 is None or sig2 is None:
            return 0.0

        # Calculate cosine similarity and guard numerical issues
        similarity = float(np.dot(sig1, sig2))
        if np.isnan(similarity):
            return 0.0
        similarity = max(-1.0, min(1.0, similarity))

        # Convert to percentage
        percentage = similarity * 100.0

        return percentage
    
    def process_video_frame(self, frame, reference_signature, threshold=60):
        """Process a single video frame"""
        # Detect faces
        faces = self.detect_faces(frame)
        
        matches = []
        
        for face in faces:
            # Create signature for detected face
            face_sig = self.create_signature(face['face'])
            
            # Compare with reference
            match_percentage = self.compare_faces(reference_signature, face_sig)
            
            # Check if it's a match
            is_match = match_percentage >= threshold
            
            matches.append({
                'bbox': face['bbox'],
                'match_percentage': match_percentage,
                'is_match': is_match,
                'confidence': face['confidence']
            })
        
        return matches