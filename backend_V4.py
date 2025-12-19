import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ARCFACE MODEL - STATE-OF-THE-ART FACE RECOGNITION
# ============================================================================

class ArcFaceModel(nn.Module):
    """ArcFace model for highly accurate face recognition"""
    def __init__(self, embedding_size=512):
        super(ArcFaceModel, self).__init__()
        import torchvision.models as models
        
        # Use ResNet50 as backbone
        backbone = models.resnet50(pretrained=True)
        
        # Remove last layers
        modules = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        
        # Adaptive pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bottleneck(x)
        
        # L2 normalization (important for face recognition)
        x = F.normalize(x, p=2, dim=1)
        return x

# ============================================================================
# FACE ALIGNMENT FOR BETTER ACCURACY
# ============================================================================

class FaceAligner:
    """Align faces using facial landmarks for better recognition"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
    
    def align_face(self, face_image):
        """Align face to standard position using landmarks"""
        try:
            # Get facial landmarks
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return face_image  # Return original if no landmarks
            
            # Get eye landmarks
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = face_image.shape[:2]
            
            # Left eye (average of multiple points)
            left_eye_points = [33, 133, 157, 158, 159, 160, 161, 173]
            left_eye_x = sum([landmarks[i].x for i in left_eye_points]) / len(left_eye_points)
            left_eye_y = sum([landmarks[i].y for i in left_eye_points]) / len(left_eye_points)
            
            # Right eye (average of multiple points)
            right_eye_points = [362, 263, 383, 384, 385, 386, 387, 398]
            right_eye_x = sum([landmarks[i].x for i in right_eye_points]) / len(right_eye_points)
            right_eye_y = sum([landmarks[i].y for i in right_eye_points]) / len(right_eye_points)
            
            # Calculate angle between eyes
            delta_x = right_eye_x - left_eye_x
            delta_y = right_eye_y - left_eye_y
            angle = np.degrees(np.arctan2(delta_y, delta_x))
            
            # Calculate center between eyes
            eyes_center = ((left_eye_x + right_eye_x) * 0.5 * w,
                          (left_eye_y + right_eye_y) * 0.5 * h)
            
            # Rotate image to align eyes horizontally
            M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
            aligned = cv2.warpAffine(face_image, M, (w, h), flags=cv2.INTER_CUBIC)
            
            return aligned
            
        except Exception as e:
            return face_image

# ============================================================================
# MAIN ACCURATE FACE RECOGNIZER
# ============================================================================

class UltraAccurateFaceRecognizer:
    """ULTRA-ACCURATE face recognition with strict filtering"""
    
    def __init__(self):
        print("="*70)
        print("DRISTI - ULTRA ACCURATE FACE RECOGNITION")
        print("State-of-the-art accuracy with strict filtering")
        print("="*70)
        
        # Initialize face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Long-range detection
            min_detection_confidence=0.8  # HIGH confidence for detection
        )
        
        # Load advanced face recognition model
        self.face_model = self.load_advanced_model()
        
        # Face aligner for better accuracy
        self.face_aligner = FaceAligner()
        
        # Target information
        self.target_embedding = None
        self.target_name = ""
        self.target_face = None
        
        # STRICT SETTINGS - for CCTV footage
        self.similarity_threshold = 0.82  # 82% - STRICT but reasonable
        self.detection_confidence = 0.7   # 70% for CCTV footage
        self.face_size_threshold = 5000   # Minimum face area in pixels for CCTV
        self.face_quality_threshold = 0.6 # Overall face quality score for CCTV
        
        # Performance settings
        self.show_preview = True
        self.frame_skip = 2  # Process every 2nd frame for better accuracy
        self.ensemble_voting = 3  # Need multiple frames to confirm match
        
        # Match tracking
        self.match_history = {}  # Track matches over multiple frames
        self.min_consecutive_matches = 2  # Need 2 consecutive frame matches
        
        # Create directories
        self.create_directories()
        
        print("✓ System initialized with ULTRA-ACCURATE settings")
        print(f"  Similarity threshold: {self.similarity_threshold*100}% (STRICT)")
        print(f"  Detection confidence: {self.detection_confidence*100}%")
        print(f"  Requires {self.min_consecutive_matches} consecutive matches")
        print("="*70)
    
    def load_advanced_model(self):
        """Load advanced face recognition model"""
        print("Loading advanced face recognition model...")
        
        try:
            model = ArcFaceModel(embedding_size=512)
            model.eval()
            print("✓ ArcFace-like model loaded")
            
            # Also try to load pre-trained weights if available
            model_path = 'models/arcface_model.pth'
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print("✓ Pre-trained weights loaded")
            
            return model
            
        except Exception as e:
            print(f"✗ Could not load ArcFace: {e}")
            print("  Using ResNet50 with modifications...")
            import torchvision.models as models
            model = models.resnet50(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
            model.eval()
            return model
    
    def create_directories(self):
        """Create all necessary directories"""
        dirs = [
            'output',
            'output/evidence',
            'output/reports',
            'output/snapshots',
            'output/alerts',
            'input',
            'models'
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def check_face_quality(self, face_image):
        """Check if face is of good quality (not blurry, good lighting)"""
        if face_image is None or face_image.size == 0:
            return 0.0
        
        try:
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # 1. Check face size
            h, w = gray.shape
            size_score = min(1.0, (h * w) / 5000)  # Normalized to 5000 pixels
            
            # 2. Check blurriness using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(1.0, blur_score / 500)  # Normalize
            
            # Combined quality score
            quality = (size_score * 0.3 + blur_score * 0.7)
            
            return quality
            
        except Exception as e:
            return 0.5  # Default medium quality
    
    def align_and_preprocess(self, face_image):
        """Align face and preprocess for recognition"""
        try:
            # Align face
            aligned = self.face_aligner.align_face(face_image)
            
            # Resize to standard size
            aligned = cv2.resize(aligned, (112, 112))  # Standard face recognition size
            
            return aligned
            
        except Exception as e:
            # Fallback: simple resize
            return cv2.resize(face_image, (112, 112))
    
    def extract_face_embedding(self, face_image):
        """Extract high-quality embedding with preprocessing"""
        try:
            # Preprocess and align face
            processed_face = self.align_and_preprocess(face_image)
            
            # Transform for model
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                   std=[0.5, 0.5, 0.5])
            ])
            
            # Convert to tensor
            img_tensor = transform(processed_face).unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.face_model(img_tensor)
            
            # Convert to numpy and normalize
            embedding = embedding.squeeze().numpy()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
            
            return embedding
            
        except Exception as e:
            # Fallback: use simple features
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
            gray = cv2.resize(gray, (100, 100))
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
    
    def set_target_person(self, image_path, person_name=""):
        """Set target person with FLEXIBLE requirements"""
        print(f"\n🎯 SETTING TARGET PERSON")
        print("-" * 60)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"✗ Cannot read image: {image_path}")
            return False
        
        # Try with VERY FLEXIBLE detection settings
        face_detection_flexible = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.3  # VERY LOW confidence for target
        )
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with flexible settings
        results = face_detection_flexible.process(rgb_image)
        
        faces = []
        
        if results.detections:
            for detection in results.detections:
                confidence = detection.score[0]
                
                # VERY FLEXIBLE for target: Only 30% confidence
                if confidence < 0.3:
                    continue
                
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                box_width = int(bboxC.width * w)
                box_height = int(bboxC.height * h)
                
                # VERY FLEXIBLE size: Minimum 30x30 pixels for target
                min_target_size = 900  # 30x30 pixels
                if box_width * box_height < min_target_size:
                    continue
                
                # Add generous padding for target
                padding = int(min(box_width, box_height) * 0.4)  # 40% padding
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x1 + box_width + 2*padding)
                y2 = min(h, y1 + box_height + 2*padding)
                
                # Extract face
                face_roi = image[y1:y2, x1:x2]
                
                if face_roi.size == 0:
                    continue
                
                # Resize if too small for processing
                if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
                    face_roi = cv2.resize(face_roi, (150, 150))
                
                # NO quality check for target - accept any face
                faces.append({
                    'bbox': (x1, y1, x2, y2),
                    'face': face_roi,
                    'confidence': float(confidence),
                    'size': box_width * box_height
                })
        
        # If no faces found with MediaPipe, try traditional method
        if len(faces) == 0:
            print("  Trying alternative face detection...")
            faces = self.detect_faces_traditional(image)
        
        if len(faces) == 0:
            print("✗ No face found in target image")
            print("  Please use a photo with a visible face")
            return False
        
        # Use the largest face
        best_face = max(faces, key=lambda x: x['size'])
        
        print(f"✓ Face detected (Size: {best_face['face'].shape}, Confidence: {best_face['confidence']:.1%})")
        
        # Extract embedding - try multiple times with different preprocessing
        embeddings = []
        
        # Try original face
        embedding1 = self.extract_face_embedding(best_face['face'])
        if embedding1 is not None:
            embeddings.append(embedding1)
        
        # Try with brightness adjustment
        try:
            face_bright = cv2.convertScaleAbs(best_face['face'], alpha=1.2, beta=20)
            embedding2 = self.extract_face_embedding(face_bright)
            if embedding2 is not None:
                embeddings.append(embedding2)
        except:
            pass
        
        # Try with contrast adjustment
        try:
            face_contrast = cv2.convertScaleAbs(best_face['face'], alpha=1.5, beta=0)
            embedding3 = self.extract_face_embedding(face_contrast)
            if embedding3 is not None:
                embeddings.append(embedding3)
        except:
            pass
        
        if not embeddings:
            print("✗ Could not extract face features")
            print("  Please try a clearer photo")
            return False
        
        # Average embeddings for robustness
        self.target_embedding = np.mean(embeddings, axis=0)
        self.target_embedding = self.target_embedding / (np.linalg.norm(self.target_embedding) + 1e-10)
        
        self.target_name = person_name or os.path.basename(image_path).split('.')[0]
        self.target_face = best_face['face']
        
        # Save target face
        target_path = f'output/target_{self.target_name}.jpg'
        cv2.imwrite(target_path, self.target_face)
        
        print(f"✅ TARGET SET: {self.target_name}")
        print(f"  Face size: {best_face['face'].shape}")
        print(f"  Embedding ready for matching")
        print(f"  Saved to: {target_path}")
        
        # Show target
        cv2.imshow(f'Target: {self.target_name}', cv2.resize(self.target_face, (400, 400)))
        cv2.waitKey(1500)
        cv2.destroyAllWindows()
        
        return True
    
    def detect_faces_traditional(self, image):
        """Traditional face detection as fallback"""
        faces = []
        
        try:
            # Try Haar cascade as fallback
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Enhance image for better detection
                gray = cv2.equalizeHist(gray)
                
                detected = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,  # More sensitive
                    minNeighbors=3,    # Less strict
                    minSize=(20, 20)   # Very small faces
                )
                
                for (x, y, w, h) in detected:
                    # Add padding
                    padding = int(min(w, h) * 0.4)
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(image.shape[1], x + w + padding)
                    y2 = min(image.shape[0], y + h + padding)
                    
                    face_roi = image[y1:y2, x1:x2]
                    
                    if face_roi.size > 0:
                        # Resize if too small
                        if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
                            face_roi = cv2.resize(face_roi, (150, 150))
                        
                        faces.append({
                            'bbox': (x1, y1, x2, y2),
                            'face': face_roi,
                            'confidence': 0.7,  # Default confidence
                            'size': w * h
                        })
        except Exception as e:
            print(f"  Traditional detection failed: {e}")
        
        return faces
    
    def detect_faces_strict(self, image):
        """Detect faces in CCTV footage with strict filtering"""
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.face_detection.process(rgb_image)
        
        faces = []
        
        if results.detections:
            for detection in results.detections:
                confidence = detection.score[0]
                
                # Strict for CCTV: 70% confidence minimum
                if confidence < self.detection_confidence:
                    continue
                
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                box_width = int(bboxC.width * w)
                box_height = int(bboxC.height * h)
                
                # Check minimum size for CCTV
                if box_width * box_height < self.face_size_threshold:
                    continue
                
                # Add padding (less padding for CCTV)
                padding = int(min(box_width, box_height) * 0.15)
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x1 + box_width + 2*padding)
                y2 = min(h, y1 + box_height + 2*padding)
                
                # Extract face
                face_roi = image[y1:y2, x1:x2]
                
                if face_roi.size == 0:
                    continue
                
                # Check face quality for CCTV
                quality = self.check_face_quality(face_roi)
                if quality < self.face_quality_threshold:
                    continue
                
                faces.append({
                    'bbox': (x1, y1, x2, y2),
                    'face': face_roi,
                    'confidence': float(confidence),
                    'quality': quality,
                    'size': box_width * box_height
                })
        
        # Sort by size (largest faces first)
        faces.sort(key=lambda x: x['size'], reverse=True)
        
        return faces
    
    def calculate_similarity_strict(self, emb1, emb2):
        """Calculate similarity with strict checks"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        try:
            # Ensure same dimension
            min_len = min(len(emb1), len(emb2))
            emb1_trunc = emb1[:min_len]
            emb2_trunc = emb2[:min_len]
            
            # Cosine similarity
            similarity = np.dot(emb1_trunc, emb2_trunc) / (
                np.linalg.norm(emb1_trunc) * np.linalg.norm(emb2_trunc) + 1e-10
            )
            
            # Ensure valid range
            similarity = max(-1.0, min(1.0, similarity))
            
            return float(similarity)
            
        except Exception as e:
            return 0.0
    
    def track_face_match(self, face_bbox, similarity, frame_number):
        """Track face matches across multiple frames"""
        # Create location key based on normalized position
        x1, y1, x2, y2 = face_bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        location_key = f"{center_x//50}_{center_y//50}"  # 50px grid
        
        if location_key not in self.match_history:
            self.match_history[location_key] = []
        
        # Add current match
        self.match_history[location_key].append({
            'frame': frame_number,
            'similarity': similarity,
            'time': time.time(),
            'bbox': face_bbox
        })
        
        # Remove old entries (older than 3 seconds)
        current_time = time.time()
        self.match_history[location_key] = [
            m for m in self.match_history[location_key]
            if current_time - m['time'] < 3.0
        ]
        
        # Check if we have enough consecutive matches
        if len(self.match_history[location_key]) >= self.min_consecutive_matches:
            # Calculate average similarity
            similarities = [m['similarity'] for m in self.match_history[location_key]]
            avg_similarity = np.mean(similarities)
            
            # Check variance (consistent matches)
            variance = np.var(similarities)
            if variance < 0.02:  # Low variance = consistent detection
                return True, avg_similarity
        
        return False, similarity
    
    def search_in_video(self, video_path, camera_id="CCTV_1"):
        """Search for target with ULTRA-ACCURATE matching"""
        print(f"\n🔍 ULTRA-ACCURATE SEARCH: {camera_id}")
        print(f"  Video: {os.path.basename(video_path)}")
        print(f"  Settings: {self.min_consecutive_matches} consecutive matches required")
        print("-" * 60)
        
        if self.target_embedding is None:
            print("✗ Target person not set")
            return None
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"✗ Cannot open video: {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0:
            fps = 30
        
        print(f"  Resolution: {width}x{height}, FPS: {fps}")
        print(f"  Duration: {total_frames/fps:.1f}s, Frames: {total_frames}")
        
        # Prepare output video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/evidence/{camera_id}_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps/self.frame_skip, (width, height))
        
        # Tracking
        confirmed_matches = []
        frame_count = 0
        match_count = 0
        start_time = time.time()
        
        # Reset match history for this search
        self.match_history = {}
        
        print("  Processing with STRICT validation...")
        print("  Controls: [q] Quit | [p] Pause | [s] Save frame")
        
        pause = False
        
        while True:
            if not pause:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for processing speed
                if frame_count % self.frame_skip != 0:
                    continue
                
                # Detect faces with strict filtering
                faces = self.detect_faces_strict(frame)
                
                # Check each face
                current_match = None
                match_similarity = 0.0
                
                for face in faces:
                    # Skip if quality is poor
                    if face['quality'] < self.face_quality_threshold:
                        continue
                    
                    # Extract embedding
                    face_embedding = self.extract_face_embedding(face['face'])
                    if face_embedding is None:
                        continue
                    
                    # Calculate similarity
                    similarity = self.calculate_similarity_strict(
                        self.target_embedding, 
                        face_embedding
                    )
                    
                    # STRICT: Only consider high similarity
                    if similarity < self.similarity_threshold:
                        continue
                    
                    # Track across frames
                    is_confirmed, tracked_similarity = self.track_face_match(
                        face['bbox'], similarity, frame_count
                    )
                    
                    if is_confirmed:
                        match_count += 1
                        current_match = {
                            'bbox': face['bbox'],
                            'similarity': tracked_similarity,
                            'frame': frame_count,
                            'time': frame_count / fps,
                            'quality': face['quality']
                        }
                        confirmed_matches.append(current_match)
                        
                        # Save evidence
                        if match_count == 1:
                            # Save first confirmed match
                            snap_path = f"output/snapshots/{camera_id}_confirmed_match.jpg"
                            cv2.imwrite(snap_path, face['face'])
                        
                        # Save alert
                        alert_path = f"output/alerts/{camera_id}_alert_{match_count}.jpg"
                        cv2.imwrite(alert_path, frame)
                
                # Draw results
                result_frame = self.draw_detections_ultra( frame, faces, current_match, camera_id, frame_count, total_frames, match_count, fps  # ADD fps HERE
                )
                
                # Write to output
                out.write(result_frame)
                
                # Show preview
                if self.show_preview:
                    preview = cv2.resize(result_frame, (900, 506))  # 16:9 aspect
                    cv2.imshow(f'DRISTI ULTRA - {camera_id}', preview)
            
            # Handle key presses
            if self.show_preview:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    print("  ⏹️  Stopped by user")
                    break
                elif key == ord('p'):  # Pause/Resume
                    pause = not pause
                    if pause:
                        print("  ⏸️  Paused - Press 'p' to resume")
                    else:
                        print("  ▶️  Resumed")
                elif key == ord('s'):  # Save current frame
                    save_path = f"output/snapshots/{camera_id}_frame_{frame_count}.jpg"
                    cv2.imwrite(save_path, result_frame)
                    print(f"  💾 Frame saved: {save_path}")
            
            # Show progress
            if total_frames > 0 and frame_count % max(1, total_frames // 20) == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                fps_actual = frame_count / max(elapsed, 0.1)
                print(f"    ↳ {progress:.0f}% | Time: {elapsed:.1f}s | FPS: {fps_actual:.1f} | Matches: {match_count}")
        
        # Cleanup
        cap.release()
        out.release()
        if self.show_preview:
            cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"    ✓ Search completed in {elapsed:.1f}s")
        print(f"    ✓ Confirmed matches found: {match_count}")
        
        # Generate detailed report
        if confirmed_matches:
            self.generate_detailed_report(camera_id, video_path, output_path, confirmed_matches, elapsed)
        else:
            print(f"    ℹ️  No confirmed matches found")
            print(f"    Try lowering threshold (current: {self.similarity_threshold*100}%)")
        
        return {
            'camera': camera_id,
            'matches': match_count,
            'confirmed_matches': confirmed_matches,
            'processing_time': elapsed,
            'output_video': output_path
        }

    def draw_detections_ultra(self, frame, faces, match_info=None, camera_name="", 
                             current_frame=0, total_frames=0, total_matches=0, fps=30):
        """Draw detections with detailed information"""
        result = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw all detected faces
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            confidence = face['confidence']
            quality = face.get('quality', 0.0)
            
            # Check if this is the confirmed match
            is_match = False
            if match_info and match_info.get('bbox') == (x1, y1, x2, y2):
                is_match = True
            
            if is_match:
                # CONFIRMED MATCH - Bright green with animation
                color = (0, 255, 0)  # Bright green
                thickness = 4
                
                # Pulsing effect for confirmed matches
                pulse = int(127 * (1 + np.sin(time.time() * 5))) + 128
                overlay = result.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, pulse, 0), -1)
                cv2.addWeighted(overlay, 0.1, result, 0.9, 0, result)
                
                label = f"✅ {self.target_name}: {match_info['similarity']:.1%}"
                
            elif quality > self.face_quality_threshold:
                # HIGH QUALITY FACE - Blue for potential matches
                color = (255, 200, 0)  # Blueish
                thickness = 2
                label = f"👤 Face: {confidence:.0%}"
            else:
                # LOW QUALITY - Gray (barely visible)
                color = (80, 80, 80)
                thickness = 1
                label = f"Low quality"
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result,
                        (x1, y2 - label_size[1] - 5),
                        (x1 + label_size[0], y2),
                        color, cv2.FILLED)
            
            # Draw label text
            cv2.putText(result, label, (x1, y2 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add comprehensive status bar
        status_bar = np.zeros((70, width, 3), dtype=np.uint8)
        status_bar[:] = (20, 20, 40)  # Dark blue
        result[:70, :] = status_bar
        
        # System info (left)
        cv2.putText(result, f"DRISTI ULTRA", (10, 25),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(result, f"Target: {self.target_name}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
        
        # Frame info (center)
        frame_text = f"Frame: {current_frame}/{total_frames}"
        cv2.putText(result, frame_text, (width // 3, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Fixed: Use fps parameter with default value
        time_text = f"Time: {current_frame/max(fps, 1):.1f}s"
        cv2.putText(result, time_text, (width // 3, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Match info (right)
        if total_matches > 0:
            match_color = (0, 255, 0)
        else:
            match_color = (255, 255, 255)
        
        cv2.putText(result, f"Confirmed Matches: {total_matches}", (width - 200, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2)
        
        threshold_text = f"Threshold: {self.similarity_threshold*100:.0f}%"
        cv2.putText(result, threshold_text, (width - 200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(result, timestamp, (width - 250, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Add camera name at bottom left
        cv2.putText(result, f"Camera: {camera_name}", (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Add controls info at bottom
        if self.show_preview:
            controls_bar = np.zeros((35, width, 3), dtype=np.uint8)
            controls_bar[:] = (30, 30, 30)
            result[height-35:height, :] = controls_bar
            
            controls = "[q] Quit | [p] Pause | [s] Save Frame"
            cv2.putText(result, controls, (width // 2 - 150, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return result
        
    def generate_detailed_report(self, camera_name, video_path, output_path, matches, processing_time):
        """Generate ultra-detailed report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"output/reports/{camera_name}_ultra_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DRISTI ULTRA - DETAILED FACE RECOGNITION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"🔍 TARGET PERSON: {self.target_name}\n")
            f.write(f"📹 CAMERA: {camera_name}\n")
            f.write(f"🎥 SOURCE VIDEO: {os.path.basename(video_path)}\n")
            f.write(f"⏰ SEARCH TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"⚡ PROCESSING TIME: {processing_time:.1f} seconds\n\n")
            
            f.write("="*80 + "\n")
            f.write("📊 SYSTEM SETTINGS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"• Similarity Threshold: {self.similarity_threshold*100:.0f}%\n")
            f.write(f"• Detection Confidence: {self.detection_confidence*100:.0f}%\n")
            f.write(f"• Face Quality Threshold: {self.face_quality_threshold*100:.0f}%\n")
            f.write(f"• Min Face Size: {self.face_size_threshold} pixels\n")
            f.write(f"• Consecutive Matches Required: {self.min_consecutive_matches}\n")
            f.write(f"• Frame Skip: {self.frame_skip}x\n\n")
            
            f.write("="*80 + "\n")
            f.write("✅ CONFIRMED MATCHES FOUND\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Confirmed Matches: {len(matches)}\n\n")
            
            if matches:
                # Group matches by time clusters
                time_groups = []
                current_group = [matches[0]]
                
                for i in range(1, len(matches)):
                    time_diff = matches[i]['time'] - matches[i-1]['time']
                    if time_diff < 5.0:  # 5 seconds between matches = same appearance
                        current_group.append(matches[i])
                    else:
                        time_groups.append(current_group)
                        current_group = [matches[i]]
                
                if current_group:
                    time_groups.append(current_group)
                
                f.write(f"📈 APPEARANCES DETECTED: {len(time_groups)}\n")
                f.write("-" * 40 + "\n\n")
                
                for idx, group in enumerate(time_groups, 1):
                    best_match = max(group, key=lambda x: x['similarity'])
                    avg_similarity = np.mean([m['similarity'] for m in group])
                    start_time = group[0]['time']
                    end_time = group[-1]['time']
                    duration = end_time - start_time
                    
                    f.write(f"APPEARANCE #{idx}:\n")
                    f.write(f"  ⏱️  Duration: {duration:.1f} seconds\n")
                    f.write(f"  🕐 Time: {start_time:.1f}s to {end_time:.1f}s\n")
                    f.write(f"  🎯 Best Similarity: {best_match['similarity']:.1%}\n")
                    f.write(f"  📊 Average Similarity: {avg_similarity:.1%}\n")
                    f.write(f"  📈 Detection Quality: {best_match['quality']:.1%}\n")
                    f.write(f"  📍 Frames: {group[0]['frame']} to {group[-1]['frame']}\n")
                    f.write(f"  🔢 Match Count: {len(group)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("📁 EVIDENCE FILES\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"1. Marked Video Evidence: {os.path.basename(output_path)}\n")
            f.write(f"2. Target Face: target_{self.target_name}.jpg\n")
            if matches:
                f.write(f"3. Match Snapshots: {camera_name}_*.jpg in snapshots folder\n")
                f.write(f"4. Alert Images: {camera_name}_alert_*.jpg in alerts folder\n")
            f.write(f"5. This Report: {os.path.basename(report_file)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("🔬 VERDICT\n")
            f.write("="*80 + "\n\n")
            
            if matches:
                best_similarity = max([m['similarity'] for m in matches])
                if best_similarity > 0.9:
                    f.write("✅✅ POSITIVE IDENTIFICATION - VERY HIGH CONFIDENCE\n")
                elif best_similarity > 0.85:
                    f.write("✅ LIKELY IDENTIFICATION - HIGH CONFIDENCE\n")
                else:
                    f.write("⚠️ POSSIBLE MATCH - MODERATE CONFIDENCE\n")
            else:
                f.write("❌ NO MATCH FOUND\n")
                f.write("   The target person was not detected in this video.\n")
            
            f.write("="*80 + "\n")

# ============================================================================
# SIMPLIFIED MENU SYSTEM
# ============================================================================

class DristiUltraMenu:
    """Menu system for DRISTI ULTRA"""
    
    def __init__(self):
        self.recognizer = UltraAccurateFaceRecognizer()
        self.cctv_videos = self.scan_videos()
    
    def scan_videos(self):
        """Scan for CCTV videos"""
        videos = {}
        if os.path.exists('input'):
            for file in os.listdir('input'):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    cam_name = os.path.splitext(file)[0]
                    videos[cam_name] = f"input/{file}"
        return videos
    
    def display_header(self):
        """Display header"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*70)
        print("🚀 DRISTI ULTRA - ULTRA ACCURATE FACE RECOGNITION")
        print("="*70)
        print(f"Target: {self.recognizer.target_name if self.recognizer.target_name else 'Not Set'}")
        print(f"CCTV Videos: {len(self.cctv_videos)} found")
        print(f"Accuracy Mode: ULTRA (Threshold: {self.recognizer.similarity_threshold*100:.0f}%)")
        print("="*70)
    
    def main_menu(self):
        """Main menu"""
        while True:
            self.display_header()
            
            print("\n📋 MAIN MENU")
            print("-" * 40)
            print("1. 🎯 Set Target Person (ANY photo accepted)")
            print("2. 🔍 Search in Single CCTV (ULTRA ACCURATE)")
            print("3. ⚡ Adjust Accuracy Settings")
            print("4. 📁 View Files")
            print("5. 🚪 Exit")
            print("-" * 40)
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                self.set_target()
            elif choice == "2":
                self.search_single()
            elif choice == "3":
                self.adjust_settings()
            elif choice == "4":
                self.view_files()
            elif choice == "5":
                print("\nThank you for using DRISTI ULTRA!")
                break
            else:
                print("Invalid option")
                input("Press Enter to continue...")
    
    def set_target(self):
        """Set target person - accepts ANY photo with face"""
        self.display_header()
        print("\n🎯 SET TARGET PERSON")
        print("-" * 40)
        print("ANY photo with a visible face will be accepted")
        print("Small, blurry, or low-quality photos are OK")
        print("-" * 40)
        
        # Find photos
        photos = []
        if os.path.exists('input'):
            photos = [f for f in os.listdir('input') 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if photos:
            print("\nAvailable photos in 'input/' folder:")
            for i, p in enumerate(photos, 1):
                print(f"{i}. {p}")
            print(f"{len(photos)+1}. Enter custom path")
        else:
            print("\nNo photos found in input folder.")
            print("Enter custom path or add photos to 'input/' folder:")
        
        try:
            if photos:
                choice = input(f"\nSelect option (1-{len(photos)+1}): ").strip()
                if choice.isdigit():
                    choice = int(choice)
                    if 1 <= choice <= len(photos):
                        path = f"input/{photos[choice-1]}"
                    elif choice == len(photos) + 1:
                        path = input("Enter photo path: ").strip()
                    else:
                        print("Invalid selection")
                        return
                else:
                    path = choice
            else:
                path = input("Enter photo path: ").strip()
            
            name = input("Person name (optional): ").strip()
            
            print(f"\nProcessing target photo...")
            print("(Even small/blurry photos will be accepted)")
            
            if self.recognizer.set_target_person(path, name):
                print(f"\n✅ Target set: {self.recognizer.target_name}")
            else:
                print("\n❌ Failed to set target")
                print("Please try another photo with a visible face")
        
        except Exception as e:
            print(f"Error: {e}")
        
        input("\nPress Enter to continue...")
    
    def search_single(self):
        """Search in single CCTV"""
        if self.recognizer.target_embedding is None:
            print("⚠️  Set target person first!")
            input("\nPress Enter to continue...")
            return
        
        if not self.cctv_videos:
            print("No CCTV videos found in 'input/' folder")
            print("Add videos as MP4, AVI, etc. to the 'input' folder")
            input("\nPress Enter to continue...")
            return
        
        self.display_header()
        print("\n🔍 ULTRA-ACCURATE SEARCH")
        print("-" * 40)
        
        print("Available CCTV cameras:")
        for i, (cam_name, path) in enumerate(self.cctv_videos.items(), 1):
            size_mb = os.path.getsize(path) / (1024*1024) if os.path.exists(path) else 0
            print(f"{i}. {cam_name} ({size_mb:.1f} MB)")
        
        try:
            choice = int(input(f"\nSelect camera (1-{len(self.cctv_videos)}): "))
            if 1 <= choice <= len(self.cctv_videos):
                cam_name = list(self.cctv_videos.keys())[choice-1]
                video_path = self.cctv_videos[cam_name]
                
                print(f"\nStarting ULTRA-ACCURATE search in {cam_name}...")
                print("This mode uses strict filtering for maximum accuracy")
                print("Only clear, high-quality matches will be detected")
                
                self.recognizer.search_in_video(video_path, cam_name)
                
                print("\n✅ Search completed!")
                print("   Check 'output/' folder for detailed results.")
            else:
                print("Invalid selection")
        except ValueError:
            print("Please enter a number")
        
        input("\nPress Enter to continue...")
    
    def adjust_settings(self):
        """Adjust accuracy settings"""
        self.display_header()
        print("\n⚡ ADJUST ACCURACY SETTINGS")
        print("-" * 40)
        
        print("Current ULTRA-ACCURATE settings:")
        print(f"1. Similarity Threshold: {self.recognizer.similarity_threshold*100:.0f}% (Higher = stricter)")
        print(f"2. Detection Confidence: {self.recognizer.detection_confidence*100:.0f}%")
        print(f"3. Face Quality Threshold: {self.recognizer.face_quality_threshold*100:.0f}%")
        print(f"4. Consecutive Matches: {self.recognizer.min_consecutive_matches}")
        print(f"5. Min Face Size (CCTV): {self.recognizer.face_size_threshold} pixels")
        
        print("\nSelect setting to adjust (1-5, or 0 to go back):")
        
        try:
            choice = input("\nYour choice: ").strip()
            
            if choice == "1":
                new_val = float(input(f"New similarity threshold (0.7-0.95, current: {self.recognizer.similarity_threshold}): "))
                if 0.7 <= new_val <= 0.95:
                    self.recognizer.similarity_threshold = new_val
                    print(f"✓ Threshold set to {new_val*100:.0f}%")
                    if new_val > 0.85:
                        print("  Warning: Very high threshold may miss some matches")
                else:
                    print("Invalid value (must be between 0.7 and 0.95)")
            
            elif choice == "2":
                new_val = float(input(f"New detection confidence (0.5-0.9, current: {self.recognizer.detection_confidence}): "))
                if 0.5 <= new_val <= 0.9:
                    self.recognizer.detection_confidence = new_val
                    print(f"✓ Detection confidence set to {new_val*100:.0f}%")
                else:
                    print("Invalid value")
            
            elif choice == "3":
                new_val = float(input(f"New face quality threshold (0.4-0.8, current: {self.recognizer.face_quality_threshold}): "))
                if 0.4 <= new_val <= 0.8:
                    self.recognizer.face_quality_threshold = new_val
                    print(f"✓ Face quality threshold set to {new_val*100:.0f}%")
                else:
                    print("Invalid value")
            
            elif choice == "4":
                new_val = int(input(f"New consecutive matches (1-5, current: {self.recognizer.min_consecutive_matches}): "))
                if 1 <= new_val <= 5:
                    self.recognizer.min_consecutive_matches = new_val
                    print(f"✓ Consecutive matches set to {new_val}")
                else:
                    print("Invalid value")
            
            elif choice == "5":
                new_val = int(input(f"New min face size (2000-10000, current: {self.recognizer.face_size_threshold}): "))
                if 2000 <= new_val <= 10000:
                    self.recognizer.face_size_threshold = new_val
                    print(f"✓ Minimum face size set to {new_val} pixels")
                else:
                    print("Invalid value")
            
            elif choice == "0":
                return
            
            else:
                print("Invalid selection")
        
        except ValueError:
            print("Please enter a valid number")
        
        input("\nPress Enter to continue...")
    
    def view_files(self):
        """View files"""
        self.display_header()
        print("\n📁 VIEW FILES")
        print("-" * 40)
        
        print("1. View input files")
        print("2. View output files")
        print("3. Back to main menu")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            print("\n📂 INPUT FOLDER:")
            print("-" * 30)
            if os.path.exists('input'):
                files = os.listdir('input')
                if files:
                    for file in files:
                        path = f"input/{file}"
                        if os.path.isfile(path):
                            size_kb = os.path.getsize(path) / 1024
                            print(f"  {file} ({size_kb:.1f} KB)")
                else:
                    print("  Empty folder")
            else:
                print("  Folder does not exist")
        
        elif choice == "2":
            print("\n📂 OUTPUT FOLDER:")
            print("-" * 30)
            if os.path.exists('output'):
                for root, dirs, files in os.walk('output'):
                    level = root.replace('output', '').count(os.sep)
                    indent = '  ' * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = '  ' * (level + 1)
                    for file in files[:15]:  # Show first 15 files
                        print(f"{subindent}{file}")
            else:
                print("  Folder does not exist")
        
        elif choice == "3":
            return
        
        else:
            print("Invalid option")
        
        input("\nPress Enter to continue...")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Starting DRISTI ULTRA - Ultra Accurate Face Recognition...")
    print("\nIMPORTANT: This version accepts ANY photo for target person!")
    print("Required packages:")
    print("pip install opencv-python mediapipe torch torchvision numpy pillow")
    print("="*70)
    
    try:
        # Check for required packages
        try:
            import mediapipe
            import torch
            import torchvision
        except ImportError as e:
            print(f"Missing package: {e}")
            print("Install with: pip install mediapipe torch torchvision")
            input("\nPress Enter to exit...")
            exit(1)
        
        # Start the system
        menu = DristiUltraMenu()
        menu.main_menu()
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")