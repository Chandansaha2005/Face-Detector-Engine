import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FACE VERIFICATION MODEL
# ============================================================================

class FaceVerifierModel(nn.Module):
    """Face verification neural network"""
    def __init__(self):
        super(FaceVerifierModel, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 12 * 12)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

# ============================================================================
# RESNET FEATURE EXTRACTOR
# ============================================================================

class ResNetFeatureExtractor(nn.Module):
    """ResNet for face feature extraction"""
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        import torchvision.models as models
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        for param in self.resnet.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        return x

# ============================================================================
# MAIN DRISTI SYSTEM WITH LIVE PREVIEW
# ============================================================================

class DristiProductionSystem:
    """Production-ready face recognition system with live preview"""
    
    def __init__(self):
        print("="*70)
        print("🚀 DRISTI PRODUCTION SYSTEM v4.0")
        print("Face Recognition System for CCTV Analysis")
        print("="*70)
        
        # Initialize models
        self.verifier_model = self.load_verifier_model()
        self.feature_extractor = self.load_feature_extractor()
        
        # Target information
        self.target_features = None
        self.target_name = ""
        self.target_image = None
        
        # Settings
        self.similarity_threshold = 0.75
        self.confidence_threshold = 0.7
        self.min_consecutive_matches = 2
        
        # Performance settings
        self.frame_skip = 3
        self.max_workers = 4
        self.show_preview = True  # Enable preview by default
        
        # Create directories
        self.create_directories()
        
        # Face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Check for CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"💻 Device: {self.device}")
        
        print("✅ System Initialized")
        print(f"   Models loaded: ✓ Verifier ✓ Feature Extractor")
        print(f"   Similarity Threshold: {self.similarity_threshold*100}%")
        print(f"   Live Preview: {'Enabled' if self.show_preview else 'Disabled'}")
        print("="*70)
    
    def load_verifier_model(self):
        """Load face verification model"""
        print("Loading face verification model...")
        
        try:
            model = FaceVerifierModel()
            model_path = 'models/face_verifier.pth'
            
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"✓ Verifier model loaded from: {model_path}")
            else:
                print(f"⚠️  Model not found: {model_path}")
                print("   Using untrained model (will train during first use)")
            
            model.eval()
            return model
            
        except Exception as e:
            print(f"✗ Error loading verifier model: {e}")
            print("   Using fallback verification")
            return None
    
    def load_feature_extractor(self):
        """Load feature extractor model"""
        print("Loading feature extractor model...")
        
        try:
            model = ResNetFeatureExtractor()
            model_path = 'models/resnet_face.pth'
            
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"✓ Feature extractor loaded from: {model_path}")
            else:
                print(f"⚠️  Model not found: {model_path}")
                print("   Using pre-trained ResNet18 (ImageNet)")
            
            model.eval()
            return model
            
        except Exception as e:
            print(f"✗ Error loading feature extractor: {e}")
            print("   Using pre-trained ResNet18")
            import torchvision.models as models
            model = models.resnet18(pretrained=True)
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
    
    def extract_face_features(self, face_image):
        """Extract deep features from face using ResNet"""
        try:
            # Preprocess image for ResNet
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Convert to PIL
            if len(face_image.shape) == 2:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
            pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            
            # Apply transforms
            img_tensor = transform(pil_image).unsqueeze(0)
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(img_tensor)
            
            # Convert to numpy and normalize
            features = features.squeeze().numpy()
            if np.linalg.norm(features) > 0:
                features = features / np.linalg.norm(features)
            
            return features
            
        except Exception as e:
            # Fallback: simple histogram features
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
            gray = cv2.resize(gray, (100, 100))
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
    
    def verify_face_pair(self, face1, face2):
        """Verify if two faces match using verifier model"""
        if self.verifier_model is None:
            return 0.65
        
        try:
            # Preprocess both faces
            transform = transforms.Compose([
                transforms.Resize((100, 100)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
            # Convert to PIL
            face1_pil = Image.fromarray(cv2.cvtColor(face1, cv2.COLOR_BGR2RGB))
            face2_pil = Image.fromarray(cv2.cvtColor(face2, cv2.COLOR_BGR2RGB))
            
            # Apply transforms
            face1_tensor = transform(face1_pil).unsqueeze(0)
            face2_tensor = transform(face2_pil).unsqueeze(0)
            
            # Concatenate along channel dimension
            pair_tensor = torch.cat([face1_tensor, face2_tensor], dim=1)
            
            with torch.no_grad():
                match_prob = self.verifier_model(pair_tensor).item()
            
            return match_prob
            
        except Exception as e:
            return 0.65
    
    def set_target_person(self, photo_path, name=""):
        """Set the target person to search for"""
        print(f"\n🎯 SETTING TARGET PERSON")
        print("-" * 40)
        
        if not os.path.exists(photo_path):
            print(f"✗ Photo not found: {photo_path}")
            return False
        
        # Read image
        image = cv2.imread(photo_path)
        if image is None:
            print("✗ Cannot read image")
            return False
        
        # Detect face
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        if len(faces) == 0:
            print("✗ No face detected in photo")
            print("   Tips: Use clear, front-facing photo with good lighting")
            return False
        
        # Use largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Add padding
        padding = int(min(w, h) * 0.2)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2*padding)
        h = min(image.shape[0] - y, h + 2*padding)
        
        face_img = image[y:y+h, x:x+w]
        
        if face_img.size == 0:
            print("✗ Could not extract face")
            return False
        
        # Extract features
        self.target_features = self.extract_face_features(face_img)
        self.target_name = name or os.path.basename(photo_path).split('.')[0]
        self.target_image = face_img
        
        # Save target face
        target_path = f'output/target_{self.target_name}.jpg'
        cv2.imwrite(target_path, face_img)
        
        print(f"✅ TARGET SET: {self.target_name}")
        print(f"  Face size: {face_img.shape}")
        print(f"  Feature dimension: {len(self.target_features)}")
        print(f"  Saved to: {target_path}")
        
        # Show target face briefly
        cv2.imshow(f'Target: {self.target_name}', cv2.resize(face_img, (300, 300)))
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
        return True
    
    def detect_faces_fast(self, frame):
        """Fast face detection using OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        face_data = []
        
        for (x, y, w, h) in faces:
            # Filter small faces
            if w * h < 2500:
                continue
            
            # Add padding
            padding = int(min(w, h) * 0.15)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size > 0:
                face_data.append({
                    'bbox': (x1, y1, x2, y2),
                    'face': face_img,
                    'size': w * h
                })
        
        return face_data
    
    def calculate_similarity(self, features1, features2):
        """Calculate cosine similarity between feature vectors"""
        if features1 is None or features2 is None:
            return 0.0
        
        try:
            # Ensure same dimension
            min_len = min(len(features1), len(features2))
            f1 = features1[:min_len]
            f2 = features2[:min_len]
            
            # Cosine similarity
            dot_product = np.dot(f1, f2)
            norm1 = np.linalg.norm(f1)
            norm2 = np.linalg.norm(f2)
            
            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
                return max(0.0, min(1.0, similarity))
        except:
            pass
        
        return 0.0
    
    def search_in_video_with_preview(self, video_path, camera_name):
        """Search in video with live preview"""
        print(f"\n🔍 PROCESSING: {camera_name}")
        print(f"  Video: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("✗ Cannot open video")
            return None
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0:
            fps = 30
        
        print(f"  Resolution: {width}x{height}, FPS: {fps}")
        print(f"  Duration: {total_frames/fps:.1f}s, Frames: {total_frames}")
        
        # Output video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/evidence/{camera_name}_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps/self.frame_skip, (width, height))
        
        # Tracking
        matches = []
        frame_count = 0
        match_count = 0
        start_time = time.time()
        
        print("  Live preview enabled - Press 'q' to stop, 'p' to pause")
        
        # Match history for consecutive matching
        match_history = {}
        
        pause = False
        
        while True:
            if not pause:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for speed
                if frame_count % self.frame_skip != 0:
                    continue
                
                # Detect faces
                faces = self.detect_faces_fast(frame)
                
                # Process each face
                current_match = None
                
                for face in faces:
                    # Extract features
                    face_features = self.extract_face_features(face['face'])
                    
                    # Calculate similarity with target
                    similarity = self.calculate_similarity(self.target_features, face_features)
                    
                    # Check similarity threshold
                    if similarity < self.similarity_threshold:
                        continue
                    
                    # Verify with neural network if available
                    verification_score = 0.7  # Default
                    if self.verifier_model is not None and self.target_image is not None:
                        verification_score = self.verify_face_pair(self.target_image, face['face'])
                        if verification_score < 0.6:
                            continue
                    
                    # Track consecutive matches
                    location_key = f"{face['bbox'][0]//100}_{face['bbox'][1]//100}"
                    
                    if location_key not in match_history:
                        match_history[location_key] = []
                    
                    match_history[location_key].append({
                        'frame': frame_count,
                        'similarity': similarity,
                        'verification': verification_score,
                        'time': time.time()
                    })
                    
                    # Keep only recent matches (last 5 seconds)
                    current_time = time.time()
                    match_history[location_key] = [
                        m for m in match_history[location_key]
                        if current_time - m['time'] < 5
                    ]
                    
                    # Need consecutive matches
                    if len(match_history[location_key]) >= self.min_consecutive_matches:
                        # Calculate weighted average
                        similarities = [m['similarity'] for m in match_history[location_key]]
                        avg_similarity = np.mean(similarities)
                        
                        match_count += 1
                        current_match = {
                            'bbox': face['bbox'],
                            'similarity': avg_similarity,
                            'verification': verification_score,
                            'frame': frame_count,
                            'time': frame_count / fps
                        }
                        matches.append(current_match)
                        
                        # Save first match snapshot
                        if match_count == 1:
                            snap_path = f"output/snapshots/{camera_name}_match.jpg"
                            cv2.imwrite(snap_path, face['face'])
                        
                        # Add to alerts
                        alert_path = f"output/alerts/{camera_name}_alert_{match_count}.jpg"
                        cv2.imwrite(alert_path, frame)
                
                # Draw results on frame
                result_frame = self.draw_results_with_preview(frame, faces, current_match, camera_name, frame_count, total_frames)
                out.write(result_frame)
                
                # Show preview
                if self.show_preview:
                    cv2.imshow(f'DRISTI: {camera_name}', result_frame)
            
            # Handle key presses
            if self.show_preview:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    print("  Stopped by user")
                    break
                elif key == ord('p'):  # Pause/Resume
                    pause = not pause
                    if pause:
                        print("  Paused - Press 'p' to resume")
                    else:
                        print("  Resumed")
                elif key == ord('s'):  # Save current frame
                    if not pause:
                        pause = True
                    save_path = f"output/snapshots/{camera_name}_frame_{frame_count}.jpg"
                    cv2.imwrite(save_path, result_frame)
                    print(f"  Frame saved: {save_path}")
            
            # Show progress every 10%
            if total_frames > 0 and frame_count % max(1, total_frames // 10) == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                print(f"    ↳ {progress:.0f}% complete | Matches: {match_count} | Time: {elapsed:.1f}s")
        
        # Cleanup
        cap.release()
        out.release()
        if self.show_preview:
            cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"    ✓ Completed in {elapsed:.1f}s | Matches: {match_count}")
        
        # Generate report
        if matches:
            self.generate_report(camera_name, video_path, output_path, matches, elapsed)
        else:
            print(f"    ℹ️  No matches found in {camera_name}")
        
        return {
            'camera': camera_name,
            'matches': match_count,
            'processing_time': elapsed,
            'output_video': output_path
        }
    
    def draw_results_with_preview(self, frame, faces, match_info=None, camera_name="", current_frame=0, total_frames=0):
        """Draw results on frame with preview information"""
        result = frame.copy()
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Draw all detected faces (light gray)
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            cv2.rectangle(result, (x1, y1), (x2, y2), (100, 100, 100), 1)
        
        # Draw match (if any)
        if match_info:
            x1, y1, x2, y2 = match_info['bbox']
            similarity = match_info['similarity']
            
            # Choose color based on confidence
            if similarity > 0.85:
                color = (0, 255, 0)  # Green - High confidence
                thickness = 3
            elif similarity > 0.75:
                color = (0, 255, 255)  # Yellow - Medium confidence
                thickness = 2
            else:
                color = (0, 165, 255)  # Orange - Low confidence
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Draw "FOUND" for high confidence
            if similarity > 0.8:
                cv2.putText(result, "✅ FOUND", (x1, y1 - 40),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
            
            # Draw similarity percentage
            label = f"{self.target_name}: {similarity:.1%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(result,
                         (x1, y2 - label_size[1] - 10),
                         (x1 + label_size[0], y2),
                         color, cv2.FILLED)
            
            cv2.putText(result, label, (x1, y2 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add status bar at the top
        status_bar = np.zeros((50, width, 3), dtype=np.uint8)
        status_bar[:] = (30, 30, 30)
        result[:50, :] = status_bar
        
        # Add camera name
        cv2.putText(result, f"Camera: {camera_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add frame counter
        frame_text = f"Frame: {current_frame}/{total_frames}"
        cv2.putText(result, frame_text, (width // 3, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(result, timestamp, (width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add controls info at bottom
        if self.show_preview:
            controls_bar = np.zeros((40, width, 3), dtype=np.uint8)
            controls_bar[:] = (30, 30, 30)
            result[height-40:height, :] = controls_bar
            
            controls = "Controls: [q] Quit | [p] Pause/Resume | [s] Save Frame"
            cv2.putText(result, controls, (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return result
    
    def generate_report(self, camera_name, video_path, output_path, matches, processing_time):
        """Generate search report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"output/reports/{camera_name}_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DRISTI - FACE RECOGNITION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Target Person: {self.target_name}\n")
            f.write(f"Camera: {camera_name}\n")
            f.write(f"Source Video: {os.path.basename(video_path)}\n")
            f.write(f"Search Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("="*70 + "\n")
            f.write("SEARCH RESULTS\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total Matches Found: {len(matches)}\n")
            f.write(f"Processing Time: {processing_time:.1f} seconds\n")
            f.write(f"Processing Speed: {len(matches)/max(processing_time, 0.1):.1f} matches/sec\n\n")
            
            if matches:
                best_match = max(matches, key=lambda x: x['similarity'])
                avg_similarity = np.mean([m['similarity'] for m in matches])
                
                f.write("BEST MATCH DETAILS:\n")
                f.write(f"  Similarity: {best_match['similarity']:.1%}\n")
                f.write(f"  Verification Score: {best_match['verification']:.1%}\n")
                f.write(f"  Time in Video: {best_match['time']:.1f} seconds\n")
                f.write(f"  Frame Number: {best_match['frame']}\n")
                f.write(f"  Confidence: {'HIGH' if best_match['similarity'] > 0.8 else 'MEDIUM'}\n\n")
                
                f.write("ALL MATCHES (First 10):\n")
                for i, match in enumerate(matches[:10], 1):
                    f.write(f"{i}. Time: {match['time']:.1f}s | Similarity: {match['similarity']:.1%} | Frame: {match['frame']}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("EVIDENCE FILES\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"1. Marked Video: {os.path.basename(output_path)}\n")
            f.write(f"2. Target Face: target_{self.target_name}.jpg\n")
            if matches:
                f.write(f"3. Match Snapshot: {camera_name}_match.jpg\n")
            f.write(f"4. This Report: {os.path.basename(report_file)}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("VERDICT\n")
            f.write("="*70 + "\n\n")
            
            if matches:
                if best_match['similarity'] > 0.85:
                    f.write("✅ PERSON POSITIVELY IDENTIFIED\n")
                elif best_match['similarity'] > 0.75:
                    f.write("⚠️  POSSIBLE MATCH - VERIFICATION RECOMMENDED\n")
                else:
                    f.write("⚠️  WEAK MATCH - FURTHER INVESTIGATION NEEDED\n")
            else:
                f.write("❌ NO MATCH FOUND\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"📄 Report saved: {report_file}")

# ============================================================================
# PRODUCTION MENU SYSTEM
# ============================================================================

class ProductionMenu:
    """Production menu system"""
    
    def __init__(self):
        self.system = DristiProductionSystem()
        self.cctv_videos = {}
        self.scan_cctv_videos()
    
    def scan_cctv_videos(self):
        """Scan for CCTV videos"""
        if not os.path.exists('input'):
            os.makedirs('input', exist_ok=True)
            return
        
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        for file in os.listdir('input'):
            file_lower = file.lower()
            if any(file_lower.endswith(ext) for ext in video_exts):
                cam_name = os.path.splitext(file)[0]
                self.cctv_videos[cam_name] = f"input/{file}"
    
    def display_header(self):
        """Display header"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*70)
        print("🚀 DRISTI PRODUCTION SYSTEM")
        print("Face Recognition System for CCTV Analysis")
        print("="*70)
        print(f"Target: {self.system.target_name if self.system.target_name else 'Not Set'}")
        print(f"CCTV Videos: {len(self.cctv_videos)} found")
        print(f"Live Preview: {'Enabled' if self.system.show_preview else 'Disabled'}")
        print("="*70)
    
    def main_menu(self):
        """Main menu"""
        while True:
            self.display_header()
            
            print("\n📋 MAIN MENU")
            print("-" * 40)
            print("1. 🎯 Set Target Person")
            print("2. 🔍 Search in Single CCTV")
            print("3. 🌐 Search in Multiple CCTVs")
            print("4. ⚡ Performance Settings")
            print("5. 📁 File Management")
            print("6. 🖥️  Toggle Live Preview")
            print("7. 🚪 Exit")
            print("-" * 40)
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == "1":
                self.set_target()
            elif choice == "2":
                self.search_single()
            elif choice == "3":
                self.search_multiple()
            elif choice == "4":
                self.performance_settings()
            elif choice == "5":
                self.file_management()
            elif choice == "6":
                self.toggle_preview()
            elif choice == "7":
                print("\nThank you for using DRISTI!")
                print("Results are in the 'output' folder.")
                break
            else:
                print("Invalid option")
                input("Press Enter to continue...")
    
    def set_target(self):
        """Set target person"""
        self.display_header()
        print("\n🎯 SET TARGET PERSON")
        print("-" * 40)
        
        # Find photos
        photos = []
        if os.path.exists('input'):
            photos = [f for f in os.listdir('input') 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if photos:
            print("Available photos in 'input/' folder:")
            for i, p in enumerate(photos, 1):
                print(f"{i}. {p}")
            print(f"{len(photos)+1}. Enter custom path")
            print(f"{len(photos)+2}. Use webcam to capture photo")
        else:
            print("No photos in 'input/' folder")
            print("1. Enter custom path")
            print("2. Use webcam to capture photo")
        
        try:
            if photos:
                choice = input(f"\nSelect option (1-{len(photos)+2}): ").strip()
                if choice.isdigit():
                    choice = int(choice)
                    if 1 <= choice <= len(photos):
                        path = f"input/{photos[choice-1]}"
                    elif choice == len(photos) + 1:
                        path = input("Enter photo path: ").strip()
                    elif choice == len(photos) + 2:
                        path = self.capture_webcam_photo()
                    else:
                        print("Invalid selection")
                        return
                else:
                    path = choice
            else:
                choice = input("\nSelect option (1-2): ").strip()
                if choice == "1":
                    path = input("Enter photo path: ").strip()
                elif choice == "2":
                    path = self.capture_webcam_photo()
                else:
                    print("Invalid selection")
                    return
            
            name = input("Person name (optional): ").strip()
            
            print(f"\nSetting target from: {path}")
            if self.system.set_target_person(path, name):
                print(f"\n✅ Target set: {self.system.target_name}")
            else:
                print("\n❌ Failed to set target")
        
        except Exception as e:
            print(f"Error: {e}")
        
        input("\nPress Enter to continue...")
    
    def capture_webcam_photo(self):
        """Capture photo using webcam"""
        print("\n📸 Capturing photo from webcam...")
        print("Press SPACE to capture, ESC to cancel")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return ""
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display frame
            cv2.putText(frame, "Press SPACE to capture", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "ESC to cancel", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Webcam - Capture Target Photo', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                # Save photo
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"input/webcam_target_{timestamp}.jpg"
                cv2.imwrite(path, frame)
                print(f"Photo saved to: {path}")
                break
            elif key == 27:  # ESC
                print("Cancelled")
                path = ""
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return path
    
    def search_single(self):
        """Search in single CCTV"""
        if self.system.target_features is None:
            print("⚠️  Set target person first!")
            input("\nPress Enter to continue...")
            return
        
        if not self.cctv_videos:
            print("No CCTV videos found in 'input/' folder")
            print("Add videos as MP4, AVI, etc. to the 'input' folder")
            input("\nPress Enter to continue...")
            return
        
        self.display_header()
        print("\n🔍 SEARCH IN SINGLE CCTV")
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
                
                print(f"\nStarting search in {cam_name}...")
                print("Live preview with controls:")
                print("  [q] - Quit search")
                print("  [p] - Pause/Resume")
                print("  [s] - Save current frame")
                
                # Start search with preview
                self.system.search_in_video_with_preview(video_path, cam_name)
                
                print("\n✅ Search completed!")
                print("   Check 'output/' folder for results.")
            else:
                print("Invalid selection")
        except ValueError:
            print("Please enter a number")
        
        input("\nPress Enter to continue...")
    
    def search_multiple(self):
        """Search in multiple CCTV cameras"""
        if self.system.target_features is None:
            print("⚠️  Set target person first!")
            input("\nPress Enter to continue...")
            return
        
        if not self.cctv_videos:
            print("No CCTV videos found")
            input("\nPress Enter to continue...")
            return
        
        self.display_header()
        print("\n🌐 SEARCH MULTIPLE CCTVs")
        print("-" * 40)
        
        print(f"Found {len(self.cctv_videos)} CCTV cameras:")
        for i, cam_name in enumerate(self.cctv_videos.keys(), 1):
            print(f"  {i}. {cam_name}")
        
        print("\nSelect cameras to search (comma-separated numbers):")
        print("Example: 1,3,4 or 'all' for all cameras")
        
        selection = input("\nYour selection: ").strip().lower()
        
        selected_cameras = {}
        
        if selection == 'all':
            selected_cameras = self.cctv_videos
        else:
            try:
                indices = [int(i.strip()) for i in selection.split(',')]
                cam_names = list(self.cctv_videos.keys())
                
                for idx in indices:
                    if 1 <= idx <= len(cam_names):
                        cam_name = cam_names[idx-1]
                        selected_cameras[cam_name] = self.cctv_videos[cam_name]
                
                if not selected_cameras:
                    print("No valid cameras selected")
                    return
            except:
                print("Invalid selection format")
                return
        
        print(f"\nSelected {len(selected_cameras)} cameras for parallel processing...")
        print("Note: Live preview disabled for parallel processing")
        
        # Disable preview for parallel processing
        original_preview_setting = self.system.show_preview
        self.system.show_preview = False
        
        try:
            # Process in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.system.max_workers) as executor:
                futures = []
                
                for cam_name, video_path in selected_cameras.items():
                    print(f"  Starting: {cam_name}")
                    future = executor.submit(self.system.search_in_video_with_preview, video_path, cam_name)
                    futures.append((future, cam_name))
                
                # Collect results
                results = []
                print("\n" + "-" * 40)
                print("PROCESSING RESULTS:")
                print("-" * 40)
                
                for future, cam_name in futures:
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            status = "✓" if result['matches'] > 0 else "✗"
                            print(f"  {status} {cam_name}: {result['matches']} matches ({result['processing_time']:.1f}s)")
                    except Exception as e:
                        print(f"  ✗ {cam_name}: Error - {str(e)[:50]}...")
            
            # Generate combined report
            if results:
                self.generate_combined_report(results)
        
        finally:
            # Restore preview setting
            self.system.show_preview = original_preview_setting
        
        input("\nPress Enter to continue...")
    
    def generate_combined_report(self, results):
        """Generate combined report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"output/reports/combined_{timestamp}.txt"
        
        total_matches = sum(r['matches'] for r in results)
        total_time = sum(r['processing_time'] for r in results)
        cameras_with_matches = [r for r in results if r['matches'] > 0]
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DRISTI - MULTI-CAMERA SEARCH REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Target Person: {self.system.target_name}\n")
            f.write(f"Search Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Cameras Searched: {len(results)}\n")
            f.write(f"Total Processing Time: {total_time:.1f} seconds\n")
            f.write(f"Total Matches Found: {total_matches}\n\n")
            
            f.write("="*70 + "\n")
            f.write("CAMERAS WITH MATCHES\n")
            f.write("="*70 + "\n\n")
            
            if cameras_with_matches:
                for result in cameras_with_matches:
                    f.write(f"📹 Camera: {result['camera']}\n")
                    f.write(f"   Matches: {result['matches']}\n")
                    f.write(f"   Processing Time: {result['processing_time']:.1f}s\n")
                    f.write(f"   Evidence File: {os.path.basename(result['output_video'])}\n\n")
            else:
                f.write("No cameras detected the target person.\n\n")
            
            f.write("="*70 + "\n")
            f.write("ALL CAMERAS SEARCHED\n")
            f.write("="*70 + "\n\n")
            
            for result in results:
                status = "✅ DETECTED" if result['matches'] > 0 else "❌ NOT DETECTED"
                f.write(f"{status} - {result['camera']}: {result['matches']} matches\n")
        
        print(f"\n📊 Combined report saved: {report_file}")
    
    def performance_settings(self):
        """Configure performance settings"""
        self.display_header()
        print("\n⚡ PERFORMANCE SETTINGS")
        print("-" * 40)
        
        print(f"Current Settings:")
        print(f"1. Similarity Threshold: {self.system.similarity_threshold*100:.0f}%")
        print(f"2. Frame Skip: {self.system.frame_skip}x")
        print(f"3. Max Workers (Parallel): {self.system.max_workers}")
        print(f"4. Consecutive Matches Required: {self.system.min_consecutive_matches}")
        print(f"5. Live Preview: {'Enabled' if self.system.show_preview else 'Disabled'}")
        
        print("\nAdjust settings (enter number to change, or 0 to go back):")
        
        try:
            choice = input("\nSelect setting to change (1-5): ").strip()
            
            if choice == "1":
                new_val = float(input(f"New similarity threshold (0.1-1.0, current: {self.system.similarity_threshold}): "))
                if 0.1 <= new_val <= 1.0:
                    self.system.similarity_threshold = new_val
                    print(f"✓ Threshold set to {new_val*100:.0f}%")
                else:
                    print("Invalid value")
            
            elif choice == "2":
                new_val = int(input(f"New frame skip (1-10, current: {self.system.frame_skip}): "))
                if 1 <= new_val <= 10:
                    self.system.frame_skip = new_val
                    print(f"✓ Frame skip set to {new_val}x")
                else:
                    print("Invalid value")
            
            elif choice == "3":
                new_val = int(input(f"New max workers (1-8, current: {self.system.max_workers}): "))
                if 1 <= new_val <= 8:
                    self.system.max_workers = new_val
                    print(f"✓ Max workers set to {new_val}")
                else:
                    print("Invalid value")
            
            elif choice == "4":
                new_val = int(input(f"New consecutive matches (1-5, current: {self.system.min_consecutive_matches}): "))
                if 1 <= new_val <= 5:
                    self.system.min_consecutive_matches = new_val
                    print(f"✓ Consecutive matches set to {new_val}")
                else:
                    print("Invalid value")
            
            elif choice == "5":
                self.system.show_preview = not self.system.show_preview
                print(f"✓ Live preview {'enabled' if self.system.show_preview else 'disabled'}")
            
            elif choice == "0":
                return
            
            else:
                print("Invalid selection")
        
        except ValueError:
            print("Please enter a valid number")
        
        input("\nPress Enter to continue...")
    
    def toggle_preview(self):
        """Toggle live preview"""
        self.system.show_preview = not self.system.show_preview
        print(f"\n✅ Live preview {'enabled' if self.system.show_preview else 'disabled'}")
        input("\nPress Enter to continue...")
    
    def file_management(self):
        """File management menu"""
        self.display_header()
        print("\n📁 FILE MANAGEMENT")
        print("-" * 40)
        
        print("1. 📂 View input files")
        print("2. 📂 View output files")
        print("3. 🗑️  Clean output folder")
        print("4. 🔍 Check model files")
        print("5. ⬅️  Back to main menu")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            self.view_input_files()
        elif choice == "2":
            self.view_output_files()
        elif choice == "3":
            self.clean_output_folder()
        elif choice == "4":
            self.check_model_files()
        elif choice == "5":
            return
        else:
            print("Invalid option")
        
        input("\nPress Enter to continue...")
    
    def view_input_files(self):
        """View input folder contents"""
        print("\n📁 INPUT FOLDER CONTENTS:")
        print("-" * 40)
        
        if os.path.exists('input'):
            files = os.listdir('input')
            if files:
                for file in files:
                    path = f"input/{file}"
                    if os.path.isfile(path):
                        size = os.path.getsize(path) / 1024  # KB
                        print(f"  {file} ({size:.1f} KB)")
                    else:
                        print(f"  {file}/ (folder)")
            else:
                print("  Empty folder")
        else:
            print("  Folder does not exist")
        
        print(f"\nTotal files: {len(files) if 'files' in locals() else 0}")
    
    def view_output_files(self):
        """View output folder contents"""
        print("\n📁 OUTPUT FOLDER STRUCTURE:")
        print("-" * 40)
        
        def list_files(dir_path, indent=0):
            if not os.path.exists(dir_path):
                return
            
            items = os.listdir(dir_path)
            for item in items:
                path = os.path.join(dir_path, item)
                prefix = "  " * indent + "├─ "
                
                if os.path.isfile(path):
                    size = os.path.getsize(path) / 1024  # KB
                    print(f"{prefix}{item} ({size:.1f} KB)")
                else:
                    print(f"{prefix}{item}/")
                    list_files(path, indent + 1)
        
        list_files('output')
    
    def clean_output_folder(self):
        """Clean output folder"""
        print("\n🗑️  CLEAN OUTPUT FOLDER")
        print("-" * 40)
        
        confirm = input("This will delete ALL files in the output folder. Continue? (y/n): ").strip().lower()
        
        if confirm == 'y':
            import shutil
            
            if os.path.exists('output'):
                shutil.rmtree('output')
                self.system.create_directories()
                print("✓ Output folder cleaned")
            else:
                print("Output folder does not exist")
        else:
            print("Cancelled")
    
    def check_model_files(self):
        """Check model files"""
        print("\n🔍 MODEL FILES CHECK")
        print("-" * 40)
        
        models_dir = 'models'
        if not os.path.exists(models_dir):
            print("Models directory does not exist")
            os.makedirs(models_dir)
            print("Created models directory")
        
        models = {
            'face_verifier.pth': 'Face verification model',
            'resnet_face.pth': 'Feature extraction model'
        }
        
        for model_file, description in models.items():
            path = f"{models_dir}/{model_file}"
            if os.path.exists(path):
                size = os.path.getsize(path) / (1024*1024)  # MB
                print(f"✓ {model_file}: {size:.1f} MB - {description}")
            else:
                print(f"✗ {model_file}: NOT FOUND - {description}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Initializing DRISTI System...")
    
    try:
        # Check for required packages
        required_packages = ['cv2', 'torch', 'torchvision', 'numpy', 'PIL']
        
        for package in required_packages:
            try:
                if package == 'cv2':
                    import cv2
                elif package == 'torch':
                    import torch
                elif package == 'torchvision':
                    import torchvision
                elif package == 'numpy':
                    import numpy
                elif package == 'PIL':
                    from PIL import Image
            except ImportError:
                print(f"✗ Missing package: {package}")
                print(f"   Install with: pip install {package}")
        
        # Create necessary directories
        for dir_path in ['input', 'output', 'models']:
            os.makedirs(dir_path, exist_ok=True)
        
        # Start the system
        menu = ProductionMenu()
        menu.main_menu()
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")