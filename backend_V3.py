import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
import mediapipe as mp
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')

class FastAccurateFinder:
    """Fast and accurate face finder with multi-CCTV support"""
    
    def __init__(self):
        print("="*70)
        print("🚀 DRISTI - FAST ACCURATE FINDER v3.0")
        print("="*70)
        print("Features:")
        print("• Multi-CCTV parallel processing")
        print("• 10x faster with frame skipping")
        print("• Smart caching for instant results")
        print("• Batch processing without live preview")
        print("="*70)
        
        # Initialize MediaPipe ONCE (reuse for all frames)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )
        
        # Face mesh for better features
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # Target info
        self.target_features = None
        self.target_name = ""
        self.target_face_img = None
        
        # Smart thresholds (tuned for accuracy)
        self.detection_thresh = 0.7
        self.similarity_thresh = 0.78  # 78% match required
        self.min_consecutive = 2  # Reduced for speed
        
        # Performance settings
        self.frame_skip = 3  # Process every 3rd frame (3x faster)
        self.resize_factor = 0.5  # Process at half resolution (4x faster)
        self.enable_preview = False  # No live preview for speed
        
        # Tracking and caching
        self.match_history = {}
        self.feature_cache = {}  # Cache face features for speed
        
        # Multi-processing
        self.max_workers = 4  # Number of parallel processes
        
        # Create directories
        self.create_directories()
        
        print("✓ System Ready")
        print(f"  Speed Mode: {self.frame_skip}x frame skip")
        print(f"  Accuracy: {self.similarity_thresh*100}% similarity required")
        print(f"  Parallel Processing: {self.max_workers} workers")
        print("="*70)
    
    def create_directories(self):
        """Create organized output directories"""
        dirs = [
            'output',
            'output/evidence',
            'output/reports', 
            'output/snapshots',
            'output/alerts',
            'output/targets',
            'input'
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def extract_face_features_fast(self, face_image):
        """Fast feature extraction with caching"""
        # Create cache key
        cache_key = hash(face_image.tobytes())
        
        # Check cache
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        try:
            # Fast resize for feature extraction
            small_face = cv2.resize(face_image, (100, 100))
            
            # Convert to RGB for MediaPipe
            rgb_face = cv2.cvtColor(small_face, cv2.COLOR_BGR2RGB)
            
            # Get face mesh landmarks (fast)
            results = self.face_mesh.process(rgb_face)
            
            features = []
            
            if results.multi_face_landmarks:
                # Extract landmarks (468 points)
                landmarks = []
                for landmark in results.multi_face_landmarks[0].landmark:
                    landmarks.extend([landmark.x, landmark.y])
                
                features = np.array(landmarks)
            else:
                # Fallback: simple grayscale features
                gray = cv2.cvtColor(small_face, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                features = gray.flatten() / 255.0
            
            # Add color histogram (fast)
            hsv = cv2.cvtColor(small_face, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Combine features
            combined = np.concatenate([features, hist])
            combined = combined / np.linalg.norm(combined)
            
            # Cache result
            self.feature_cache[cache_key] = combined
            return combined
            
        except:
            # Return simple features on error
            return np.zeros(100, dtype='float32')
    
    def set_target_person(self, photo_path, name=""):
        """Set target person with enhanced processing"""
        print(f"\n🎯 SETTING TARGET PERSON")
        print("-" * 40)
        
        if not os.path.exists(photo_path):
            print(f"✗ Photo not found: {photo_path}")
            return False
        
        # Read and enhance image
        img = cv2.imread(photo_path)
        if img is None:
            print("✗ Cannot read image")
            return False
        
        # Enhance image for better detection
        img = self.enhance_image(img)
        
        # Detect face
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)
        
        if not results.detections:
            print("✗ No face found in photo")
            print("  Tips: Use clear front-facing photo with good lighting")
            return False
        
        # Get best face
        best_det = max(results.detections, key=lambda x: x.score[0])
        bboxC = best_det.location_data.relative_bounding_box
        h, w = img.shape[:2]
        
        x1 = int(bboxC.xmin * w)
        y1 = int(bboxC.ymin * h)
        x2 = int((bboxC.xmin + bboxC.width) * w)
        y2 = int((bboxC.ymin + bboxC.height) * h)
        
        # Add padding
        pad = int(min(x2-x1, y2-y1) * 0.2)
        x1, y1 = max(0, x1-pad), max(0, y1-pad)
        x2, y2 = min(w, x2+pad), min(h, y2+pad)
        
        face_img = img[y1:y2, x1:x2]
        
        if face_img.size == 0:
            print("✗ Could not extract face")
            return False
        
        # Extract features
        self.target_features = self.extract_face_features_fast(face_img)
        self.target_name = name or os.path.basename(photo_path).split('.')[0]
        self.target_face_img = face_img
        
        # Save enhanced target face
        target_path = f'output/targets/{self.target_name}.jpg'
        cv2.imwrite(target_path, face_img)
        
        print(f"✅ TARGET SET: {self.target_name}")
        print(f"  Face Quality: {best_det.score[0]:.1%}")
        print(f"  Face Size: {face_img.shape}")
        print(f"  Features: {len(self.target_features)} dimensions")
        print(f"  Saved: {target_path}")
        
        return True
    
    def enhance_image(self, image):
        """Fast image enhancement"""
        # Simple enhancement - faster than CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Simple contrast stretching
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        l = clahe.apply(l)
        
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detect_faces_fast(self, frame):
        """Fast face detection with optimizations"""
        # Resize frame for faster processing
        if self.resize_factor < 1.0:
            h, w = frame.shape[:2]
            small_frame = cv2.resize(frame, 
                                   (int(w * self.resize_factor), 
                                    int(h * self.resize_factor)))
        else:
            small_frame = frame
        
        # Convert to RGB
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb)
        
        faces = []
        
        if results.detections:
            scale = 1.0 / self.resize_factor
            
            for det in results.detections:
                conf = det.score[0]
                if conf < self.detection_thresh:
                    continue
                
                bboxC = det.location_data.relative_bounding_box
                h, w = small_frame.shape[:2]
                
                # Scale back to original size
                x1 = int(bboxC.xmin * w * scale)
                y1 = int(bboxC.ymin * h * scale)
                bw = int(bboxC.width * w * scale)
                bh = int(bboxC.height * h * scale)
                
                # Filter very small faces
                if bw * bh < 2000:
                    continue
                
                # Add padding
                pad = int(min(bw, bh) * 0.15)
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(frame.shape[1], x1 + bw + 2*pad)
                y2 = min(frame.shape[0], y1 + bh + 2*pad)
                
                # Extract face
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size > 0 and face_img.shape[0] > 40 and face_img.shape[1] > 40:
                    faces.append({
                        'bbox': (x1, y1, x2, y2),
                        'face': face_img,
                        'confidence': float(conf)
                    })
        
        return faces
    
    def calculate_similarity_fast(self, features1, features2):
        """Fast similarity calculation"""
        min_len = min(len(features1), len(features2))
        if min_len == 0:
            return 0.0
        
        f1 = features1[:min_len]
        f2 = features2[:min_len]
        
        # Fast cosine similarity
        dot = np.dot(f1, f2)
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot / (norm1 * norm2)
            # Square similarity to make it more discriminative
            similarity = similarity ** 2
            return float(similarity)
        
        return 0.0
    
    def is_real_match_fast(self, face_img, frame_num, location_key):
        """Fast match verification"""
        if self.target_features is None:
            return False, 0.0
        
        # Extract features
        features = self.extract_face_features_fast(face_img)
        
        # Calculate similarity
        similarity = self.calculate_similarity_fast(self.target_features, features)
        
        if similarity < self.similarity_thresh:
            return False, similarity
        
        # Fast tracking - only check last 2 matches
        if location_key not in self.match_history:
            self.match_history[location_key] = []
        
        self.match_history[location_key].append({
            'frame': frame_num,
            'similarity': similarity,
            'time': time.time()
        })
        
        # Keep only recent matches (last 3 seconds)
        current_time = time.time()
        self.match_history[location_key] = [
            m for m in self.match_history[location_key]
            if current_time - m['time'] < 3
        ]
        
        # Need at least 2 consecutive matches
        if len(self.match_history[location_key]) >= self.min_consecutive:
            # Weighted average (recent matches weighted more)
            weights = np.exp(np.arange(len(self.match_history[location_key])))
            weights = weights / weights.sum()
            
            similarities = np.array([m['similarity'] for m in self.match_history[location_key]])
            avg_sim = np.dot(similarities, weights)
            
            return True, avg_sim
        
        return False, similarity
    
    def process_single_video(self, video_path, camera_name):
        """Process a single video file (no preview)"""
        print(f"  📹 Processing: {camera_name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"    ✗ Cannot open video")
            return None
        
        # Get video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare output video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/evidence/{camera_name}_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Reset match history for this video
        self.match_history.clear()
        
        # Tracking
        matches = []
        frame_count = 0
        match_count = 0
        start_time = time.time()
        
        # Progress tracking
        last_progress_update = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for speed
            if frame_count % self.frame_skip != 0:
                continue
            
            # Detect faces
            faces = self.detect_faces_fast(frame)
            
            # Check each face
            current_match = None
            
            for face in faces:
                # Create location key
                x1, y1, x2, y2 = face['bbox']
                loc_key = f"{x1//100}_{y1//100}"  # Coarser grid for speed
                
                # Check if match
                is_match, similarity = self.is_real_match_fast(
                    face['face'], frame_count, loc_key
                )
                
                if is_match:
                    match_count += 1
                    current_match = {
                        'bbox': face['bbox'],
                        'similarity': similarity,
                        'frame': frame_count,
                        'time': frame_count / fps,
                        'confidence': face['confidence']
                    }
                    matches.append(current_match)
                    
                    # Save first match snapshot
                    if match_count == 1:
                        snap_path = f"output/snapshots/{camera_name}_first_match.jpg"
                        cv2.imwrite(snap_path, face['face'])
            
            # Draw results
            result_frame = self.draw_results_fast(frame, faces, current_match)
            out.write(result_frame)
            
            # Show progress every 10%
            progress = (frame_count / total_frames) * 100
            if progress - last_progress_update >= 10:
                elapsed = time.time() - start_time
                print(f"    ↳ Progress: {progress:.0f}% | Matches: {match_count} | Time: {elapsed:.1f}s")
                last_progress_update = progress
        
        # Cleanup
        cap.release()
        out.release()
        
        elapsed = time.time() - start_time
        print(f"    ✓ Completed in {elapsed:.1f}s | Matches: {match_count}")
        
        return {
            'camera': camera_name,
            'video': os.path.basename(video_path),
            'output_video': output_path,
            'total_frames': frame_count,
            'processed_frames': frame_count // self.frame_skip,
            'matches_found': match_count,
            'match_details': matches,
            'processing_time': elapsed
        }
    
    def search_single_cctv(self, video_path, camera_name):
        """Search in single CCTV (public method)"""
        print(f"\n🔍 SINGLE CCTV SEARCH")
        print("-" * 40)
        print(f"Camera: {camera_name}")
        print(f"Video: {os.path.basename(video_path)}")
        print(f"Settings: {self.frame_skip}x speed, No preview")
        print("-" * 40)
        
        result = self.process_single_video(video_path, camera_name)
        
        if result:
            self.generate_single_report(result)
        
        return result
    
    def search_all_cctvs(self, cctv_list):
        """Search in ALL CCTV cameras in parallel"""
        print(f"\n🔍 MULTI-CCTV PARALLEL SEARCH")
        print("-" * 40)
        print(f"Total Cameras: {len(cctv_list)}")
        print(f"Parallel Workers: {self.max_workers}")
        print(f"Speed Mode: {self.frame_skip}x frame skip")
        print("-" * 40)
        
        all_results = []
        start_time = time.time()
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_camera = {
                executor.submit(self.process_single_video, path, name): name
                for name, path in cctv_list.items()
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_camera):
                camera_name = future_to_camera[future]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                        print(f"  ✓ {camera_name}: {result['matches_found']} matches")
                except Exception as e:
                    print(f"  ✗ {camera_name}: Error - {e}")
        
        # Generate combined report
        if all_results:
            self.generate_combined_report(all_results, start_time)
        
        return all_results
    
    def draw_results_fast(self, frame, faces, match_info=None):
        """Fast drawing of results"""
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Draw all detected faces (faint)
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            cv2.rectangle(result, (x1, y1), (x2, y2), (60, 60, 60), 1)
        
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
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw similarity percentage
            label = f"{similarity:.1%}"
            cv2.putText(result, label, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add watermark (fast)
        cv2.putText(result, "DRISTI AI", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return result
    
    def generate_single_report(self, result):
        """Generate report for single CCTV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"output/reports/{result['camera']}_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Also create text summary
        text_file = f"output/reports/{result['camera']}_summary_{timestamp}.txt"
        with open(text_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"DRISTI - SEARCH REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Target Person: {self.target_name}\n")
            f.write(f"Camera: {result['camera']}\n")
            f.write(f"Video: {result['video']}\n")
            f.write(f"Search Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("="*70 + "\n")
            f.write("RESULTS\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Matches Found: {result['matches_found']}\n")
            f.write(f"Total Frames: {result['total_frames']}\n")
            f.write(f"Processing Time: {result['processing_time']:.1f} seconds\n")
            f.write(f"Processing Speed: {result['processed_frames']/result['processing_time']:.1f} FPS\n\n")
            
            if result['matches_found'] > 0:
                best_match = max(result['match_details'], key=lambda x: x['similarity'])
                f.write("BEST MATCH:\n")
                f.write(f"  Time: {best_match['time']:.1f} seconds\n")
                f.write(f"  Frame: {best_match['frame']}\n")
                f.write(f"  Similarity: {best_match['similarity']:.1%}\n")
                f.write(f"  Confidence: {'HIGH' if best_match['similarity'] > 0.8 else 'MEDIUM'}\n\n")
                
                f.write("EVIDENCE FILES:\n")
                f.write(f"  1. Marked Video: {os.path.basename(result['output_video'])}\n")
                f.write(f"  2. First Match Snapshot: {result['camera']}_first_match.jpg\n")
                f.write(f"  3. Target Face: targets/{self.target_name}.jpg\n")
            else:
                f.write("NO MATCHES FOUND\n")
                f.write("\nSuggestions:\n")
                f.write("  1. Check if target appears in video\n")
                f.write("  2. Try lower similarity threshold\n")
                f.write("  3. Use clearer target photo\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        print(f"📄 Report saved: {text_file}")
        print(f"🎥 Evidence video: {result['output_video']}")
    
    def generate_combined_report(self, all_results, start_time):
        """Generate combined report for all CCTVs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"output/reports/combined_report_{timestamp}.txt"
        
        total_matches = sum(r['matches_found'] for r in all_results)
        total_time = time.time() - start_time
        cameras_with_matches = [r for r in all_results if r['matches_found'] > 0]
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DRISTI - MULTI-CCTV SEARCH REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Target Person: {self.target_name}\n")
            f.write(f"Search Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Cameras: {len(all_results)}\n")
            f.write(f"Total Processing Time: {total_time:.1f} seconds\n")
            f.write(f"Average Speed: {sum(r['processed_frames'] for r in all_results)/total_time:.1f} FPS\n\n")
            
            f.write("="*80 + "\n")
            f.write("SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Matches Found: {total_matches}\n")
            f.write(f"Cameras with Matches: {len(cameras_with_matches)}/{len(all_results)}\n\n")
            
            if cameras_with_matches:
                f.write("📍 LOCATIONS WHERE PERSON WAS FOUND:\n")
                f.write("-" * 50 + "\n\n")
                
                for result in sorted(cameras_with_matches, 
                                   key=lambda x: max(m['similarity'] for m in x['match_details']), 
                                   reverse=True):
                    best = max(result['match_details'], key=lambda x: x['similarity'])
                    
                    f.write(f"📹 {result['camera']}\n")
                    f.write(f"   Video: {result['video']}\n")
                    f.write(f"   Matches: {result['matches_found']}\n")
                    f.write(f"   Best Match: {best['similarity']:.1%}\n")
                    f.write(f"   Time: {best['time']:.1f} seconds\n")
                    f.write(f"   Evidence: {os.path.basename(result['output_video'])}\n\n")
                
                f.write("🚨 RECOMMENDED ACTION:\n")
                f.write("1. Dispatch team to cameras with highest matches first\n")
                f.write("2. Review evidence videos in order of confidence\n")
                f.write("3. Use timestamps to track movement between cameras\n")
            else:
                f.write("⚠️  PERSON NOT FOUND IN ANY CAMERA\n\n")
                f.write("POSSIBLE REASONS:\n")
                f.write("1. Target not in any provided videos\n")
                f.write("2. Poor lighting/visibility in videos\n")
                f.write("3. Target face not clear enough\n")
                f.write("4. Similarity threshold too high (currently {self.similarity_thresh:.0%})\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for result in all_results:
                f.write(f"{result['camera']}:\n")
                f.write(f"  Status: {'✅ MATCH' if result['matches_found'] > 0 else '❌ NO MATCH'}\n")
                f.write(f"  Matches: {result['matches_found']}\n")
                f.write(f"  Processing: {result['processing_time']:.1f}s\n")
                f.write(f"  Video: {result['output_video']}\n\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\n📊 COMBINED REPORT")
        print("-" * 40)
        print(f"Total Cameras: {len(all_results)}")
        print(f"Cameras with matches: {len(cameras_with_matches)}")
        print(f"Total matches: {total_matches}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Report saved: {report_file}")

class FastMenu:
    """Fast menu system with multi-CCTV options"""
    
    def __init__(self):
        self.finder = FastAccurateFinder()
        self.cctv_videos = self.scan_cctv_videos()
    
    def scan_cctv_videos(self):
        """Scan for CCTV videos in input folder"""
        videos = {}
        
        # Look for video files
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        
        for file in os.listdir('input'):
            if any(file.lower().endswith(ext) for ext in video_exts):
                cam_name = os.path.splitext(file)[0]
                videos[cam_name] = f"input/{file}"
        
        return videos
    
    def display_header(self):
        """Display header"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*70)
        print("🚀 DRISTI - FAST ACCURATE FACE FINDER")
        print("="*70)
        print("Find missing persons across multiple CCTV cameras")
        print(f"Found {len(self.cctv_videos)} CCTV videos")
        print("="*70)
    
    def main_menu(self):
        """Main menu"""
        while True:
            self.display_header()
            
            print("\n📋 MAIN MENU")
            print("-" * 40)
            print("1. 🎯 Set Target Person")
            print("2. 🔍 Search in Single CCTV")
            print("3. 🌐 Search in ALL CCTVs (Parallel)")
            print("4. ⚡ Performance Settings")
            print("5. 📁 Manage Files")
            print("6. 📊 View Results")
            print("7. 🚪 Exit")
            print("-" * 40)
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == "1":
                self.set_target()
            elif choice == "2":
                self.search_single()
            elif choice == "3":
                self.search_all()
            elif choice == "4":
                self.performance_settings()
            elif choice == "5":
                self.manage_files()
            elif choice == "6":
                self.view_results()
            elif choice == "7":
                self.exit_system()
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
        photos = [f for f in os.listdir('input') 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if photos:
            print("Available photos:")
            for i, p in enumerate(photos, 1):
                size = os.path.getsize(f'input/{p}') / 1024
                print(f"{i}. {p} ({size:.0f} KB)")
            print(f"{len(photos)+1}. Enter custom path")
        else:
            print("No photos found in 'input/' folder")
            photos = []
        
        try:
            if photos:
                choice = input(f"\nSelect photo (1-{len(photos)+1}): ").strip()
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
            if self.finder.set_target_person(path, name):
                print(f"\n✅ Target set: {self.finder.target_name}")
            else:
                print("\n❌ Failed to set target")
        
        except Exception as e:
            print(f"Error: {e}")
        
        input("\nPress Enter to continue...")
    
    def search_single(self):
        """Search in single CCTV"""
        if self.finder.target_features is None:
            print("⚠️  Set target person first!")
            input("\nPress Enter to continue...")
            return
        
        self.display_header()
        print("\n🔍 SEARCH IN SINGLE CCTV")
        print("-" * 40)
        
        if not self.cctv_videos:
            print("No CCTV videos found in 'input/' folder")
            print("Add videos as: cctv1.mp4, mall_camera.mp4, etc.")
            input("\nPress Enter to continue...")
            return
        
        print("Available CCTV cameras:")
        for i, (cam_name, path) in enumerate(self.cctv_videos.items(), 1):
            size = os.path.getsize(path) / (1024*1024)
            print(f"{i}. {cam_name} ({size:.1f} MB)")
        
        try:
            choice = int(input(f"\nSelect camera (1-{len(self.cctv_videos)}): "))
            if 1 <= choice <= len(self.cctv_videos):
                cam_name = list(self.cctv_videos.keys())[choice-1]
                video_path = self.cctv_videos[cam_name]
                
                print(f"\nStarting search in {cam_name}...")
                print("This will run in background without preview.")
                print("Check 'output/' folder for results.")
                
                # Start search
                self.finder.search_single_cctv(video_path, cam_name)
            else:
                print("Invalid selection")
        except ValueError:
            print("Please enter a number")
        
        input("\nPress Enter to continue...")
    
    def search_all(self):
        """Search in all CCTV cameras"""
        if self.finder.target_features is None:
            print("⚠️  Set target person first!")
            input("\nPress Enter to continue...")
            return
        
        if not self.cctv_videos:
            print("No CCTV videos found")
            input("\nPress Enter to continue...")
            return
        
        self.display_header()
        print("\n🌐 SEARCH IN ALL CCTV CAMERAS")
        print("-" * 40)
        
        print(f"Found {len(self.cctv_videos)} CCTV cameras:")
        for cam_name in self.cctv_videos:
            print(f"  • {cam_name}")
        
        print(f"\nSettings:")
        print(f"  • Frame skip: {self.finder.frame_skip}x")
        print(f"  • Parallel workers: {self.finder.max_workers}")
        print(f"  • Similarity threshold: {self.finder.similarity_thresh:.0%}")
        
        confirm = input("\nStart parallel search? (y/n): ").lower()
        
        if confirm == 'y':
            print(f"\n🚀 Starting parallel search across {len(self.cctv_videos)} cameras...")
            print("This may take a few minutes. Results will be saved in 'output/' folder.")
            
            # Start parallel search
            results = self.finder.search_all_cctvs(self.cctv_videos)
            
            if results:
                print(f"\n✅ Search completed!")
                print(f"   Check 'output/reports/combined_report_*.txt' for details")
            else:
                print("\nℹ️  No matches found in any camera")
        else:
            print("Search cancelled")
        
        input("\nPress Enter to continue...")
    
    def performance_settings(self):
        """Adjust performance settings"""
        self.display_header()
        print("\n⚡ PERFORMANCE SETTINGS")
        print("-" * 40)
        
        print("Current settings:")
        print(f"1. Frame skip: {self.finder.frame_skip}x (higher = faster)")
        print(f"2. Resize factor: {self.finder.resize_factor}x (smaller = faster)")
        print(f"3. Similarity threshold: {self.finder.similarity_thresh:.0%} (higher = more accurate)")
        print(f"4. Parallel workers: {self.finder.max_workers}")
        print("5. Reset to defaults")
        print("6. Back to main menu")
        
        choice = input("\nSelect setting to adjust (1-6): ").strip()
        
        if choice == "1":
            new_val = input(f"Frame skip (1-10) [current: {self.finder.frame_skip}]: ")
            if new_val and new_val.isdigit():
                self.finder.frame_skip = int(new_val)
                print(f"✅ Frame skip set to {self.finder.frame_skip}x")
        elif choice == "2":
            new_val = input(f"Resize factor (0.1-1.0) [current: {self.finder.resize_factor}]: ")
            if new_val:
                self.finder.resize_factor = float(new_val)
                print(f"✅ Resize factor set to {self.finder.resize_factor}x")
        elif choice == "3":
            new_val = input(f"Similarity threshold (0.5-0.95) [current: {self.finder.similarity_thresh}]: ")
            if new_val:
                self.finder.similarity_thresh = float(new_val)
                print(f"✅ Similarity threshold set to {self.finder.similarity_thresh:.0%}")
        elif choice == "4":
            new_val = input(f"Parallel workers (1-8) [current: {self.finder.max_workers}]: ")
            if new_val and new_val.isdigit():
                self.finder.max_workers = int(new_val)
                print(f"✅ Parallel workers set to {self.finder.max_workers}")
        elif choice == "5":
            self.finder.frame_skip = 3
            self.finder.resize_factor = 0.5
            self.finder.similarity_thresh = 0.78
            self.finder.max_workers = 4
            print("✅ Settings reset to defaults")
        elif choice == "6":
            return
        else:
            print("Invalid option")
        
        input("\nPress Enter to continue...")
    
    def manage_files(self):
        """Manage input files"""
        self.display_header()
        print("\n📁 MANAGE FILES")
        print("-" * 40)
        
        print("1. View input folder contents")
        print("2. Add new CCTV video")
        print("3. Add new target photo")
        print("4. Clear output folder")
        print("5. Back to main menu")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            self.view_folder_contents()
        elif choice == "2":
            self.add_cctv_video()
        elif choice == "3":
            self.add_target_photo()
        elif choice == "4":
            self.clear_output()
        elif choice == "5":
            return
        else:
            print("Invalid option")
        
        input("\nPress Enter to continue...")
    
    def view_folder_contents(self):
        """View folder contents"""
        print("\n📂 INPUT FOLDER CONTENTS")
        print("-" * 40)
        
        if not os.path.exists('input'):
            print("'input/' folder doesn't exist")
            return
        
        files = os.listdir('input')
        
        if not files:
            print("No files in 'input/' folder")
            return
        
        photos = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        videos = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if photos:
            print("\n📸 Photos:")
            for photo in photos:
                size = os.path.getsize(f'input/{photo}') / 1024
                print(f"  • {photo} ({size:.0f} KB)")
        
        if videos:
            print("\n🎥 Videos:")
            for video in videos:
                size = os.path.getsize(f'input/{video}') / (1024*1024)
                print(f"  • {video} ({size:.1f} MB)")
        
        if not photos and not videos:
            print("No photos or videos found")
    
    def add_cctv_video(self):
        """Add new CCTV video"""
        print("\n➕ ADD NEW CCTV VIDEO")
        print("-" * 40)
        
        source = input("Enter video path (or drag & drop file): ").strip('"')
        
        if not os.path.exists(source):
            print("File not found!")
            return
        
        cam_name = input("Camera name (e.g., 'Mall_Entrance'): ").strip()
        if not cam_name:
            cam_name = os.path.splitext(os.path.basename(source))[0]
        
        # Copy to input folder
        import shutil
        dest = f"input/{cam_name}.mp4"
        
        try:
            shutil.copy2(source, dest)
            print(f"✅ Video added: {dest}")
            
            # Update CCTV list
            self.cctv_videos = self.scan_cctv_videos()
        except Exception as e:
            print(f"Error: {e}")
    
    def add_target_photo(self):
        """Add new target photo"""
        print("\n➕ ADD NEW TARGET PHOTO")
        print("-" * 40)
        
        source = input("Enter photo path (or drag & drop file): ").strip('"')
        
        if not os.path.exists(source):
            print("File not found!")
            return
        
        # Copy to input folder
        import shutil
        dest = f"input/{os.path.basename(source)}"
        
        try:
            shutil.copy2(source, dest)
            print(f"✅ Photo added: {dest}")
        except Exception as e:
            print(f"Error: {e}")
    
    def clear_output(self):
        """Clear output folder"""
        print("\n🗑️  CLEAR OUTPUT FOLDER")
        print("-" * 40)
        
        confirm = input("Are you sure? This will delete all results. (y/n): ").lower()
        
        if confirm == 'y':
            import shutil
            if os.path.exists('output'):
                shutil.rmtree('output')
                os.makedirs('output')
                os.makedirs('output/evidence')
                os.makedirs('output/reports')
                os.makedirs('output/snapshots')
                os.makedirs('output/alerts')
                os.makedirs('output/targets')
                print("✅ Output folder cleared")
            else:
                print("Output folder doesn't exist")
        else:
            print("Cancelled")
    
    def view_results(self):
        """View search results"""
        self.display_header()
        print("\n📊 VIEW RESULTS")
        print("-" * 40)
        
        if not os.path.exists('output/reports'):
            print("No results found")
            input("\nPress Enter to continue...")
            return
        
        reports = [f for f in os.listdir('output/reports') 
                  if f.endswith('.txt') and not f.startswith('combined')]
        
        if not reports:
            print("No individual reports found")
            
            # Check for combined reports
            combined = [f for f in os.listdir('output/reports') 
                       if f.startswith('combined') and f.endswith('.txt')]
            
            if combined:
                print("\nCombined reports available:")
                for report in sorted(combined, reverse=True)[:3]:
                    print(f"  • {report}")
                
                view = input("\nView latest combined report? (y/n): ").lower()
                if view == 'y':
                    latest = sorted(combined, reverse=True)[0]
                    self.show_report(f'output/reports/{latest}')
            
            input("\nPress Enter to continue...")
            return
        
        print("Recent reports:")
        for i, report in enumerate(sorted(reports, reverse=True)[:10], 1):
            size = os.path.getsize(f'output/reports/{report}') / 1024
            print(f"{i}. {report} ({size:.0f} KB)")
        
        print(f"\n{len(reports)+1}. Open reports folder")
        print(f"{len(reports)+2}. View evidence videos")
        print(f"{len(reports)+3}. Back to main menu")
        
        try:
            choice = input(f"\nSelect report (1-{len(reports)+3}): ").strip()
            
            if choice.isdigit():
                choice = int(choice)
                
                if 1 <= choice <= len(reports):
                    report_file = f"output/reports/{sorted(reports, reverse=True)[choice-1]}"
                    self.show_report(report_file)
                elif choice == len(reports) + 1:
                    # Open folder
                    if os.name == 'nt':
                        os.startfile('output/reports')
                    else:
                        os.system('open "output/reports"')
                elif choice == len(reports) + 2:
                    self.view_evidence_videos()
                elif choice == len(reports) + 3:
                    return
                else:
                    print("Invalid selection")
            else:
                print("Please enter a number")
        
        except Exception as e:
            print(f"Error: {e}")
        
        input("\nPress Enter to continue...")
    
    def show_report(self, report_file):
        """Show report content"""
        try:
            with open(report_file, 'r') as f:
                content = f.read()
            
            # Clear screen and show report
            os.system('cls' if os.name == 'nt' else 'clear')
            print(content)
        except Exception as e:
            print(f"Error reading report: {e}")
    
    def view_evidence_videos(self):
        """View evidence videos"""
        if not os.path.exists('output/evidence'):
            print("No evidence videos found")
            return
        
        videos = [f for f in os.listdir('output/evidence') if f.endswith('.mp4')]
        
        if not videos:
            print("No evidence videos found")
            return
        
        print("\n🎥 EVIDENCE VIDEOS")
        print("-" * 40)
        
        for i, video in enumerate(sorted(videos, reverse=True)[:5], 1):
            size = os.path.getsize(f'output/evidence/{video}') / (1024*1024)
            print(f"{i}. {video} ({size:.1f} MB)")
        
        try:
            choice = int(input(f"\nSelect video to view (1-{min(5, len(videos))}): "))
            if 1 <= choice <= min(5, len(videos)):
                video_file = f"output/evidence/{sorted(videos, reverse=True)[choice-1]}"
                
                print(f"\nPlaying: {video_file}")
                print("Press 'q' to stop playback")
                
                cap = cv2.VideoCapture(video_file)
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Resize for display
                    display = cv2.resize(frame, (800, 450))
                    cv2.imshow("DRISTI - Evidence Video", display)
                    
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                
                cap.release()
                cv2.destroyAllWindows()
            else:
                print("Invalid selection")
        except ValueError:
            print("Please enter a number")
    
    def exit_system(self):
        """Exit the system"""
        print("\n" + "="*70)
        print("Thank you for using DRISTI!")
        print("Making the world safer, one face at a time.")
        print("="*70)

def main():
    """Main function"""
    print("\n" + "="*70)
    print("🚀 DRISTI - FAST ACCURATE FACE FINDER")
    print("="*70)
    print("Version 3.0 | Multi-CCTV | Parallel Processing")
    print("="*70)
    
    try:
        menu = FastMenu()
        menu.main_menu()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nDRISTI system closed.")

if __name__ == "__main__":
    main()