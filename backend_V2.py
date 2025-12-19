import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
import mediapipe as mp
import warnings
warnings.filterwarnings('ignore')

class SimpleAccurateFinder:
    """Simplified but accurate face finder - NO TORCH NEEDED"""
    
    def __init__(self):
        print("="*70)
        print("DRISTI - SIMPLE ACCURATE FINDER")
        print("MediaPipe + Smart Filtering")
        print("="*70)
        
        # Initialize MediaPipe
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
        
        # Smart thresholds
        self.detection_thresh = 0.7
        self.similarity_thresh = 0.75  # 75% match required
        self.min_consecutive = 3
        
        # Tracking
        self.match_history = {}
        
        # Create directories
        self.create_directories()
        
        print("✓ System Ready (No model download needed)")
        print(f"  Similarity Threshold: {self.similarity_thresh*100}%")
        print(f"  Min Consecutive Matches: {self.min_consecutive}")
        print("="*70)
    
    def create_directories(self):
        """Create output directories"""
        dirs = ['output', 'output/evidence', 'output/reports', 
                'output/snapshots', 'input']
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def extract_face_features(self, face_image):
        """Extract features using MediaPipe landmarks"""
        try:
            # Convert to RGB
            rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Get face mesh landmarks
            results = self.face_mesh.process(rgb_face)
            
            if results.multi_face_landmarks:
                # Extract all 468 landmarks
                landmarks = []
                for landmark in results.multi_face_landmarks[0].landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                features = np.array(landmarks)
                
                # Add LBP texture features
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (100, 100))
                gray = cv2.equalizeHist(gray)
                
                # Simple LBP
                lbp = np.zeros_like(gray)
                for i in range(1, gray.shape[0]-1):
                    for j in range(1, gray.shape[1]-1):
                        center = gray[i, j]
                        code = 0
                        code |= (gray[i-1, j-1] >= center) << 7
                        code |= (gray[i-1, j] >= center) << 6
                        code |= (gray[i-1, j+1] >= center) << 5
                        code |= (gray[i, j+1] >= center) << 4
                        code |= (gray[i+1, j+1] >= center) << 3
                        code |= (gray[i+1, j] >= center) << 2
                        code |= (gray[i+1, j-1] >= center) << 1
                        code |= (gray[i, j-1] >= center) << 0
                        lbp[i-1, j-1] = code
                
                lbp_flat = lbp.flatten() / 255.0
                
                # Combine landmarks + texture
                combined = np.concatenate([features, lbp_flat])
                combined = combined / np.linalg.norm(combined)
                
                return combined
        except:
            pass
        
        # Fallback: simple histogram features
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def set_target_person(self, photo_path, name=""):
        """Set target person"""
        print(f"\n🎯 Setting target person...")
        
        if not os.path.exists(photo_path):
            print(f"✗ Photo not found: {photo_path}")
            return False
        
        img = cv2.imread(photo_path)
        if img is None:
            print("✗ Cannot read image")
            return False
        
        # Detect face
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)
        
        if not results.detections:
            print("✗ No face found")
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
        pad = 20
        x1, y1 = max(0, x1-pad), max(0, y1-pad)
        x2, y2 = min(w, x2+pad), min(h, y2+pad)
        
        face_img = img[y1:y2, x1:x2]
        
        if face_img.size == 0:
            print("✗ Could not extract face")
            return False
        
        # Extract features
        self.target_features = self.extract_face_features(face_img)
        self.target_name = name or os.path.basename(photo_path).split('.')[0]
        self.target_face_img = face_img
        
        # Save target face
        cv2.imwrite(f'output/target_{self.target_name}.jpg', face_img)
        
        print(f"✅ Target set: {self.target_name}")
        print(f"  Face size: {face_img.shape}")
        print(f"  Features: {len(self.target_features)} dimensions")
        
        # Show target
        cv2.imshow('Target Person', cv2.resize(face_img, (300, 300)))
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
        return True
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)
        
        faces = []
        
        if results.detections:
            for det in results.detections:
                conf = det.score[0]
                if conf < self.detection_thresh:
                    continue
                
                bboxC = det.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                bw = int(bboxC.width * w)
                bh = int(bboxC.height * h)
                
                # Filter small faces
                if bw * bh < 2500:
                    continue
                
                # Add padding
                pad = 20
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x1 + bw + 2*pad)
                y2 = min(h, y1 + bh + 2*pad)
                
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size > 0:
                    faces.append({
                        'bbox': (x1, y1, x2, y2),
                        'face': face_img,
                        'confidence': float(conf)
                    })
        
        return faces
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature vectors"""
        min_len = min(len(features1), len(features2))
        f1 = features1[:min_len]
        f2 = features2[:min_len]
        
        # Cosine similarity
        dot = np.dot(f1, f2)
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot / (norm1 * norm2)
            return max(0, min(1, similarity))  # Clamp to [0, 1]
        
        return 0.0
    
    def is_real_match(self, face_img, frame_num, location_key):
        """Check if this is a real match"""
        if self.target_features is None:
            return False, 0.0
        
        # Extract features
        features = self.extract_face_features(face_img)
        if features is None:
            return False, 0.0
        
        # Calculate similarity
        similarity = self.calculate_similarity(self.target_features, features)
        
        if similarity < self.similarity_thresh:
            return False, similarity
        
        # Track consecutive matches
        if location_key not in self.match_history:
            self.match_history[location_key] = []
        
        self.match_history[location_key].append({
            'frame': frame_num,
            'similarity': similarity,
            'time': time.time()
        })
        
        # Keep only recent (last 5 seconds)
        current_time = time.time()
        self.match_history[location_key] = [
            m for m in self.match_history[location_key]
            if current_time - m['time'] < 5
        ]
        
        # Need consecutive matches
        if len(self.match_history[location_key]) >= self.min_consecutive:
            avg_sim = np.mean([m['similarity'] for m in self.match_history[location_key]])
            return True, avg_sim
        
        return False, similarity
    
    def search_in_video(self, video_path, camera_name="CCTV_1"):
        """Search for target in video"""
        print(f"\n🔍 Searching in: {camera_name}")
        print(f"  Video: {os.path.basename(video_path)}")
        print("-" * 40)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("✗ Cannot open video")
            return None
        
        # Video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"  Size: {w}x{h}, FPS: {fps}, Duration: {total_frames/fps:.1f}s")
        
        # Output video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/evidence/{camera_name}_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Tracking
        matches = []
        frame_count = 0
        match_count = 0
        
        print("  Processing... (Q to quit, SPACE to pause)")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for speed
                if frame_count % 2 != 0:
                    continue
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Check each face
                current_match = None
                
                for face in faces:
                    # Create location key
                    x1, y1, x2, y2 = face['bbox']
                    loc_key = f"{x1//50}_{y1//50}"
                    
                    # Check if match
                    is_match, similarity = self.is_real_match(
                        face['face'], frame_count, loc_key
                    )
                    
                    if is_match:
                        match_count += 1
                        current_match = {
                            'bbox': face['bbox'],
                            'similarity': similarity,
                            'frame': frame_count,
                            'time': frame_count / fps
                        }
                        matches.append(current_match)
                        
                        # Save first few matches
                        if match_count <= 3:
                            snap_path = f"output/snapshots/{camera_name}_match_{match_count}.jpg"
                            cv2.imwrite(snap_path, face['face'])
                
                # Draw results
                result_frame = self.draw_results(frame, faces, current_match)
                out.write(result_frame)
                
                # Show preview every 0.5 seconds
                if frame_count % (fps // 2) == 0:
                    preview = cv2.resize(result_frame, (800, 450))
                    status = f"Frame: {frame_count}/{total_frames} | Matches: {match_count}"
                    cv2.putText(preview, status, (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.imshow(f'DRISTI - {camera_name}', preview)
            
            # Handle keys
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            
            if key == ord('q'):
                print("\n🛑 Stopped by user")
                break
            elif key == ord(' '):
                paused = not paused
                print("⏸️  Paused" if paused else "▶️  Resumed")
            elif key == ord('s'):
                snap = f"output/snapshots/{camera_name}_frame_{frame_count}.jpg"
                cv2.imwrite(snap, frame)
                print(f"📸 Saved: {snap}")
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Generate report
        if matches:
            self.generate_report(matches, camera_name, video_path, output_path)
            print(f"\n✅ Found {match_count} matches in {camera_name}")
            print(f"   Evidence: {output_path}")
        else:
            print(f"\nℹ️  No matches found in {camera_name}")
        
        return matches
    
    def draw_results(self, frame, faces, match_info=None):
        """Draw detection results"""
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Draw all faces (faint gray)
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            cv2.rectangle(result, (x1, y1), (x2, y2), (80, 80, 80), 1)
            
            # Small confidence
            conf_text = f"{face['confidence']:.0%}"
            cv2.putText(result, conf_text, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # Draw match (prominent green)
        if match_info:
            x1, y1, x2, y2 = match_info['bbox']
            similarity = match_info['similarity']
            
            # Green box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # "FOUND" text
            cv2.putText(result, "✅ FOUND", (x1, y1 - 40),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
            
            # Similarity
            if similarity > 0.8:
                color = (0, 255, 0)  # Green
                conf = "HIGH"
            elif similarity > 0.7:
                color = (0, 200, 255)  # Yellow
                conf = "MEDIUM"
            else:
                color = (0, 150, 255)  # Orange
                conf = "LOW"
            
            label = f"{self.target_name}: {similarity:.1%} ({conf})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(result,
                         (x1, y2 - label_size[1] - 10),
                         (x1 + label_size[0], y2),
                         color, cv2.FILLED)
            
            cv2.putText(result, label, (x1, y2 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Header
        cv2.rectangle(result, (0, 0), (w, 40), (0, 0, 0), -1)
        header = f"DRISTI AI | Target: {self.target_name}"
        cv2.putText(result, header, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Timestamp
        time_str = datetime.now().strftime("%H:%M:%S")
        cv2.putText(result, time_str, (w - 100, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result
    
    def generate_report(self, matches, camera_name, video_path, output_path):
        """Generate report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"output/reports/{camera_name}_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DRISTI - FACE MATCH REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Target Person: {self.target_name}\n")
            f.write(f"Camera: {camera_name}\n")
            f.write(f"Video: {os.path.basename(video_path)}\n")
            f.write(f"Search Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("="*70 + "\n")
            f.write("MATCH DETAILS\n")
            f.write("="*70 + "\n\n")
            
            if matches:
                best = max(matches, key=lambda x: x['similarity'])
                avg = np.mean([m['similarity'] for m in matches])
                
                f.write(f"Total Matches: {len(matches)}\n")
                f.write(f"Best Match: {best['similarity']:.1%}\n")
                f.write(f"Average Similarity: {avg:.1%}\n\n")
                
                f.write("Timeline:\n")
                for i, match in enumerate(matches[:10], 1):
                    f.write(f"{i}. Time: {match['time']:.1f}s | ")
                    f.write(f"Similarity: {match['similarity']:.1%} | ")
                    f.write(f"Frame: {match['frame']}\n")
            else:
                f.write("No matches found\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("EVIDENCE FILES\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"1. Evidence Video: {os.path.basename(output_path)}\n")
            f.write(f"2. Target Face: target_{self.target_name}.jpg\n")
            f.write(f"3. Match Snapshots: {camera_name}_match_*.jpg\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("VERDICT: ")
            
            if matches and best['similarity'] > 0.8:
                f.write("✅ PERSON POSITIVELY IDENTIFIED\n")
            elif matches:
                f.write("⚠️  POSSIBLE MATCH - VERIFY MANUALLY\n")
            else:
                f.write("❌ NO MATCH FOUND\n")
            
            f.write("="*70 + "\n")
        
        print(f"📄 Report saved: {report_file}")

class SimpleMenu:
    """Simple menu system"""
    
    def __init__(self):
        self.finder = SimpleAccurateFinder()
    
    def run(self):
        """Run the menu"""
        while True:
            print("\n" + "="*60)
            print("DRISTI - SIMPLE ACCURATE FINDER")
            print("="*60)
            print("1. 🎯 Set Target Person")
            print("2. 🔍 Search in CCTV Video")
            print("3. 📊 View Reports")
            print("4. ℹ️  Help")
            print("5. 🚪 Exit")
            print("="*60)
            
            choice = input("Select option (1-5): ").strip()
            
            if choice == "1":
                self.set_target()
            elif choice == "2":
                self.search_video()
            elif choice == "3":
                self.view_reports()
            elif choice == "4":
                self.show_help()
            elif choice == "5":
                print("\nThank you for using DRISTI!")
                break
            else:
                print("Invalid option")
    
    def set_target(self):
        """Set target person"""
        print("\n🎯 SET TARGET PERSON")
        print("-" * 40)
        
        # Check for photos
        photos = [f for f in os.listdir('input') 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if photos:
            print("Available photos:")
            for i, p in enumerate(photos, 1):
                print(f"{i}. {p}")
            print(f"{len(photos)+1}. Enter custom path")
        else:
            print("No photos in 'input/' folder")
            photos = []
        
        try:
            if photos:
                choice = input(f"\nSelect (1-{len(photos)+1}): ").strip()
                if choice.isdigit():
                    choice = int(choice)
                    if 1 <= choice <= len(photos):
                        path = f"input/{photos[choice-1]}"
                    elif choice == len(photos) + 1:
                        path = input("Enter path: ").strip()
                    else:
                        print("Invalid")
                        return
                else:
                    path = choice
            else:
                path = input("Enter photo path: ").strip()
            
            name = input("Person name (optional): ").strip()
            
            print(f"\nSetting target from: {path}")
            if self.finder.set_target_person(path, name):
                print(f"✅ Target set: {self.finder.target_name}")
            else:
                print("❌ Failed to set target")
        
        except Exception as e:
            print(f"Error: {e}")
        
        input("\nPress Enter to continue...")
    
    def search_video(self):
        """Search in video"""
        if self.finder.target_features is None:
            print("⚠️  Set target person first!")
            input("\nPress Enter to continue...")
            return
        
        print("\n🔍 SEARCH IN VIDEO")
        print("-" * 40)
        
        # Find videos
        videos = [f for f in os.listdir('input') 
                 if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        
        if not videos:
            print("No videos in 'input/' folder")
            print("Add videos as: cctv1.mp4, cctv2.mp4, etc.")
            input("\nPress Enter to continue...")
            return
        
        print("Available videos:")
        for i, v in enumerate(videos, 1):
            print(f"{i}. {v}")
        
        try:
            choice = int(input(f"\nSelect video (1-{len(videos)}): "))
            if 1 <= choice <= len(videos):
                video_path = f"input/{videos[choice-1]}"
                cam_name = os.path.splitext(videos[choice-1])[0]
                
                print(f"\nSearching in: {cam_name}")
                self.finder.search_in_video(video_path, cam_name)
            else:
                print("Invalid selection")
        except ValueError:
            print("Enter a number")
        
        input("\nPress Enter to continue...")
    
    def view_reports(self):
        """View reports"""
        reports_dir = 'output/reports'
        if not os.path.exists(reports_dir):
            print("No reports found")
            input("\nPress Enter to continue...")
            return
        
        reports = [f for f in os.listdir(reports_dir) if f.endswith('.txt')]
        
        if not reports:
            print("No reports")
            input("\nPress Enter to continue...")
            return
        
        print("\n📊 RECENT REPORTS")
        print("-" * 40)
        for i, r in enumerate(sorted(reports, reverse=True)[:5], 1):
            print(f"{i}. {r}")
        
        try:
            choice = int(input(f"\nSelect report (1-{min(5, len(reports))}): "))
            if 1 <= choice <= min(5, len(reports)):
                report_file = f"{reports_dir}/{sorted(reports, reverse=True)[choice-1]}"
                with open(report_file, 'r') as f:
                    print("\n" + "="*60)
                    print(f.read())
            else:
                print("Invalid")
        except:
            print("Error reading report")
        
        input("\nPress Enter to continue...")
    
    def show_help(self):
        """Show help"""
        print("\nℹ️  HELP")
        print("-" * 40)
        print("1. Add files to 'input/' folder:")
        print("   - missing_photo.png (clear face photo)")
        print("   - cctv1.mp4, cctv2.mp4 (videos with faces)")
        print("2. Set target person first")
        print("3. Search in videos")
        print("4. Check 'output/' folder for results")
        print("\n📁 Folder structure:")
        print("   input/      - Photos & videos")
        print("   output/     - Results")
        print("     evidence/ - Marked videos")
        print("     reports/  - Search reports")
        print("     snapshots/- Match images")
        print("\n🎯 For best accuracy:")
        print("   - Use clear, front-facing photos")
        print("   - Good lighting in videos")
        print("   - Target should appear in videos")
        
        input("\nPress Enter to continue...")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("DRISTI - SIMPLE ACCURATE FACE FINDER")
    print("No Model Download Needed")
    print("="*70)
    
    try:
        menu = SimpleMenu()
        menu.run()
    except KeyboardInterrupt:
        print("\n\nProgram stopped.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()