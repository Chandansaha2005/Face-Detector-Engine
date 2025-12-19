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
import mediapipe as mp
import warnings
warnings.filterwarnings('ignore')

class AccurateFaceRecognizer:
    """Accurate face recognition using deep learning with live preview"""
    
    def __init__(self):
        print("="*70)
        print("DRISTI - ACCURATE FACE RECOGNITION")
        print("="*70)
        
        # Initialize MediaPipe for accurate face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for long-range (CCTV)
            min_detection_confidence=0.7
        )
        
        # Load face recognition model
        self.face_model = self.load_face_model()
        
        # Target information
        self.target_embedding = None
        self.target_name = ""
        self.target_face = None
        
        # Settings
        self.similarity_threshold = 0.75
        self.detection_confidence = 0.7
        self.show_preview = True
        self.frame_skip = 3
        
        # Create directories
        self.create_directories()
        
        print("✓ System initialized")
        print(f"  Similarity threshold: {self.similarity_threshold*100}%")
        print(f"  Live preview: {'Enabled' if self.show_preview else 'Disabled'}")
        print("="*70)
    
    def load_face_model(self):
        """Load pre-trained ResNet for face recognition"""
        print("Loading face recognition model...")
        
        try:
            import torchvision.models as models
            model = models.resnet18(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-1])
            model.eval()
            print("✓ ResNet18 loaded for face features")
            return model
        except Exception as e:
            print(f"✗ Could not load ResNet: {e}")
            return None
    
    def create_directories(self):
        """Create all necessary directories"""
        dirs = [
            'output',
            'output/evidence',
            'output/reports',
            'output/snapshots',
            'output/alerts',
            'input'
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def extract_face_embedding(self, face_image):
        """Extract deep embedding from face"""
        try:
            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # Convert to PIL
            pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            
            # Apply transforms
            image_tensor = transform(pil_image).unsqueeze(0)
            
            # Extract features
            with torch.no_grad():
                embedding = self.face_model(image_tensor)
            
            # Convert to numpy and normalize
            embedding = embedding.squeeze().numpy()
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            # Fallback to simple features
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (100, 100))
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
    
    def detect_faces(self, image):
        """Detect faces in image using MediaPipe"""
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.face_detection.process(rgb_image)
        
        faces = []
        
        if results.detections:
            for detection in results.detections:
                confidence = detection.score[0]
                
                if confidence < self.detection_confidence:
                    continue
                
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                box_width = int(bboxC.width * w)
                box_height = int(bboxC.height * h)
                
                # Add padding
                padding = int(min(box_width, box_height) * 0.2)
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x1 + box_width + 2*padding)
                y2 = min(h, y1 + box_height + 2*padding)
                
                # Extract face
                face_roi = image[y1:y2, x1:x2]
                
                if face_roi.size > 0:
                    faces.append({
                        'bbox': (x1, y1, x2, y2),
                        'face': face_roi,
                        'confidence': float(confidence)
                    })
        
        return faces
    
    def calculate_similarity(self, emb1, emb2):
        """Calculate cosine similarity between embeddings"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        try:
            # Ensure same dimension
            min_len = min(len(emb1), len(emb2))
            emb1_trunc = emb1[:min_len]
            emb2_trunc = emb2[:min_len]
            
            # Calculate cosine similarity
            similarity = np.dot(emb1_trunc, emb2_trunc) / (
                np.linalg.norm(emb1_trunc) * np.linalg.norm(emb2_trunc)
            )
            
            return float(similarity)
        except:
            return 0.0
    
    def set_target_person(self, image_path, person_name=""):
        """Set the target person to find"""
        print(f"\n🎯 SETTING TARGET PERSON")
        print("-" * 40)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"✗ Cannot read image: {image_path}")
            return False
        
        # Detect faces
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            print("✗ No face found in target image")
            return False
        
        # Use the face with highest confidence
        best_face = max(faces, key=lambda x: x['confidence'])
        
        # Extract embedding
        self.target_embedding = self.extract_face_embedding(best_face['face'])
        self.target_name = person_name or os.path.basename(image_path).split('.')[0]
        self.target_face = best_face['face']
        
        # Save target face
        target_path = f'output/target_{self.target_name}.jpg'
        cv2.imwrite(target_path, self.target_face)
        
        print(f"✓ Target person set: {self.target_name}")
        print(f"  Face size: {best_face['face'].shape}")
        print(f"  Saved to: {target_path}")
        
        # Show target briefly
        cv2.imshow(f'Target: {self.target_name}', cv2.resize(self.target_face, (300, 300)))
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
        return True
    
    def search_in_video(self, video_path, camera_id="CCTV_1"):
        """Search for target person in a video with live preview"""
        print(f"\n🔍 PROCESSING: {camera_id}")
        print(f"  Video: {os.path.basename(video_path)}")
        
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
        matches = []
        frame_count = 0
        match_count = 0
        start_time = time.time()
        
        print("  Live preview with controls:")
        print("    [q] - Quit | [p] - Pause | [s] - Save frame")
        
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
                faces = self.detect_faces(frame)
                
                # Check each face
                current_match = None
                for face in faces:
                    # Extract embedding
                    face_embedding = self.extract_face_embedding(face['face'])
                    
                    # Calculate similarity
                    similarity = self.calculate_similarity(
                        self.target_embedding, 
                        face_embedding
                    )
                    
                    # Check if it's a match
                    if similarity >= self.similarity_threshold:
                        match_count += 1
                        current_match = {
                            'bbox': face['bbox'],
                            'similarity': similarity,
                            'frame': frame_count,
                            'time': frame_count / fps
                        }
                        matches.append(current_match)
                        
                        # Save first match snapshot
                        if match_count == 1:
                            snap_path = f"output/snapshots/{camera_id}_match.jpg"
                            cv2.imwrite(snap_path, face['face'])
                        
                        # Save alert
                        alert_path = f"output/alerts/{camera_id}_alert_{match_count}.jpg"
                        cv2.imwrite(alert_path, frame)
                
                # Draw results
                result_frame = self.draw_detections(frame, faces, current_match, camera_id, frame_count, total_frames)
                
                # Write to output
                out.write(result_frame)
                
                # Show preview
                if self.show_preview:
                    preview = cv2.resize(result_frame, (800, 450))
                    cv2.imshow(f'DRISTI - {camera_id}', preview)
            
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
            self.generate_report(camera_id, video_path, output_path, matches, elapsed)
        else:
            print(f"    ℹ️  No matches found in {camera_id}")
        
        return {
            'camera': camera_id,
            'matches': match_count,
            'processing_time': elapsed,
            'output_video': output_path,
            'match_details': matches
        }
    
    def draw_detections(self, frame, faces, match_info=None, camera_name="", current_frame=0, total_frames=0):
        """Draw detections on frame with detailed info"""
        result = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw all detected faces
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            confidence = face['confidence']
            
            # Check if this face is a match
            is_match = False
            if match_info and match_info['bbox'] == (x1, y1, x2, y2):
                is_match = True
            
            if is_match:
                # MATCH - Green box
                color = (0, 255, 0)
                thickness = 3
                label = f"{self.target_name}: {match_info['similarity']:.1%}"
            else:
                # NOT A MATCH - Gray box
                color = (100, 100, 100)
                thickness = 1
                label = f"Face: {confidence:.0%}"
            
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
            
            # Draw "FOUND" for matches
            if is_match and match_info['similarity'] > 0.8:
                cv2.putText(result, "✅ FOUND", (x1, y1 - 40),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        
        # Add status bar at top
        status_bar = np.zeros((50, width, 3), dtype=np.uint8)
        status_bar[:] = (30, 30, 30)
        result[:50, :] = status_bar
        
        # Camera name
        cv2.putText(result, f"Camera: {camera_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Frame counter
        frame_text = f"Frame: {current_frame}/{total_frames}"
        cv2.putText(result, frame_text, (width // 3, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Target name
        cv2.putText(result, f"Target: {self.target_name}", (width // 2, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(result, timestamp, (width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add controls info at bottom
        if self.show_preview:
            controls_bar = np.zeros((40, width, 3), dtype=np.uint8)
            controls_bar[:] = (30, 30, 30)
            result[height-40:height, :] = controls_bar
            
            controls = "[q] Quit | [p] Pause | [s] Save Frame"
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
            f.write(f"Processing Time: {processing_time:.1f} seconds\n\n")
            
            if matches:
                best_match = max(matches, key=lambda x: x['similarity'])
                
                f.write("BEST MATCH DETAILS:\n")
                f.write(f"  Similarity: {best_match['similarity']:.1%}\n")
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
        
        print(f"📄 Report saved: {report_file}")

class DristiMenu:
    """Menu system for DRISTI"""
    
    def __init__(self):
        self.recognizer = AccurateFaceRecognizer()
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
        print("🚀 DRISTI - ACCURATE FACE RECOGNITION")
        print("="*70)
        print(f"Target: {self.recognizer.target_name if self.recognizer.target_name else 'Not Set'}")
        print(f"CCTV Videos: {len(self.cctv_videos)} found")
        print(f"Live Preview: {'Enabled' if self.recognizer.show_preview else 'Disabled'}")
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
            print("4. ⚡ Settings")
            print("5. 📁 View Files")
            print("6. 🚪 Exit")
            print("-" * 40)
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == "1":
                self.set_target()
            elif choice == "2":
                self.search_single()
            elif choice == "3":
                self.search_multiple()
            elif choice == "4":
                self.settings()
            elif choice == "5":
                self.view_files()
            elif choice == "6":
                print("\nThank you for using DRISTI!")
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
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if photos:
            print("Available photos in 'input/' folder:")
            for i, p in enumerate(photos, 1):
                print(f"{i}. {p}")
            print(f"{len(photos)+1}. Enter custom path")
        else:
            print("No photos found. Enter custom path:")
        
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
            
            if self.recognizer.set_target_person(path, name):
                print(f"\n✅ Target set: {self.recognizer.target_name}")
            else:
                print("\n❌ Failed to set target")
        
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
                self.recognizer.search_in_video(video_path, cam_name)
                
                print("\n✅ Search completed!")
                print("   Check 'output/' folder for results.")
            else:
                print("Invalid selection")
        except ValueError:
            print("Please enter a number")
        
        input("\nPress Enter to continue...")
    
    def search_multiple(self):
        """Search in multiple CCTV cameras"""
        if self.recognizer.target_embedding is None:
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
        
        print(f"\nSelected {len(selected_cameras)} cameras...")
        
        # Process each camera
        all_results = []
        for cam_name, video_path in selected_cameras.items():
            print(f"\n{'='*60}")
            print(f"Processing: {cam_name}")
            print(f"{'='*60}")
            
            result = self.recognizer.search_in_video(video_path, cam_name)
            if result:
                all_results.append(result)
        
        # Generate final report
        self.generate_final_report(all_results)
        
        input("\nPress Enter to continue...")
    
    def generate_final_report(self, results):
        """Generate final search report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"output/reports/final_report_{timestamp}.txt"
        
        total_matches = sum(r['matches'] for r in results)
        cameras_with_matches = [r for r in results if r['matches'] > 0]
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DRISTI - FINAL SEARCH REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Target Person: {self.recognizer.target_name}\n")
            f.write(f"Search Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Cameras Searched: {len(results)}\n")
            f.write(f"Cameras with Matches: {len(cameras_with_matches)}\n")
            f.write(f"Total Matches Found: {total_matches}\n\n")
            
            if cameras_with_matches:
                f.write("PERSON WAS FOUND IN:\n")
                f.write("-" * 50 + "\n")
                for result in cameras_with_matches:
                    if result['match_details']:
                        best = max(result['match_details'], key=lambda x: x['similarity'])
                        f.write(f"\n📹 {result['camera']}\n")
                        f.write(f"   Matches: {result['matches']}\n")
                        f.write(f"   Best similarity: {best['similarity']:.1%}\n")
                        f.write(f"   Time: {best['time']:.1f}s\n")
                        f.write(f"   Evidence: {os.path.basename(result['output_video'])}\n")
            else:
                f.write("\n⚠️  Person not found in any camera\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        print(f"\n📄 Final report saved: {report_file}")
    
    def settings(self):
        """Configure settings"""
        self.display_header()
        print("\n⚡ SETTINGS")
        print("-" * 40)
        
        print(f"Current Settings:")
        print(f"1. Similarity Threshold: {self.recognizer.similarity_threshold*100:.0f}%")
        print(f"2. Frame Skip: {self.recognizer.frame_skip}x")
        print(f"3. Live Preview: {'Enabled' if self.recognizer.show_preview else 'Disabled'}")
        
        print("\nAdjust settings (enter number to change, or 0 to go back):")
        
        try:
            choice = input("\nSelect setting to change (1-3): ").strip()
            
            if choice == "1":
                new_val = float(input(f"New similarity threshold (0.1-1.0, current: {self.recognizer.similarity_threshold}): "))
                if 0.1 <= new_val <= 1.0:
                    self.recognizer.similarity_threshold = new_val
                    print(f"✓ Threshold set to {new_val*100:.0f}%")
                else:
                    print("Invalid value")
            
            elif choice == "2":
                new_val = int(input(f"New frame skip (1-10, current: {self.recognizer.frame_skip}): "))
                if 1 <= new_val <= 10:
                    self.recognizer.frame_skip = new_val
                    print(f"✓ Frame skip set to {new_val}x")
                else:
                    print("Invalid value")
            
            elif choice == "3":
                self.recognizer.show_preview = not self.recognizer.show_preview
                print(f"✓ Live preview {'enabled' if self.recognizer.show_preview else 'disabled'}")
            
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
                    for file in files[:10]:  # Show first 10 files
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
    print("Starting DRISTI Face Recognition System...")
    print("Install required packages if not already installed:")
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
        menu = DristiMenu()
        menu.main_menu()
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("\nPress Enter to exit...")