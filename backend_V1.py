import cv2
import numpy as np
import os
import json
from datetime import datetime
import torch
import torchvision.transforms as transforms
from PIL import Image
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class AccurateFaceRecognizer:
    """Accurate face recognition using deep learning"""
    
    def __init__(self):
        print("="*70)
        print("DRISTI - ACCURATE FACE RECOGNITION")
        print("="*70)
        
        # Initialize MediaPipe for face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for long-range (CCTV)
            min_detection_confidence=0.7  # Higher confidence
        )
        
        # Load face recognition model (using ResNet)
        self.face_model = self.load_face_model()
        
        # Face database
        self.target_embedding = None
        self.target_name = ""
        
        # Thresholds
        self.similarity_threshold = 0.75  # 75% similarity (STRICT)
        self.detection_confidence = 0.7
        
        # Create directories
        os.makedirs('output', exist_ok=True)
        os.makedirs('input', exist_ok=True)
        
        print("✓ System initialized")
        print(f"  Similarity threshold: {self.similarity_threshold*100}%")
        print(f"  Detection confidence: {self.detection_confidence*100}%")
    
    def load_face_model(self):
        """Load pre-trained face recognition model"""
        print("Loading face recognition model...")
        
        try:
            # Try to load pre-trained ResNet
            import torchvision.models as models
            model = models.resnet18(pretrained=True)
            
            # Remove last layer for feature extraction
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model.eval()  # Set to evaluation mode
            
            print("✓ ResNet18 loaded for face features")
            return model
            
        except Exception as e:
            print(f"✗ Could not load ResNet: {e}")
            print("  Using MediaPipe Face Landmarks instead")
            return None
    
    def extract_face_embedding(self, face_image):
        """Extract deep embedding from face"""
        if self.face_model is not None:
            return self.extract_deep_embedding(face_image)
        else:
            return self.extract_mediapipe_embedding(face_image)
    
    def extract_deep_embedding(self, face_image):
        """Extract embedding using ResNet"""
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
            
            # Convert to numpy array
            embedding = embedding.squeeze().numpy()
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Error in deep embedding: {e}")
            return self.extract_mediapipe_embedding(face_image)
    
    def extract_mediapipe_embedding(self, face_image):
        """Extract embedding using MediaPipe landmarks (more accurate)"""
        try:
            # Initialize face mesh
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
            
            # Convert to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Process
            results = face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                # Extract 468 landmarks
                landmarks = []
                for landmark in results.multi_face_landmarks[0].landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                embedding = np.array(landmarks)
                
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)
                
                return embedding
            
            # Fallback to basic features
            return self.extract_basic_features(face_image)
            
        except:
            return self.extract_basic_features(face_image)
    
    def extract_basic_features(self, face_image):
        """Extract basic features as fallback"""
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Resize
        gray = cv2.resize(gray, (100, 100))
        
        # Apply histogram equalization
        gray = cv2.equalizeHist(gray)
        
        # Extract LBP features (more distinctive)
        lbp = self.extract_lbp_features(gray)
        
        # Normalize
        features = np.concatenate([gray.flatten()/255.0, lbp])
        features = features / np.linalg.norm(features)
        
        return features
    
    def extract_lbp_features(self, image):
        """Extract Local Binary Pattern features (good for faces)"""
        rows, cols = image.shape
        lbp = np.zeros((rows-2, cols-2), dtype=np.uint8)
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] >= center) << 7
                code |= (image[i-1, j] >= center) << 6
                code |= (image[i-1, j+1] >= center) << 5
                code |= (image[i, j+1] >= center) << 4
                code |= (image[i+1, j+1] >= center) << 3
                code |= (image[i+1, j] >= center) << 2
                code |= (image[i+1, j-1] >= center) << 1
                code |= (image[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code
        
        return lbp.flatten()
    
    def detect_faces(self, image):
        """Detect faces in image"""
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
                padding = 20
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
    
    def set_target_person(self, image_path, person_name=""):
        """Set the target person to find"""
        print(f"\n[1/3] Setting target person: {person_name}")
        
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
        self.target_name = person_name
        
        print(f"✓ Target person set")
        print(f"  Face size: {best_face['face'].shape}")
        print(f"  Embedding size: {len(self.target_embedding)}")
        
        # Save target face for reference
        cv2.imwrite('output/target_face.jpg', best_face['face'])
        
        return True
    
    def search_in_video(self, video_path, camera_id="CCTV_1", save_output=True):
        """Search for target person in a video"""
        print(f"\n[2/3] Searching in: {os.path.basename(video_path)}")
        print(f"  Camera: {camera_id}")
        
        if self.target_embedding is None:
            print("✗ Target person not set. Call set_target_person() first.")
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
        
        print(f"  Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Prepare output video
        output_path = None
        out = None
        if save_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output/{camera_id}_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Track matches
        matches = []
        frame_count = 0
        processed_frames = 0
        
        print("  Processing... (Press 'q' in preview window to skip)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 3rd frame for speed
            if frame_count % 3 != 0:
                continue
            
            processed_frames += 1
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Check each face
            frame_matches = []
            for face in faces:
                # Extract embedding
                face_embedding = self.extract_face_embedding(face['face'])
                
                # Calculate similarity
                similarity = self.calculate_similarity(
                    self.target_embedding, 
                    face_embedding
                )
                
                # Check if it's a match
                is_match = similarity >= self.similarity_threshold
                
                if is_match:
                    match_info = {
                        'frame': frame_count,
                        'time_seconds': frame_count / fps,
                        'similarity': similarity,
                        'bbox': face['bbox'],
                        'camera': camera_id
                    }
                    matches.append(match_info)
                    frame_matches.append(match_info)
                    
                    # Print first few matches
                    if len(matches) <= 3:
                        print(f"    ✓ Match at {match_info['time_seconds']:.1f}s: {similarity:.1%}")
            
            # Draw results
            result_frame = self.draw_detections(frame, faces, frame_matches)
            
            # Write to output
            if out is not None:
                out.write(result_frame)
            
            # Show preview (every 2 seconds)
            if processed_frames % (fps * 2) == 0:
                preview = cv2.resize(result_frame, (800, 450))
                cv2.imshow(f'DRISTI - {camera_id}', preview)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("  ⏩ Skipping remaining frames...")
                    break
        
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        # Save match information
        if matches:
            self.save_matches(matches, camera_id, video_path, output_path)
        
        return {
            'camera_id': camera_id,
            'video_path': video_path,
            'total_frames': frame_count,
            'processed_frames': processed_frames,
            'faces_detected': len(faces),
            'matches_found': len(matches),
            'match_details': matches,
            'output_video': output_path
        }
    
    def calculate_similarity(self, emb1, emb2):
        """Calculate cosine similarity between embeddings"""
        # Ensure same shape
        min_len = min(len(emb1), len(emb2))
        emb1_trunc = emb1[:min_len]
        emb2_trunc = emb2[:min_len]
        
        # Calculate cosine similarity
        similarity = np.dot(emb1_trunc, emb2_trunc) / (
            np.linalg.norm(emb1_trunc) * np.linalg.norm(emb2_trunc)
        )
        
        return float(similarity)
    
    def draw_detections(self, frame, faces, matches):
        """Draw detections on frame"""
        result = frame.copy()
        
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            confidence = face['confidence']
            
            # Check if this face is a match
            is_match = False
            match_similarity = 0
            
            for match in matches:
                if match['bbox'] == (x1, y1, x2, y2):
                    is_match = True
                    match_similarity = match['similarity']
                    break
            
            if is_match:
                # MATCH - Green box
                color = (0, 255, 0)  # Green
                thickness = 3
                
                # Draw "FOUND" text
                cv2.putText(result, "FOUND", (x1, y1 - 40),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
                
                # Draw match percentage
                label = f"{self.target_name}: {match_similarity:.1%}"
            else:
                # NOT A MATCH - Gray box (less prominent)
                color = (100, 100, 100)  # Gray
                thickness = 1
                label = f"Face: {confidence:.0%}"
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result,
                         (x1, y2 - label_size[1] - 5),
                         (x1 + label_size[0], y2),
                         color, cv2.FILLED)
            
            cv2.putText(result, label, (x1, y2 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add timestamp and camera info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(result, f"DRISTI AI | {timestamp}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return result
    
    def save_matches(self, matches, camera_id, video_path, output_path):
        """Save match information to file"""
        if not matches:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"output/{camera_id}_matches_{timestamp}.json"
        
        summary = {
            'search_timestamp': datetime.now().isoformat(),
            'target_person': self.target_name,
            'camera_id': camera_id,
            'video_source': video_path,
            'output_video': output_path,
            'similarity_threshold': self.similarity_threshold,
            'matches_found': len(matches),
            'match_details': matches,
            'best_match': {
                'similarity': max(matches, key=lambda x: x['similarity'])['similarity'],
                'frame': max(matches, key=lambda x: x['similarity'])['frame'],
                'time': max(matches, key=lambda x: x['similarity'])['time_seconds']
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ Match details saved: {summary_file}")
        
        # Also save text summary
        text_file = f"output/{camera_id}_summary_{timestamp}.txt"
        with open(text_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("DRISTI - FACE MATCH SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Target Person: {self.target_name}\n")
            f.write(f"Camera: {camera_id}\n")
            f.write(f"Video: {os.path.basename(video_path)}\n")
            f.write(f"Search Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("="*60 + "\n")
            f.write("MATCHES FOUND:\n")
            f.write("="*60 + "\n")
            
            for i, match in enumerate(matches, 1):
                f.write(f"\nMatch #{i}:\n")
                f.write(f"  Frame: {match['frame']}\n")
                f.write(f"  Time: {match['time_seconds']:.1f} seconds\n")
                f.write(f"  Similarity: {match['similarity']:.1%}\n")
                f.write(f"  Location: {match['bbox']}\n")
            
            f.write(f"\n" + "="*60 + "\n")
            f.write(f"Total Matches: {len(matches)}\n")
            f.write(f"Best Match: {summary['best_match']['similarity']:.1%} ")
            f.write(f"at {summary['best_match']['time']:.1f}s\n")
            f.write("="*60 + "\n")
        
        print(f"  ✓ Text summary saved: {text_file}")
    
    def search_multiple_cameras(self, target_photo, cctv_videos):
        """Search across multiple CCTV cameras"""
        print("\n" + "="*70)
        print("MULTI-CAMERA SEARCH INITIATED")
        print("="*70)
        
        # Set target person
        target_name = input("Enter target person name: ").strip()
        if not target_name:
            target_name = "Missing Person"
        
        if not self.set_target_person(target_photo, target_name):
            return
        
        # Search in each CCTV
        all_results = []
        
        for i, (camera_id, video_path) in enumerate(cctv_videos.items(), 1):
            print(f"\n{'='*60}")
            print(f"CAMERA {i}/{len(cctv_videos)}: {camera_id}")
            print(f"{'='*60}")
            
            if not os.path.exists(video_path):
                print(f"✗ Video not found: {video_path}")
                continue
            
            result = self.search_in_video(video_path, camera_id)
            
            if result:
                all_results.append(result)
                
                if result['matches_found'] > 0:
                    print(f"\n✅ PERSON FOUND IN {camera_id}!")
                    print(f"   Matches: {result['matches_found']}")
                    print(f"   Best match: {max(m['similarity'] for m in result['match_details']):.1%}")
                else:
                    print(f"\nℹ️  No matches in {camera_id}")
            
            print(f"{'='*60}")
        
        # Generate final report
        self.generate_final_report(all_results, target_name)
    
    def generate_final_report(self, results, target_name):
        """Generate final search report"""
        print("\n" + "="*70)
        print("SEARCH COMPLETE - FINAL REPORT")
        print("="*70)
        
        total_matches = sum(r['matches_found'] for r in results)
        cameras_with_matches = [r for r in results if r['matches_found'] > 0]
        
        print(f"\nTarget: {target_name}")
        print(f"Cameras searched: {len(results)}")
        print(f"Cameras with matches: {len(cameras_with_matches)}")
        print(f"Total matches found: {total_matches}")
        
        if cameras_with_matches:
            print("\n📍 LOCATIONS WHERE PERSON WAS FOUND:")
            print("-" * 50)
            
            for result in cameras_with_matches:
                best_match = max(result['match_details'], key=lambda x: x['similarity'])
                print(f"\n📹 {result['camera_id']}:")
                print(f"   Video: {os.path.basename(result['video_path'])}")
                print(f"   Best match: {best_match['similarity']:.1%}")
                print(f"   Time: {best_match['time_seconds']:.1f} seconds")
                print(f"   Frame: {best_match['frame']}")
                print(f"   Output video: {os.path.basename(result['output_video'])}")
        else:
            print("\n⚠️  Person not found in any camera")
            print("   Suggestions:")
            print("   1. Lower similarity threshold (currently {self.similarity_threshold:.0%})")
            print("   2. Use clearer target photo")
            print("   3. Ensure person appears in videos")
        
        # Save final report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"output/final_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DRISTI - FINAL SEARCH REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Target Person: {target_name}\n")
            f.write(f"Search Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Similarity Threshold: {self.similarity_threshold:.0%}\n\n")
            f.write("="*70 + "\n")
            f.write("SEARCH RESULTS\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Cameras searched: {len(results)}\n")
            f.write(f"Cameras with matches: {len(cameras_with_matches)}\n")
            f.write(f"Total matches: {total_matches}\n\n")
            
            if cameras_with_matches:
                f.write("PERSON WAS FOUND IN:\n")
                f.write("-" * 50 + "\n")
                for result in cameras_with_matches:
                    best = max(result['match_details'], key=lambda x: x['similarity'])
                    f.write(f"\n• {result['camera_id']}\n")
                    f.write(f"  Best match: {best['similarity']:.1%}\n")
                    f.write(f"  Time: {best['time_seconds']:.1f}s (Frame: {best['frame']})\n")
                    f.write(f"  Video evidence: {os.path.basename(result['output_video'])}\n")
            else:
                f.write("PERSON NOT FOUND IN ANY CAMERA\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("EVIDENCE FILES\n")
            f.write("="*70 + "\n\n")
            
            for result in results:
                if result['matches_found'] > 0:
                    f.write(f"• {result['camera_id']}:\n")
                    f.write(f"  - Video: {os.path.basename(result['output_video'])}\n")
                    f.write(f"  - Matches: {result['matches_found']}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        print(f"\n📄 Final report saved: {report_file}")
        print("="*70)

def main():
    """Main function"""
    print("\n" + "="*70)
    print("DRISTI - ACCURATE MULTI-CAMERA FACE SEARCH")
    print("Find missing persons across multiple CCTV cameras")
    print("="*70)
    
    recognizer = AccurateFaceRecognizer()
    
    # Define CCTV videos (modify these paths)
    cctv_videos = {
        "CCTV_1_Mall_Entrance": "input/cctv1.mp4",
        "CCTV_2_Mall_Corridor": "input/cctv2.mp4", 
        "CCTV_3_Parking_Lot": "input/cctv3.mp4",
        "CCTV_4_Exit_Gate": "input/cctv4.mp4"
    }
    
    # Check which videos exist
    available_videos = {}
    for cam_id, path in cctv_videos.items():
        if os.path.exists(path):
            available_videos[cam_id] = path
    
    if not available_videos:
        print("\n⚠️  No CCTV videos found in 'input/' folder")
        print("   Please add videos with names:")
        print("   - cctv1.mp4, cctv2.mp4, cctv3.mp4, cctv4.mp4")
        return
    
    print(f"\n📹 Available CCTV cameras: {len(available_videos)}")
    for cam_id in available_videos:
        print(f"   • {cam_id}")
    
    # Get target photo
    target_photo = "input/missing_photo.png"
    if not os.path.exists(target_photo):
        print(f"\n⚠️  Target photo not found: {target_photo}")
        print("   Please place target photo as 'input/missing_photo.png'")
        return
    
    print(f"\n🎯 Target photo: {target_photo}")
    
    # Start search
    recognizer.search_multiple_cameras(target_photo, available_videos)

if __name__ == "__main__":
    main()