import cv2
import numpy as np
import json
import time
import sqlite3
from datetime import datetime
import os
import sys

# Add utils to path
sys.path.append('utils')

from mediapipe_wrapper import MediaPipeFaceSystem

class DristiMediaPipeSystem:
    """Dristi system using MediaPipe for face detection"""
    
    def __init__(self, config_file='config.json'):
        print("="*60)
        print("DRISTI - MediaPipe Implementation")
        print("="*60)
        
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Initialize MediaPipe system
        self.face_system = MediaPipeFaceSystem()
        
        # Create necessary directories
        self.create_directories()
        
        # Initialize database
        self.db_path = self.config.get('database', {}).get('path', 'dristi_mediapipe.db')
        self.init_database()
        
        # Load existing faces
        self.face_system.load_from_file('face_database.pkl')
        
        print("✓ System initialized with MediaPipe")
    
    def load_config(self, config_file):
        """Load configuration"""
        default_config = {
            'database': {'path': 'dristi_mediapipe.db'},
            'recognition': {'threshold': 0.65}
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except:
                pass
        
        return default_config
    
    def create_directories(self):
        """Create required directories"""
        dirs = ['detections', 'uploads', 'snapshots', 'logs', 'reference_faces']
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)
    
    def init_database(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Missing persons table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS missing_persons (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                case_number TEXT UNIQUE,
                last_seen TEXT,
                contact TEXT,
                photo_path TEXT,
                added_at TIMESTAMP,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        # Detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                camera_id TEXT,
                person_id TEXT,
                similarity REAL,
                location TEXT,
                frame_path TEXT,
                alert_sent BOOLEAN DEFAULT 0
            )
        ''')
        
        self.conn.commit()
        print("✓ Database initialized")
    
    def add_missing_person(self, name, age, gender, case_number, last_seen, contact, photo_path):
        """Add a missing person to system"""
        try:
            if not os.path.exists(photo_path):
                print(f"✗ Photo not found: {photo_path}")
                return False
            
            # Read the photo
            image = cv2.imread(photo_path)
            if image is None:
                print(f"✗ Could not read image: {photo_path}")
                return False
            
            # Generate person ID
            person_id = f"MP_{case_number}"
            
            # Add to MediaPipe system
            success = self.face_system.add_known_face(person_id, name, image)
            
            if not success:
                return False
            
            # Save to database
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO missing_persons 
                (id, name, age, gender, case_number, last_seen, contact, photo_path, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                person_id,
                name,
                age,
                gender,
                case_number,
                last_seen,
                contact,
                photo_path,
                datetime.now().isoformat()
            ))
            
            self.conn.commit()
            
            # Save embeddings to file
            self.face_system.save_to_file('face_database.pkl')
            
            print(f"✓ Added missing person: {name}")
            print(f"  Case Number: {case_number}")
            print(f"  Contact: {contact}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error adding missing person: {e}")
            return False
    
    def process_camera(self, camera_id=0, duration=60):
        """Process camera feed for face detection"""
        print(f"\n📹 Starting Camera Feed (ID: {camera_id})")
        print("   Press 'q' to quit, 's' to save snapshot")
        print("-" * 40)
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"✗ Cannot open camera {camera_id}")
            print("   Trying camera ID 0...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("✗ No camera found")
                return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        start_time = time.time()
        frame_count = 0
        faces_detected = 0
        matches_found = 0
        
        print("🔍 Scanning for faces...")
        
        while True:
            # Check duration
            elapsed = time.time() - start_time
            if duration and elapsed > duration:
                print(f"\n⏰ Duration limit reached: {duration} seconds")
                break
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("✗ Failed to read frame")
                break
            
            frame_count += 1
            
            # Skip frames for performance (process every 2nd frame)
            if frame_count % 2 != 0:
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = self.face_system.detect_faces(frame)
            faces_detected += len(faces)
            
            # Process each face
            for face_data in faces:
                # Try to recognize
                person_id, similarity = self.face_system.recognize_face(
                    face_data['embedding'],
                    threshold=self.config['recognition']['threshold']
                )
                
                # Get person name if recognized
                recognized_name = None
                if person_id:
                    person_info = self.face_system.known_faces.get(person_id)
                    if person_info:
                        recognized_name = person_info['metadata']['name']
                        matches_found += 1
                        
                        # Save detection if high confidence
                        if similarity > 0.7:
                            self.save_detection(person_id, recognized_name, similarity, face_data['bbox'])
                
                # Draw on frame
                frame = self.face_system.draw_detection(frame, face_data, recognized_name, similarity)
            
            # Show FPS
            fps = frame_count / (elapsed + 0.0001)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show stats
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(frame, f"Known: {len(self.face_system.known_faces)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow('DRISTI - MediaPipe Face Detection', frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Stopped by user")
                break
            elif key == ord('s'):
                # Save snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"snapshots/snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Snapshot saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "="*40)
        print("SCAN SUMMARY")
        print("="*40)
        print(f"Frames processed: {frame_count}")
        print(f"Faces detected: {faces_detected}")
        print(f"Matches found: {matches_found}")
        print(f"Known faces in DB: {len(self.face_system.known_faces)}")
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print("="*40)
    
    def save_detection(self, person_id, person_name, similarity, bbox):
        """Save detection to database and send alert"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Save frame as evidence
            frame_path = f"detections/{person_id}_{timestamp.replace(':', '')}.jpg"
            
            # For now, we'll skip saving the actual frame
            # You can implement this by saving the current frame
            
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, camera_id, person_id, similarity, location, frame_path)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                'webcam_0',
                person_id,
                float(similarity),
                json.dumps(bbox),
                frame_path
            ))
            
            self.conn.commit()
            detection_id = cursor.lastrowid
            
            # Send alert
            self.send_alert(detection_id, person_id, person_name, similarity)
            
        except Exception as e:
            print(f"Error saving detection: {e}")
    
    def send_alert(self, detection_id, person_id, person_name, similarity):
        """Send alert when person is found"""
        print("\n" + "="*60)
        print("🚨 ALERT: MISSING PERSON FOUND!")
        print("="*60)
        print(f"Name: {person_name}")
        print(f"Person ID: {person_id}")
        print(f"Confidence: {similarity:.1%}")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        
        # Here you can add:
        # 1. SMS/Email notifications
        # 2. Webhook to police dashboard
        # 3. Save to alert log file
        # 4. Play sound alert
        
        # Save alert to file
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'detection_id': detection_id,
            'person_id': person_id,
            'person_name': person_name,
            'similarity': similarity,
            'alert_type': 'missing_person_found'
        }
        
        with open(f"logs/alert_{detection_id}.json", 'w') as f:
            json.dump(alert_data, f, indent=2)
    
    def view_database(self):
        """View all missing persons in database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, name, age, gender, case_number, last_seen, contact, added_at
            FROM missing_persons 
            WHERE status = 'active'
            ORDER BY added_at DESC
        ''')
        
        rows = cursor.fetchall()
        
        print("\n" + "="*70)
        print("MISSING PERSONS DATABASE")
        print("="*70)
        
        if len(rows) == 0:
            print("No missing persons in database")
        else:
            for row in rows:
                print(f"\nID: {row[0]}")
                print(f"Name: {row[1]}")
                print(f"Age: {row[2]} | Gender: {row[3]}")
                print(f"Case #: {row[4]}")
                print(f"Last Seen: {row[5]}")
                print(f"Contact: {row[6]}")
                print(f"Added: {row[7][:19]}")
                print("-" * 40)
        
        print(f"\nTotal: {len(rows)} missing persons")
        print("="*70)
    
    def run(self):
        """Main application loop"""
        while True:
            print("\n" + "="*60)
            print("DRISTI - MAIN MENU")
            print("="*60)
            print("1. Add Missing Person")
            print("2. Start CCTV Monitoring (Webcam)")
            print("3. View Database")
            print("4. Test with Photo")
            print("5. Exit")
            print("="*60)
            
            choice = input("Select option (1-5): ").strip()
            
            if choice == "1":
                self.add_person_menu()
            
            elif choice == "2":
                self.monitor_menu()
            
            elif choice == "3":
                self.view_database()
            
            elif choice == "4":
                self.test_with_photo()
            
            elif choice == "5":
                print("\nThank you for using DRISTI!")
                print("Remember: Every face matters.")
                break
            
            else:
                print("Invalid option. Please try again.")
    
    def add_person_menu(self):
        """Menu for adding missing person"""
        print("\n📝 ADD MISSING PERSON")
        print("-" * 40)
        
        name = input("Full Name: ").strip()
        age = input("Age: ").strip()
        gender = input("Gender (M/F/O): ").strip()
        case_number = input("Case Number: ").strip()
        last_seen = input("Last Seen Location/Time: ").strip()
        contact = input("Contact Number: ").strip()
        photo_path = input("Photo File Path: ").strip()
        
        if not photo_path:
            photo_path = "test_face.jpg"
        
        print("\nPlease wait, processing photo...")
        
        success = self.add_missing_person(
            name=name,
            age=age,
            gender=gender,
            case_number=case_number,
            last_seen=last_seen,
            contact=contact,
            photo_path=photo_path
        )
        
        if success:
            print(f"\n✅ {name} has been added to the search database.")
        else:
            print(f"\n❌ Failed to add {name}. Please check the photo.")
    
    def monitor_menu(self):
        """Menu for starting monitoring"""
        print("\n📹 CCTV MONITORING")
        print("-" * 40)
        
        camera_id = input("Camera ID (0 for default webcam): ").strip()
        if not camera_id.isdigit():
            camera_id = 0
        else:
            camera_id = int(camera_id)
        
        duration = input("Duration in seconds (0 for unlimited): ").strip()
        if duration.isdigit():
            duration = int(duration)
            if duration == 0:
                duration = None
        else:
            duration = 60
        
        print(f"\nStarting monitoring for {duration or 'unlimited'} seconds...")
        self.process_camera(camera_id=camera_id, duration=duration)
    
    def test_with_photo(self):
        """Test recognition with a photo"""
        print("\n🧪 TEST WITH PHOTO")
        print("-" * 40)
        
        photo_path = input("Enter photo path to test: ").strip()
        if not os.path.exists(photo_path):
            print(f"File not found: {photo_path}")
            return
        
        image = cv2.imread(photo_path)
        if image is None:
            print("Could not read image")
            return
        
        print("Processing image...")
        faces = self.face_system.detect_faces(image)
        
        print(f"\nFound {len(faces)} face(s) in the image")
        
        for i, face in enumerate(faces):
            person_id, similarity = self.face_system.recognize_face(
                face['embedding'],
                threshold=0.6
            )
            
            if person_id:
                person_info = self.face_system.known_faces[person_id]
                print(f"\nFace {i+1}: MATCH FOUND!")
                print(f"  Name: {person_info['metadata']['name']}")
                print(f"  Similarity: {similarity:.1%}")
                print(f"  Confidence: {'HIGH' if similarity > 0.7 else 'MEDIUM'}")
            else:
                print(f"\nFace {i+1}: Unknown person")
                print(f"  Similarity: {similarity:.1%}")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("DRISTI - Missing Person Detection System")
    print("Powered by Google MediaPipe")
    print("="*70)
    print("Team: TeesMaarKhaCoders")
    print("Mission: Finding missing persons with AI")
    print("="*70)
    
    try:
        system = DristiMediaPipeSystem()
        system.run()
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
