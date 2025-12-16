import cv2
import numpy as np
import json
import time
import sqlite3
from datetime import datetime
import os
from utils.facenet_wrapper import FaceNetRecognizer

class DristiFaceNetSystem:
    """Dristi system using FaceNet instead of face-recognition library"""
    
    def __init__(self):
        print("="*60)
        print("DRISTI - FaceNet Implementation")
        print("="*60)
        
        # Initialize FaceNet (handle model load failures gracefully)
        try:
           self.face_net = FaceNetRecognizer('openface_model/nn4.small2.v1.h5')
        except Exception as e:
            print(f"Error initializing FaceNet model: {e}")
            print("Face recognition will be disabled until you provide a compatible model in 'facenet_model/facenet_keras.h5'.")
            self.face_net = None
        
        # Database for storing known faces
        self.known_faces = {}  # person_id -> embedding
        self.face_metadata = {}  # person_id -> metadata
        
        # Create necessary directories
        self.create_directories()
        
        # Initialize database
        self.init_database()
        
        print("System initialized with FaceNet")
    
    def create_directories(self):
        """Create required directories"""
        dirs = ['detections', 'uploads', 'snapshots', 'logs', 'reference_faces']
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)
    
    def init_database(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect('dristi_facenet.db')
        cursor = self.conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS missing_persons (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                case_number TEXT UNIQUE,
                photo_path TEXT,
                embedding BLOB,
                created_at TIMESTAMP,
                status TEXT DEFAULT 'active'
            )
        ''')
        
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
        print("Database initialized")
    
    def add_missing_person(self, name, age, gender, case_number, photo_path):
        """Add a missing person to the system"""
        try:
            # Extract face embedding
            if not self.face_net:
                print("ERROR: FaceNet model not loaded; cannot extract embeddings.")
                return False

            embeddings = self.face_net.get_embeddings_from_image(photo_path)
            
            if len(embeddings) == 0:
                print(f"No face found in {photo_path}")
                return False
            
            # Use the first face found
            embedding = np.asarray(embeddings[0]['embedding'], dtype=np.float32)
            
            # Generate person ID
            person_id = f"MP_{case_number}"
            
            # Store in memory
            self.known_faces[person_id] = embedding
            self.face_metadata[person_id] = {
                'name': name,
                'age': age,
                'gender': gender,
                'case_number': case_number
            }
            
            # Store in database
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO missing_persons 
                (id, name, age, gender, case_number, photo_path, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                person_id,
                name,
                age,
                gender,
                case_number,
                photo_path,
                embedding.tobytes(),  # Convert numpy array to bytes
                datetime.now().isoformat()
            ))
            
            self.conn.commit()
            
            print(f"Added missing person: {name} (Case: {case_number})")
            print(f"  Embedding size: {len(embedding)} dimensions")
            
            return True
            
        except Exception as e:
            print(f"Error adding missing person: {e}")
            return False
    
    def process_camera(self, camera_id=0, duration=60):
        """Process camera feed"""
        print(f"\nStarting camera feed (ID: {camera_id})...")
        print("Press 'q' to quit, 's' to save snapshot")
        
        if not self.face_net:
            print("ERROR: FaceNet model not loaded; cannot start camera monitoring.")
            return

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Cannot open camera {camera_id}")
            return
        
        start_time = time.time()
        frame_count = 0
        
        while True:
            # Check duration
            if time.time() - start_time > duration:
                print(f"\nDuration limit reached ({duration} seconds)")
                break
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            frame_count += 1
            
            # Skip frames for performance (process every 3rd frame)
            if frame_count % 3 != 0:
                continue
            
            # Detect faces
            faces = self.face_net.detect_faces(frame)
            
            # Process each face
            for face_data in faces:
                x1, y1, x2, y2 = face_data['bbox']
                face_img = face_data['face']
                
                # Get embedding (ensure float32 for DB storage)
                embedding = self.face_net.get_embedding(face_img)
                embedding = np.asarray(embedding, dtype=np.float32)
                
                # Find best match
                best_match, similarity, distance = self.face_net.find_best_match(
                    embedding, self.known_faces
                )
                
                # Draw bounding box
                if best_match and similarity > 0.7:  # High confidence
                    # Known person - Green box
                    color = (0, 255, 0)
                    thickness = 3
                    
                    person_name = self.face_metadata[best_match]['name']
                    label = f"{person_name} ({similarity:.2f})"
                    
                    # Draw "FOUND" text
                    cv2.putText(frame, "FOUND", (x1, y1 - 40),
                               cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
                    
                    # Save detection (pass full frame so we can save evidence)
                    self.save_detection(best_match, similarity, (x1, y1, x2, y2), frame)
                    
                else:
                    # Unknown person - Red box
                    color = (0, 0, 255)
                    thickness = 2
                    label = "Unknown"
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label
                cv2.rectangle(frame, (x1, y2 - 30), (x2, y2), color, cv2.FILLED)
                cv2.putText(frame, label, (x1 + 5, y2 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('DRISTI - FaceNet Detection', frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nStopped by user")
                break
            elif key == ord('s'):
                # Save snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"snapshots/snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Snapshot saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nSummary:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Known faces in database: {len(self.known_faces)}")
    
    def save_detection(self, person_id, similarity, bbox, frame=None):
        """Save detection to database and optionally save frame image"""
        try:
            # Use filesystem-safe timestamp for filenames
            timestamp = datetime.now().isoformat()
            filename_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save frame as evidence if provided
            frame_path = f"detections/{person_id}_{filename_ts}.jpg"
            if frame is not None:
                try:
                    cv2.imwrite(frame_path, frame)
                except Exception as e:
                    print(f"Warning: failed to write frame to {frame_path}: {e}")
                    frame_path = None

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
            
            # Send alert
            self.send_alert(person_id, similarity)
            
        except Exception as e:
            print(f"Error saving detection: {e}")    
    def send_alert(self, person_id, similarity):
        """Send alert when person is found"""
        person = self.face_metadata.get(person_id)
        if person:
            print("\n" + "="*60)
            print(f"ALERT: {person['name']} FOUND!")
            print(f"   Case: {person['case_number']}")
            print(f"   Confidence: {similarity:.1%}")
            print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
            print("="*60)
            
            # Here you would add:
            # 1. SMS/Email notification
            # 2. Webhook to police dashboard
            # 3. Save to alert log
    
    def load_from_database(self):
        """Load known faces from database"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT id, name, embedding FROM missing_persons WHERE status="active"')
        
        rows = cursor.fetchall()
        
        for row in rows:
            person_id, name, embedding_bytes = row
            
            # Convert bytes back to numpy array
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            self.known_faces[person_id] = embedding
            self.face_metadata[person_id] = {'name': name}
        
        print(f"Loaded {len(self.known_faces)} faces from database")

def main():
    """Main function"""
    system = DristiFaceNetSystem()
    
    # Menu
    while True:
        print("\n" + "="*60)
        print("DRISTI - MAIN MENU")
        print("="*60)
        print("1. Add Missing Person")
        print("2. Start CCTV Monitoring")
        print("3. View Database")
        print("4. Exit")
        print("="*60)
        
        choice = input("Select option: ").strip()
        
        if choice == "1":
            print("\n📝 Add Missing Person")
            name = input("Name: ")
            age = input("Age: ")
            gender = input("Gender: ")
            case_number = input("Case Number: ")
            photo_path = input("Photo Path: ")
            
            if os.path.exists(photo_path):
                system.add_missing_person(name, age, gender, case_number, photo_path)
            else:
                print(f"File not found: {photo_path}")
        
        elif choice == "2":
            print("\n📹 Starting CCTV Monitoring")
            duration = input("Duration in seconds (default 60): ").strip()
            duration = int(duration) if duration.isdigit() else 60
            
            system.process_camera(duration=duration)
        
        elif choice == "3":
            print("\nDatabase Summary")
            print(f"Known faces: {len(system.known_faces)}")
            for person_id, metadata in system.face_metadata.items():
                print(f"  - {metadata['name']} ({person_id})")
        
        elif choice == "4":
            print("\nThank you for using DRISTI!")
            break
        
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()