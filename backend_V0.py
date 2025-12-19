import cv2
import numpy as np
import os
import sys
from datetime import datetime

# Add utils to path
sys.path.append('utils')
from simple_detector import SimpleFaceFinder

class VideoPersonFinder:
    """Find a person in video using reference photo"""
    
    def __init__(self):
        print("="*70)
        print("DRISTI - Video Person Finder")
        print("="*70)
        print("Find missing persons in CCTV footage")
        print("="*70)
        
        # Initialize detector
        self.detector = SimpleFaceFinder()
        
        # Create output directory
        os.makedirs('output', exist_ok=True)
        os.makedirs('input', exist_ok=True)
        
        print("✓ System ready")
    
    def find_in_video(self, photo_path, video_path, output_path=None, match_threshold=65):
        """
        Find person in video
        
        Args:
            photo_path: Path to missing person photo
            video_path: Path to CCTV/crowd video
            output_path: Path to save output video
            match_threshold: Percentage threshold for match (default: 65%)
        """
        print("\n" + "="*70)
        print("STARTING PERSON SEARCH")
        print("="*70)
        
        # Step 1: Extract face from reference photo
        print("\n[1/4] Processing reference photo...")
        reference_data = self.detector.extract_face_from_photo(photo_path)
        
        if reference_data is None:
            print("✗ Failed to extract face from reference photo")
            return False
        
        reference_signature = reference_data['signature']
        print(f"✓ Reference face extracted")
        print(f"  Signature size: {len(reference_signature)}")
        
        # Step 2: Open video file
        print("\n[2/4] Opening video file...")
        if not os.path.exists(video_path):
            print(f"✗ Video file not found: {video_path}")
            return False
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"✗ Cannot open video: {video_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"✓ Video loaded successfully")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.1f} seconds")
        
        # Step 3: Prepare output video
        print("\n[3/4] Preparing output video...")
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output/detected_{timestamp}.mp4"
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"  Output will be saved to: {output_path}")
        print(f"  Match threshold: {match_threshold}%")
        
        # Step 4: Process video frame by frame
        print("\n[4/4] Processing video frames...")
        print("  Press 'q' to quit early")
        print("-" * 40)
        
        frame_count = 0
        matches_found = 0
        total_faces = 0
        
        # For displaying progress
        frames_to_skip = max(1, fps // 10)  # Process ~10 FPS for speed
        print(f"  Processing 1 of every {frames_to_skip} frames for speed")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            frame_count += 1
            
            # Skip frames for faster processing
            if frame_count % frames_to_skip != 0:
                continue
            
            # Process frame
            matches = self.detector.process_video_frame(
                frame, 
                reference_signature, 
                threshold=match_threshold
            )
            
            total_faces += len(matches)
            
            # Draw results on frame
            processed_frame = self.draw_results(frame, matches, match_threshold)
            
            # Check for matches
            for match in matches:
                if match['is_match']:
                    matches_found += 1
                    
                    # Print match info (only first few times to avoid spam)
                    if matches_found <= 3:
                        print(f"  ✓ Match found at frame {frame_count}!")
                        print(f"    Confidence: {match['match_percentage']:.1f}%")
            
            # Write frame to output video
            out.write(processed_frame)
            
            # Display progress
            if frame_count % (fps * 5) == 0:  # Every 5 seconds
                progress = (frame_count / total_frames) * 100
                print(f"  Progress: {progress:.1f}% | Frames: {frame_count}/{total_frames}")
            
            # Show preview window
            if frame_count % (fps // 2) == 0:  # Update preview 2 times per second
                preview = cv2.resize(processed_frame, (800, 600))
                cv2.imshow('DRISTI - Live Detection Preview', preview)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n🛑 Processing stopped by user")
                    break
        
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Step 5: Generate summary
        print("\n" + "="*70)
        print("SEARCH COMPLETE - SUMMARY")
        print("="*70)
        print(f"Reference Photo: {os.path.basename(photo_path)}")
        print(f"Search Video: {os.path.basename(video_path)}")
        print(f"Total frames processed: {frame_count}")
        print(f"Total faces detected: {total_faces}")
        print(f"MATCHES FOUND: {matches_found}")
        print(f"Output saved to: {output_path}")
        
        if matches_found > 0:
            print("\n✅ SUCCESS: Person found in video!")
            print("   Green boxes indicate matches")
            print("   Percentage shows match confidence")
        else:
            print("\n⚠️  No matches found with current threshold")
            print("   Try lowering the match threshold (currently {match_threshold}%)")
        
        print("="*70)
        
        # Save summary to file
        self.save_summary(photo_path, video_path, output_path, 
                         frame_count, total_faces, matches_found, match_threshold)
        
        return True
    
    def draw_results(self, frame, matches, threshold):
        """Draw detection results on frame - FIXED VERSION"""
        result_frame = frame.copy()
        frame_height, frame_width = result_frame.shape[:2]  # ADD THIS LINE
        
        for match in matches:
            x1, y1, x2, y2 = match['bbox']
            percentage = match['match_percentage']
            is_match = match['is_match']
            
            # Choose color based on match
            if is_match:
                # MATCH - Green box
                color = (0, 255, 0)  # Green
                thickness = 3
                label = f"MATCH: {percentage:.1f}%"
                
                # Add "FOUND" text
                cv2.putText(result_frame, "FOUND", (x1, y1 - 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
            else:
                # NO MATCH - Red box
                color = (0, 0, 255)  # Red
                thickness = 2
                label = f"NO MATCH: {percentage:.1f}%"
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_frame,
                        (x1, y2 - label_size[1] - 10),
                        (x1 + label_size[0], y2),
                        color, cv2.FILLED)
            
            # Draw label
            cv2.putText(result_frame, label, (x1, y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(result_frame, timestamp, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add DRISTI logo - FIXED: Use frame_width instead of width
        cv2.putText(result_frame, "DRISTI AI", (frame_width - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return result_frame
    
    def save_summary(self, photo_path, video_path, output_path, 
                    frames, faces, matches, threshold):
        """Save search summary to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"output/search_summary_{timestamp}.txt"
        
        summary = f"""DRISTI - Search Summary
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

SEARCH PARAMETERS:
- Reference Photo: {photo_path}
- Search Video: {video_path}
- Match Threshold: {threshold}%
- Output Video: {output_path}

RESULTS:
- Frames Processed: {frames}
- Faces Detected: {faces}
- Matches Found: {matches}
- Success Rate: {(matches/max(faces,1))*100:.1f}%

STATUS: {'PERSON FOUND' if matches > 0 else 'NO MATCH FOUND'}

Notes:
- Green boxes indicate matches above threshold
- Red boxes indicate faces below threshold
- Match percentage shows confidence level
"""
        
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(f"✓ Summary saved to: {summary_file}")
    
    def run_demo(self):
        """Run a demo with sample files"""
        print("\n" + "="*70)
        print("DEMO MODE")
        print("="*70)
        
        # Check for sample files
        sample_photo = "input/missing_photo.jpg"
        sample_video = "input/crowd_video.mp4"
        
        if not os.path.exists(sample_photo):
            print(f"✗ Sample photo not found: {sample_photo}")
            print("  Please place a photo in 'input/missing_photo.jpg'")
            return False
        
        if not os.path.exists(sample_video):
            print(f"✗ Sample video not found: {sample_video}")
            print("  Please place a video in 'input/crowd_video.mp4'")
            print("  Or use any MP4/AVI video file")
            return False
        
        print("Found sample files:")
        print(f"  Photo: {sample_photo}")
        print(f"  Video: {sample_video}")
        
        # Start search
        self.find_in_video(
            photo_path=sample_photo,
            video_path=sample_video,
            match_threshold=60  # 60% match threshold
        )
    
    def run_custom(self):
        """Run with custom file paths"""
        print("\n" + "="*70)
        print("CUSTOM SEARCH")
        print("="*70)
        
        # Get file paths
        photo_path = input("Enter missing person photo path: ").strip()
        if not photo_path:
            photo_path = "input/missing_photo.jpg"
        
        video_path = input("Enter CCTV video path: ").strip()
        if not video_path:
            video_path = "input/crowd_video.mp4"
        
        threshold = input("Match threshold (60-80%, default 65): ").strip()
        if threshold and threshold.isdigit():
            threshold = int(threshold)
        else:
            threshold = 65
        
        # Start search
        self.find_in_video(
            photo_path=photo_path,
            video_path=video_path,
            match_threshold=threshold
        )

def main():
    """Main function"""
    print("\n" + "="*70)
    print("DRISTI - Video Person Finder")
    print("Find missing persons in CCTV footage")
    print("="*70)
    
    finder = VideoPersonFinder()
    
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Run Demo (use files in 'input/' folder)")
        print("2. Custom Search (choose your own files)")
        print("3. Instructions")
        print("4. Exit")
        print("="*60)
        
        choice = input("Select option (1-4): ").strip()
        
        if choice == "1":
            finder.run_demo()
        
        elif choice == "2":
            finder.run_custom()
        
        elif choice == "3":
            print("\n" + "="*60)
            print("INSTRUCTIONS")
            print("="*60)
            print("1. Prepare your files:")
            print("   - Missing person photo (clear face, good lighting)")
            print("   - CCTV/crowd video (MP4/AVI format)")
            print("2. Place files in 'input/' folder or use full paths")
            print("3. Choose match threshold (60-80% recommended)")
            print("4. System will process and save output video")
            print("5. Output includes:")
            print("   - Video with green boxes around matches")
            print("   - Match percentage display")
            print("   - Search summary report")
            print("="*60)
        
        elif choice == "4":
            print("\nThank you for using DRISTI!")
            print("Helping find missing persons with AI")
            break
        
        else:
            print("Invalid option")

if __name__ == "__main__":
    main()