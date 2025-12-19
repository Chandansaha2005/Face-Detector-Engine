Face_Detection_Demo

Structure:

- video_finder.py          # MAIN FILE - Video person finder
- requirements.txt         # Simple requirements
- input/                   # Put your reference photo and sample video here
  - missing_photo.jpg
  - crowd_video.mp4
- output/                  # Script writes `detected_video.mp4` here
- utils/
  - simple_detector.py     # Simple face detector (haar + template match)

Quick start:
1. Create a virtual env and install dependencies:
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt

2. Replace the placeholder files in `input/` with a real `.jpg` reference photo and an `.mp4` video.

3. Run:
   python video_finder.py --photo input/missing_photo.jpg --video input/crowd_video.mp4 --output output/detected_video.mp4

Notes:
- This is a demo. For production use face embeddings or a deeper face matcher for reliability.
