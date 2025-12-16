import gdown
import os

# Create facenet_model directory
os.makedirs('facenet_model', exist_ok=True)

print("Downloading FaceNet model...")
url = 'https://drive.google.com/uc?id=1PZ_6Zsy1Vb0s0JmjEmVd8FS99zoMCiN1'
output = 'facenet_model/facenet_keras.h5'
gdown.download(url, output, quiet=False)

print("FaceNet model downloaded successfully!")
print("Location: facenet_model/facenet_keras.h5")