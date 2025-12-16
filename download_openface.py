import urllib.request
import os

# Create directory
os.makedirs('openface_model', exist_ok=True)

print("Downloading OpenFace model...")
url = 'https://github.com/iwantooxxoox/Keras-OpenFace/releases/download/v1.0/nn4.small2.v1.h5'
filename = 'openface_model/nn4.small2.v1.h5'

# Download
urllib.request.urlretrieve(url, filename)
print(f"✓ OpenFace model downloaded: {filename}")
print(f"Size: {os.path.getsize(filename) / (1024*1024):.1f} MB")