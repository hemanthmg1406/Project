# Install necessary libraries
!pip install tensorflow keras opencv-python matplotlib numpy Pillow gdown

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define dataset path
dataset_path = '/content/drive/MyDrive/dataset'

# Use gdown to download your dataset folder from Google Drive (if necessary)
!pip install gdown  # Ensure gdown is installed
!gdown --folder https://drive.google.com/drive/folders/1u8JADgRLtUqjz6nxa4qk-Iz2-mdrQt72  

# List the files and directories in the dataset
import os

# Check the dataset contents
for root, dirs, files in os.walk(dataset_path):
    print(f"Root: {root}")
    print(f"Directories: {dirs}")
    print(f"Files: {files}")
    print('------------------')
