import os
import random
import shutil

# Source directory
source_dir = "C:/Users/ASUS/Desktop/psf-1.1/0.9"

# Destination directories
train_dir = "data/0.9/train"
test_dir = "data/0.9/test"

# Create destination directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get a list of all image files in the source directory
image_files = [file for file in os.listdir(source_dir) if file.endswith(".bmp")]

# Calculate the number of images for training and testing
num_train = int(len(image_files) * 0.7)
num_test = len(image_files) - num_train

# Randomly select images for training
train_images = random.sample(image_files, num_train)

# Copy training images to the train directory
for image in train_images:
    source_path = os.path.join(source_dir, image)
    dest_path = os.path.join(train_dir, image)
    shutil.copyfile(source_path, dest_path)

# Copy remaining images to the test directory
for image in image_files:
    if image not in train_images:
        source_path = os.path.join(source_dir, image)
        dest_path = os.path.join(test_dir, image)
        shutil.copyfile(source_path, dest_path)
