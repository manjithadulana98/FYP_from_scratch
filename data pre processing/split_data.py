import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# Set the directory containing the images and masks
data_dir = 'C:/FYP/FYP_from_scratch/datasets'

# Set the size of the images
img_size = (512, 512)

# Initialize empty arrays to store images and masks
images = []
masks = []

# Loop through the directory and load images and masks
for filename in os.listdir(data_dir):
    if 'image' in filename:
        image_path = os.path.join(data_dir, filename)
        for sub_files in os.listdir(image_path):
            img = Image.open(os.path.join(image_path, sub_files))
            img = img.resize(img_size)
            img_array = np.array(img)
            images.append(img_array)
    elif 'mask' in filename:
        mask_path = os.path.join(data_dir, filename)
        for sub_files in os.listdir(mask_path):
            mask = Image.open(os.path.join(mask_path, sub_files))
            mask = mask.resize(img_size)
            mask_array = np.array(mask)
            masks.append(mask_array)

# Convert the lists of images and masks to numpy arrays
images = np.array(images)
masks = np.array(masks)

# Split the data into training and testing sets
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2, random_state=42)

# Save the training and testing sets
np.save('train_images.npy', train_images)
np.save('test_images.npy', test_images)
np.save('train_masks.npy', train_masks)
np.save('test_masks.npy', test_masks)