from PIL import Image
import numpy as np

# Load the heart image
heart_image_path = 'images/heart2.webp'
heart_image = Image.open(heart_image_path)

# Convert the image to grayscale and then to a numpy array
heart_array = np.array(heart_image.convert('L'))

# To identify the exclusion zone, we find the non-transparent part of the image
# (assuming the heart is not transparent and the background is)
threshold = 128  # Threshold for binarization, may need adjustment
mask = heart_array > threshold

# Find the center of the non-transparent part (which is the heart)
y_indices, x_indices = np.where(mask)
center_x = x_indices.mean()
center_y = y_indices.mean()

# Estimate the width and height of the heart by finding the max distance from the center to non-transparent pixels
width = (x_indices.max() - x_indices.min()) * 0.5  # 50% of total width
height = (y_indices.max() - y_indices.min()) * 0.5  # 50% of total height

center_x, center_y, width, height
print(center_x, center_y, width, height)