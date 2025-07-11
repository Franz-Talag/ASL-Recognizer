# create_background.py
import cv2
import numpy as np

# Define the size of the image
# This should be large enough to fit the cropped hand
width, height = 400, 400

# Create a black image (all zeros) and then add 255 to make it white
# The dtype=np.uint8 is important for image formats
white_background = np.zeros((height, width, 3), dtype=np.uint8) + 255

# Save the image to a file
cv2.imwrite("background.jpg", white_background)

print("Successfully created background.jpg!")
