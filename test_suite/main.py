import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the coordinates from the CSV file
coords_df = pd.read_csv('coordonnees.csv')
coords = coords_df.loc[0, ['topleft', 'topright', 'bottomright', 'bottomleft']].values
coords = [eval(coord) for coord in coords]

# Load the image
image_path = 'img/img-1.png'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Extract the region of interest (ROI) using the coordinates
pts = np.array(coords, dtype=np.int32)
rect = cv2.boundingRect(pts)  # Get the bounding box of the ROI
x, y, w, h = rect
roi = image_rgb[y:y+h, x:x+w].copy()

# Create a mask and extract the ROI using the mask
pts = pts - pts.min(axis=0)  # Shift the points to the ROI's coordinate system
mask = np.zeros(roi.shape[:2], dtype=np.uint8)
cv2.drawContours(mask, [pts], -1, (255, 255, 255), thickness=cv2.FILLED)
roi = cv2.bitwise_and(roi, roi, mask=mask)

# Convert the ROI to HSV color space
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

# Define the color range for detecting tennis balls (yellow-green)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

# Create a mask for the color range
color_mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)

# Find contours in the mask
contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw blue circles around the detected tennis balls
for contour in contours:
    ((x_c, y_c), radius) = cv2.minEnclosingCircle(contour)
    if 10 < radius < 30:  # Consider only reasonable sizes for tennis balls
        center = (int(x_c), int(y_c))
        radius = int(radius)
        cv2.circle(roi, center, radius, (0, 0, 255), 4)  # Blue color in BGR

# Place the processed ROI back into the original image
result_image = image_rgb.copy()
result_image[y:y+h, x:x+w] = roi

# Display the final image
plt.figure(figsize=(10, 10))
plt.imshow(result_image)
plt.axis('off')
plt.show()

# Save the final image
result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('detected_balls.png', result_image_bgr)
