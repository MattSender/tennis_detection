import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = 'images/img23.jpg'
image = cv2.imread(image_path)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

s = cv2.add(s, 50)
v = cv2.add(v, 50)

hsv_enhanced = cv2.merge([h, s, v])
vibrant_image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

gray = cv2.cvtColor(vibrant_image, cv2.COLOR_BGR2GRAY)

_, darkened = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Final Enhanced Lines Image')
plt.imshow(darkened, cmap='gray')
plt.axis('off')

plt.show()
