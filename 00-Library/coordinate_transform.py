import cv2
import numpy as np


def detect_tennis_court(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Supposer que le terrain a une couleur distincte
    lower_bound = np.array([x, y, z])  # à ajuster
    upper_bound = np.array([a, b, c])  # à ajuster

    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Amélioration du masque
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Détecter les contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    cv2.imshow('Tennis Court Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Test de la fonction
detect_tennis_court('../01-Images/P1140706.jpg')
