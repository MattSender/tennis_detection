import cv2
import numpy as np

def detect_tennis_balls(image_path, scale_percent=25):
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print("Erreur lors de la lecture de l'image.")
        return

    # Convertir en espace de couleur HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Définir les plages de couleur pour le jaune fluorescent
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Créer un masque pour les zones jaunes
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Opérations morphologiques pour éliminer le bruit
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Trouver les cercles via Hough Circles
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 50,
                               param1=50, param2=15, minRadius=10, maxRadius=40)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Dessiner les cercles et les centres
            cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Redimensionner l'image pour la visualiser entièrement
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    # Afficher l'image redimensionnée
    cv2.imshow('Detected Tennis Balls', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionnellement, retourner les cercles détectés
    return circles
# Test du script avec une image
detect_tennis_balls('../01-Images/P1140706.jpg', scale_percent=25)
