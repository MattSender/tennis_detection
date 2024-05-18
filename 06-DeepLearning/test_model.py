import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Charger le modèle
model = load_model('tennis_court_detector.h5')

# Fonction pour transformer l'image en noir et blanc avec lignes de terrain mises en valeur
def enhance_lines(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 50)
    v = cv2.add(v, 50)
    hsv_enhanced = cv2.merge([h, s, v])
    vibrant_image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(vibrant_image, cv2.COLOR_BGR2GRAY)
    _, darkened = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return darkened

# Fonction pour prédire les points sur une nouvelle image
def predict_points(img_path, img_size=(1024, 768)):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found at path: {img_path}")

    img_resized = cv2.resize(img, img_size)
    enhanced_img = enhance_lines(img_resized)
    img_normalized = enhanced_img / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=-1)  # Ajout de la dimension du canal
    img_expanded = np.expand_dims(img_expanded, axis=0)

    # Prédire les points
    points = model.predict(img_expanded)[0]

    # Convertir les points pour qu'ils soient relatifs à la taille originale de l'image
    h, w, _ = img.shape
    points = points.reshape((4, 2))
    points[:, 0] = points[:, 0] * w
    points[:, 1] = points[:, 1] * h

    return points

# Fonction pour dessiner les points sur l'image
def draw_points(img_path, points):
    img = cv2.imread(img_path)
    for (x, y) in points:
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
    cv2.imshow("Image with Points", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Exemple d'utilisation
img_path = 'vierge/img-16.jpg'
points = predict_points(img_path, img_size=(1024, 768))
print("Predicted Points:", points)
draw_points(img_path, points)
