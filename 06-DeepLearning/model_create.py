import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import ast

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

# Charger les annotations
annotations = pd.read_csv('annotations.csv')

# Fonction pour convertir une chaîne de caractères en tuple de coordonnées
def parse_point(point_str):
    try:
        return ast.literal_eval(point_str)
    except (ValueError, SyntaxError):
        return (-1, -1)

# Charger les images et les annotations
def load_data(annotations, img_dir, img_size=(1024, 768)):
    images = []
    points = []
    missing_files = 0
    for index, row in annotations.iterrows():
        img_path = os.path.join(img_dir, row["filename"])
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                enhanced_img = enhance_lines(img)
                images.append(enhanced_img)
                topleft = parse_point(row['topleft'])
                topright = parse_point(row['topright'])
                bottomright = parse_point(row['bottomright'])
                bottomleft = parse_point(row['bottomleft'])
                points.append([topleft[0]/img_size[0], topleft[1]/img_size[1],
                               topright[0]/img_size[0], topright[1]/img_size[1],
                               bottomright[0]/img_size[0], bottomright[1]/img_size[1],
                               bottomleft[0]/img_size[0], bottomleft[1]/img_size[1]])
            else:
                print(f"Warning: Unable to read image file {img_path}")
        else:
            print(f"Warning: Image file {img_path} does not exist")
            missing_files += 1

    if missing_files > 0:
        print(f"Total missing files: {missing_files}")
    return np.expand_dims(np.array(images), axis=-1), np.array(points)

# Prétraiter les données
img_size = (1024, 768)
img_dir = 'vierge/'
X, y = load_data(annotations, img_dir, img_size)
if X.size == 0 or y.size == 0:
    raise ValueError("No images loaded. Please check the image paths and ensure the images exist.")

X = X / 255.0  # Normalisation

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Diviser les données en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Définir le modèle
model = Sequential([
    tf.keras.Input(shape=(img_size[1], img_size[0], 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(8)  # 4 points * 2 coordonnées (x, y)
])

# Compiler le modèle
model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')

# Entraîner le modèle avec augmentation des données
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=3,
                    validation_data=(X_val, y_val))

# Sauvegarder le modèle
model.save('tennis_court_detector.h5')

print("Modèle entraîné et sauvegardé avec succès.")
