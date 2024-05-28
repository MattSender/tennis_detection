import os
import cv2

def enhance_and_convert_to_black_and_white(image_path):
    # Charger l'image
    image = cv2.imread(image_path)
    # Convertir en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Appliquer l'équilibrage des histogrammes pour améliorer le contraste
    enhanced_image = cv2.equalizeHist(gray_image)
    return enhanced_image

def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(directory, filename)
            enhanced_image = enhance_and_convert_to_black_and_white(file_path)
            cv2.imwrite(file_path, enhanced_image)

train_dir = "images/train_data"
test_dir = "images/test_data"

# Traiter les deux répertoires
process_directory(train_dir)
process_directory(test_dir)

print("Toutes les images ont été converties et remplacées dans les dossiers spécifiés.")
