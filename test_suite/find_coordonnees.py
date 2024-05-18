import cv2
import numpy as np
import csv

# Variables globales pour stocker les points et l'image
points = []
image = None
filename = 'img-1.png'

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        # Ajouter le point aux coordonnées x, y
        points.append((x, y))
        # Dessiner un petit cercle pour visualiser le point
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", image)
        if len(points) == 4:
            draw_lines()

def draw_lines():
    global points
    # Relier les points par des lignes
    cv2.line(image, points[0], points[1], (255, 0, 0), 2)
    cv2.line(image, points[1], points[2], (255, 0, 0), 2)
    cv2.line(image, points[2], points[3], (255, 0, 0), 2)
    cv2.line(image, points[3], points[0], (255, 0, 0), 2)
    cv2.imshow("Image", image)
    save_coordinates()

def save_coordinates():
    global points, filename
    # Enregistrer les coordonnées dans un fichier CSV
    with open('coordonnees.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "topleft", "topright", "bottomright", "bottomleft"])
        writer.writerow([filename, points[0], points[1], points[2], points[3]])
    print("Coordonnées enregistrées dans coordonnees.csv")

def main():
    global image, filename
    # Charger l'image
    image = cv2.imread(f'img/{filename}')
    if image is None:
        print("Erreur: l'image n'a pas pu être chargée.")
        return
    # Redimensionner l'image à 1024x768 si nécessaire
    image = cv2.resize(image, (1024, 768))
    # Afficher l'image
    cv2.imshow("Image", image)
    # Définir la fonction de rappel pour la souris
    cv2.setMouseCallback("Image", mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
