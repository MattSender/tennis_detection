from PIL import Image
import os

# Chemin du dossier contenant les images
dossier = '../images/train_data'

# Taille souhaitée pour les images
nouvelle_largeur = 1024
nouvelle_hauteur = 768

# Parcourir tous les fichiers du dossier
for nom_fichier in os.listdir(dossier):
    # Vérifier si le fichier a l'extension .jpg
    if nom_fichier.lower().endswith('.jpg'):
        chemin_complet = os.path.join(dossier, nom_fichier)

        # Ouvrir l'image
        with Image.open(chemin_complet) as img:
            # Redimensionner l'image
            img_redimensionnee = img.resize((nouvelle_largeur, nouvelle_hauteur))

            # Convertir en mode RGB si nécessaire
            if img_redimensionnee.mode != 'RGB':
                img_redimensionnee = img_redimensionnee.convert('RGB')

            # Sauvegarder l'image redimensionnée, en écrasant l'originale
            img_redimensionnee.save(chemin_complet)

print("Toutes les images ont été redimensionnées avec succès.")
