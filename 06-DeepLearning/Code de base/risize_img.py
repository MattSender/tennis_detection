from PIL import Image
import os

dossier = '../images/train_data'
nouvelle_largeur = 1024
nouvelle_hauteur = 768

for nom_fichier in os.listdir(dossier):
    if nom_fichier.lower().endswith('.jpg'):
        chemin_complet = os.path.join(dossier, nom_fichier)
        with Image.open(chemin_complet) as img:
            img_redimensionnee = img.resize((nouvelle_largeur, nouvelle_hauteur))
            if img_redimensionnee.mode != 'RGB':
                img_redimensionnee = img_redimensionnee.convert('RGB')
            img_redimensionnee.save(chemin_complet)

print("Toutes les images ont été redimensionnées avec succès.")
