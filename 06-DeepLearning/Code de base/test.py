import pandas as pd

# Lire le fichier d'annotations
annotations_path = 'annotations.csv'
annotations = pd.read_csv(annotations_path)

# Ajouter une colonne pour marquer les coins hors champ
annotations['topleft_out'] = annotations['topleft'].apply(lambda x: False)
annotations['topright_out'] = annotations['topright'].apply(lambda x: False)
annotations['bottomright_out'] = annotations['bottomright'].apply(lambda x: False)
annotations['bottomleft_out'] = annotations['bottomleft'].apply(lambda x: False)

# Marquer les coins hors champ pour les images spécifiques
annotations.loc[annotations['filename'] == 'img-14.jpg', 'bottomright_out'] = True
annotations.loc[annotations['filename'] == 'img-16.jpg', 'bottomright_out'] = True
annotations.loc[annotations['filename'] == 'img-18.jpg', 'bottomright_out'] = True
annotations.loc[annotations['filename'] == 'img-6.jpg', 'topleft_out'] = True

# Sauvegarder les annotations modifiées
annotations.to_csv('annotations.csv', index=False)
