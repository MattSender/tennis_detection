import os
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter

input_folder = "../vierge/"
output_folder = "./img/"

os.makedirs(output_folder, exist_ok=True)

def process_image(file_path, output_path):
    img_vierge = Image.open(file_path)
    img_vierge_array = np.array(img_vierge)

    blurred_img = gaussian_filter(img_vierge_array, sigma=2)

    filtered_img = img_vierge_array - blurred_img
    filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)

    filtered_img_pil = Image.fromarray(filtered_img)
    filtered_img_bw = ImageOps.grayscale(filtered_img_pil)

    threshold = 128
    img_filtered_threshold = filtered_img_bw.point(lambda p: p > threshold and 255)
    img_filtered_threshold.save(output_path)

for file_name in os.listdir(input_folder):
    if file_name.endswith(('.jpg', '.jpeg', '.png')):
        file_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        process_image(file_path, output_path)

import matplotlib.pyplot as plt

example_image = os.path.join(input_folder, os.listdir(input_folder)[0])
img_vierge = Image.open(example_image)
img_vierge_array = np.array(img_vierge)
blurred_img = gaussian_filter(img_vierge_array, sigma=2)
filtered_img = img_vierge_array - blurred_img
filtered_img = np.clip(filtered_img, 0, 255).astype(np.uint8)
filtered_img_pil = Image.fromarray(filtered_img)
filtered_img_bw = ImageOps.grayscale(filtered_img_pil)
img_filtered_threshold = filtered_img_bw.point(lambda p: p > 128 and 255)

fig, axs = plt.subplots(1, 4, figsize=(24, 6))
axs[0].imshow(img_vierge)
axs[0].set_title("Image Vierge")
axs[0].axis('off')

axs[1].imshow(filtered_img)
axs[1].set_title("Image après Filtre")
axs[1].axis('off')

axs[2].imshow(filtered_img_bw, cmap='gray')
axs[2].set_title("Image en Niveaux de Gris")
axs[2].axis('off')

axs[3].imshow(img_filtered_threshold, cmap='gray')
axs[3].set_title("Image avec Seuillage")
axs[3].axis('off')

plt.show()
