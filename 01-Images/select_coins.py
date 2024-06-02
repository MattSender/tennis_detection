import matplotlib.pyplot as plt
import cv2
import csv
import os
import numpy as np

def add_margin(image, margin=50):
    height, width, channels = image.shape
    new_image = np.zeros((height + 2 * margin, width + 2 * margin, channels), dtype=np.uint8)
    new_image[margin:margin + height, margin:margin + width] = image
    return new_image

def select_points(image_path, margin=50):
    image = cv2.imread(image_path)
    image_with_margin = add_margin(image, margin)
    image_rgb = cv2.cvtColor(image_with_margin, cv2.COLOR_BGR2RGB)
    points = []

    fig, ax = plt.subplots()
    ax.imshow(image_rgb)

    def onclick(event):
        if event.xdata and event.ydata:
            points.append((int(event.xdata) - margin, int(event.ydata) - margin))
            ax.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()
            if len(points) == 4:
                plt.close()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return points

image_folder = "img"
csv_file = "borders.csv"
csv_columns = ['filename', 'topleft_x', 'topleft_y', 'topright_x', 'topright_y', 'bottomright_x', 'bottomright_y',
               'bottomleft_x', 'bottomleft_y']

with open(csv_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(csv_columns)

    for image_file in os.listdir(image_folder):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_path = os.path.join(image_folder, image_file)
            points = select_points(image_path)
            if len(points) == 4:
                row = [image_file]
                for point in points:
                    row.extend(point)
                csvwriter.writerow(row)

print(f"Les coordonnées des bordures ont été enregistrées dans le fichier {csv_file}")
