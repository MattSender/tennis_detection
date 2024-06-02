import os
import cv2
import csv
import matplotlib.pyplot as plt

image_directory = '../images/train_data2'

def annotate_image(image_path):
    image = cv2.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    points = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            points.append((int(event.xdata), int(event.ydata)))
            ax.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()
        if len(points) == 4:
            plt.close()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return points

def annotate_images_in_directory(directory):
    annotations = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            print(f"Annotating {filename}...")
            points = annotate_image(image_path)
            if len(points) == 4:
                annotations.append({
                    'filename': filename,
                    'topleft': points[0],
                    'topright': points[1],
                    'bottomright': points[2],
                    'bottomleft': points[3]
                })
            else:
                print(f"Skipping {filename}, not enough points annotated.")

    return annotations

def write_annotations_to_csv(annotations, csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'topleft', 'topright', 'bottomright', 'bottomleft']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for annotation in annotations:
            writer.writerow({
                'filename': annotation['filename'],
                'topleft': f"({annotation['topleft'][0]}, {annotation['topleft'][1]})",
                'topright': f"({annotation['topright'][0]}, {annotation['topright'][1]})",
                'bottomright': f"({annotation['bottomright'][0]}, {annotation['bottomright'][1]})",
                'bottomleft': f"({annotation['bottomleft'][0]}, {annotation['bottomleft'][1]})"
            })

annotations = annotate_images_in_directory(image_directory)
write_annotations_to_csv(annotations, 'annotations3.csv')

print("Annotation completed and saved to 'annotations3.csv'.")
