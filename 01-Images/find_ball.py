import cv2
import csv
import os
import numpy as np
from roboflow import Roboflow
import matplotlib.pyplot as plt
from scipy.spatial import distance

rf = Roboflow(api_key="3JhXCrYZNeKlaOV1Darg")
project = rf.workspace().project("tennis-uc2es")
model = project.version(1).model

csv_file = "borders.csv"

borders = {}
with open(csv_file, 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        filename = row['filename']
        borders[filename] = [
            (int(row['topleft_x']), int(row['topleft_y'])),
            (int(row['topright_x']), int(row['topright_y'])),
            (int(row['bottomright_x']), int(row['bottomright_y'])),
            (int(row['bottomleft_x']), int(row['bottomleft_y']))
        ]

def add_margin(image, margin=50):
    height, width, channels = image.shape
    new_image = np.zeros((height + 2 * margin, width + 2 * margin, channels), dtype=np.uint8)
    new_image[margin:margin + height, margin:margin + width] = image
    return new_image

def crop_image_with_padding(image, points, padding=50):
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points)], 255)
    cropped_image = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(np.array(points))
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, image.shape[1] - x)
    h = min(h + 2 * padding, image.shape[0] - y)
    cropped_image = cropped_image[y:y+h, x:x+w]
    return cropped_image

def merge_close_detections(detections, threshold=10):
    if not detections:
        return []
    merged_detections = []
    used_indices = set()
    for i in range(len(detections)):
        if i in used_indices:
            continue
        current = detections[i]
        cluster = [current]
        used_indices.add(i)
        for j in range(i + 1, len(detections)):
            if j in used_indices:
                continue
            other = detections[j]
            if distance.euclidean((current['x'], current['y']), (other['x'], other['y'])) < threshold:
                cluster.append(other)
                used_indices.add(j)
        if cluster:
            mean_x = int(np.mean([d['x'] for d in cluster]))
            mean_y = int(np.mean([d['y'] for d in cluster]))
            merged_detections.append({'x': mean_x, 'y': mean_y, 'class': current['class']})
    return merged_detections

image_folder = "img"
results = []

for image_file in os.listdir(image_folder):
    if image_file in borders:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        points = borders[image_file]
        image_with_margin = add_margin(image, 50)
        points_with_margin = [(x + 50, y + 50) for x, y in points]
        cropped_image = crop_image_with_padding(image_with_margin, points_with_margin)
        cropped_image_path = os.path.join(image_folder, f"cropped_{image_file}")
        cv2.imwrite(cropped_image_path, cropped_image)
        prediction = model.predict(cropped_image_path, confidence=2, overlap=100).json()
        merged_predictions = merge_close_detections(prediction['predictions'], threshold=10)
        ball_count = 1
        for pred in merged_predictions:
            x = pred['x']
            y = pred['y']
            class_name = f"ball{ball_count}"
            top_left = (int(x - 5), int(y - 5))
            bottom_right = (int(x + 5), int(y + 5))
            cv2.rectangle(cropped_image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(cropped_image, class_name, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            results.append([image_file, class_name, x, y])
            ball_count += 1
        image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()

output_csv_file = 'detection_results.csv'
csv_columns = ['filename', 'class', 'positionball_x', 'positionball_y']

with open(output_csv_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(csv_columns)
    csvwriter.writerows(results)

print(f"Les résultats ont été enregistrés dans le fichier {output_csv_file}")
