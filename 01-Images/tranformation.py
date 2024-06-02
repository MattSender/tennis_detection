import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

image_path = "img/P1140765.jpg"
borders_csv_path = "borders.csv"
detection_csv_path = "detection_results.csv"

image = cv2.imread(image_path)

borders = {}
with open(borders_csv_path, 'r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        filename = row['filename']
        borders[filename] = [
            (int(row['topleft_x']), int(row['topleft_y'])),
            (int(row['topright_x']), int(row['topright_y'])),
            (int(row['bottomright_x']), int(row['bottomright_y'])),
            (int(row['bottomleft_x']), int(row['bottomleft_y']))
        ]

points_original = np.float32(borders["P1140765.jpg"])

output_width = 1097
output_height = 1189

points_final = np.float32([
    [0, 0],
    [output_width, 0],
    [output_width, output_height],
    [0, output_height]
])

matrix = cv2.getPerspectiveTransform(points_original, points_final)

warped_image = cv2.warpPerspective(image, matrix, (output_width, output_height))

ball_detections = []
with open(detection_csv_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)
    for row in csvreader:
        filename, class_name, x, y = row
        if filename == "P1140765.jpg":
            ball_detections.append([int(x), int(y)])

ball_detections_transformed = []
for (x, y) in ball_detections:
    pts = np.float32([[x, y]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, matrix)
    ball_detections_transformed.append((dst[0][0][0], dst[0][0][1]))

for (x, y) in ball_detections_transformed:
    cv2.circle(warped_image, (int(x), int(y)), 10, (255, 0, 0), -1)

plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
plt.title("Vue du dessus avec les balles détectées")

for (x, y) in ball_detections_transformed:
    plt.plot(x, y, 'ro')
    plt.text(x, y, f'({int(x)}, {int(y)})', color='yellow', fontsize=12, ha='right')

plt.axis('off')
plt.show()

output_path = "warped_with_balls_annotated_P1140765.jpg"
cv2.imwrite(output_path, warped_image)

print(f"L'image finale avec les balles détectées a été enregistrée à {output_path}")

world_coordinates_csv = "ball_world_coordinates2.csv"
with open(world_coordinates_csv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['filename', 'class', 'world_x', 'world_y'])
    for i, (x, y) in enumerate(ball_detections_transformed):
        csvwriter.writerow(["P1140765.jpg", f"ball{i+1}", int(x), int(y)])

print(f"Les coordonnées des balles dans le repère du monde réel ont été enregistrées à {world_coordinates_csv}")
