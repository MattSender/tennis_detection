import cv2
import os
import csv

# Path to the directory containing images
directory_path = '../vierge'
# CSV file to store the coordinates
csv_filename = '../annotations.csv'

# Function to handle mouse clicks
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Store the point coordinates
        points.append((x, y))
        # Display the point on the image
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", img)

# Initialize CSV file
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'topleft', 'topright', 'bottomright', 'bottomleft'])

# Process each file in the directory
for filename in os.listdir(directory_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(directory_path, filename)
        img = cv2.imread(img_path)
        if img is not None:  # Make sure the image has been loaded correctly
            # Resize image to 1024x768
            img = cv2.resize(img, (1024, 768))
            points = []

            # Display the image
            cv2.imshow("Image", img)
            cv2.setMouseCallback("Image", click_event)

            # Wait for the 'p' key to be pressed or 4 points to be clicked
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('p'):
                    points.append((-1, -1))
                if len(points) == 4 or key == 27:  # 27 is the Esc key
                    break

            # Save the points to the CSV
            with open(csv_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([filename] + points)
            cv2.destroyAllWindows()
        else:
            print(f"Failed to load image {filename}")

print("Annotation complete. Data saved to", csv_filename)
