import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Function to preprocess test images
def preprocess_test_image(image_path):
    image = cv2.imread(image_path)
    # No resizing
    image_normalized = image / 255.0  # Normalize the image
    return image, np.expand_dims(image_normalized, axis=0)

# Function to display predictions on the image
def display_predictions(image, predictions):
    predicted_coords = predictions[0].reshape((4, 2)) * np.array([1024, 768])
    for coord in predicted_coords:
        cv2.circle(image, (int(coord[0]), int(coord[1])), 5, (0, 255, 0), -1)
    return predicted_coords

# Load the trained model
model = tf.keras.models.load_model('tennis_court_model_efficientnetb0.h5')

# Path to the test image
test_image_path = 'images/test_data/img-test.jpg'  # Replace with your test image path

# Preprocess the test image
original_image, preprocessed_image = preprocess_test_image(test_image_path)

# Make a prediction
predictions = model.predict(preprocessed_image)

# Display predictions on the image
predicted_coords = display_predictions(original_image, predictions)

# Save the annotated image
output_image_path = os.path.join('images/test_results', os.path.basename(test_image_path))
cv2.imwrite(output_image_path, original_image)

# Show the image with predicted coordinates
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title(f'Predicted Corners for {os.path.basename(test_image_path)}')
plt.show()

print(f"Predicted coordinates for {os.path.basename(test_image_path)}: {predicted_coords}")
