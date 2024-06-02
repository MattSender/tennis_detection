import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def preprocess_test_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_normalized = image / 255.0
    return image, np.expand_dims(image_normalized, axis=0)

def display_predictions(image, predictions):
    predicted_coords = predictions[0].reshape((4, 2)) * np.array([1024, 768])
    for coord in predicted_coords:
        cv2.circle(image, (int(coord[0]), int(coord[1])), 5, (0, 255, 0), -1)
    return predicted_coords

model = tf.keras.models.load_model('tennis_court_model_resnet50v2.h5')

test_image_path = 'images/test_data/img-test.jpg'

original_image, preprocessed_image = preprocess_test_image(test_image_path)

predictions = model.predict(preprocessed_image)

predicted_coords = display_predictions(original_image, predictions)

output_image_path = os.path.join('images/test_results', os.path.basename(test_image_path))
cv2.imwrite(output_image_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

plt.figure(figsize=(10, 8))
plt.imshow(original_image)
plt.title(f'Predicted Corners for {os.path.basename(test_image_path)}')
plt.show()

print(f"Predicted coordinates for {os.path.basename(test_image_path)}: {predicted_coords}")
