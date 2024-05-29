import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Path to the annotation file and image directory
annotations_path = 'annotation.csv'
image_directory = 'images/train_data'
test_images_directory = 'images/test_data'
output_directory = 'images/test_results'

# Load the annotations
annotations = pd.read_csv(annotations_path)


# Function to load images and labels
def load_data(annotations):
    images = []
    labels = []
    for idx, row in annotations.iterrows():
        img_path = os.path.join(image_directory, row['filename'])
        image = cv2.imread(img_path)
        if image is None:
            continue  # Skip if image is not found
        image = cv2.resize(image, (224, 224))  # Resize to match input shape of EfficientNetB0

        topleft = eval(row['topleft'])
        topright = eval(row['topright'])
        bottomright = eval(row['bottomright'])
        bottomleft = eval(row['bottomleft'])

        # Check if any point is marked as out-of-bounds
        if row['topleft_out']:
            topleft = (-1, -1)
        if row['topright_out']:
            topright = (-1, -1)
        if row['bottomright_out']:
            bottomright = (-1, -1)
        if row['bottomleft_out']:
            bottomleft = (-1, -1)

        images.append(image)
        labels.append([topleft, topright, bottomright, bottomleft])

    return np.array(images), np.array(labels)


# Prepare the data
images, labels = load_data(annotations)

# Normalize the images
images = images / 255.0

# Flatten the labels array
labels = labels.reshape((labels.shape[0], -1))

# Normalize the labels (assuming the images are 224x224)
labels = labels / np.array([224, 224, 224, 224, 224, 224, 224, 224])

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define the base model
base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Add custom layers on top
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(8)(x)  # 8 outputs for the coordinates of the 4 corners

model = Model(inputs, x)

# Compile the model
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

# Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with data augmentation and early stopping
train_generator = datagen.flow(x_train, y_train, batch_size=16)
val_generator = datagen.flow(x_val, y_val, batch_size=16)

history = model.fit(train_generator, epochs=100, validation_data=val_generator, callbacks=[early_stopping])

# Save the model
model.save('tennis_court_model_efficientnet_early_stopping.h5')

# Plot training & validation loss values
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

print("Model training complete and saved as 'tennis_court_model_efficientnet_early_stopping.h5'.")


# Function to preprocess test images
def preprocess_test_image(image_path):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (224, 224))  # Resize to match input shape of EfficientNetB0
    image_normalized = image_resized / 255.0  # Normalize the image
    return image, np.expand_dims(image_normalized, axis=0)


# Function to display predictions on the image
def display_predictions(image, predictions):
    predicted_coords = predictions[0].reshape((4, 2)) * np.array([224, 224])
    for coord in predicted_coords:
        cv2.circle(image, (int(coord[0]), int(coord[1])), 5, (0, 255, 0), -1)
    return predicted_coords


# Test the model on all images in the test directory
for filename in os.listdir(test_images_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(test_images_directory, filename)
        original_image, preprocessed_image = preprocess_test_image(image_path)

        # Make a prediction
        predictions = model.predict(preprocessed_image)

        # Display predictions on the image
        predicted_coords = display_predictions(original_image, predictions)

        # Save the annotated image
        output_image_path = os.path.join(output_directory, filename)
        cv2.imwrite(output_image_path, original_image)

        # Show the image with predicted coordinates
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Predicted Corners for {filename}')
        plt.show()

        print(f"Predicted coordinates for {filename}: {predicted_coords}")

print("Testing complete. Annotated images are saved in the 'test_results' directory.")
