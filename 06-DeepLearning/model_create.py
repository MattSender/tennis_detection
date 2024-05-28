import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

annotations_path = 'annotations.csv'
image_directory = 'images/train_data'

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
        # No resizing
        topleft = eval(row['topleft'])
        topright = eval(row['topright'])
        bottomright = eval(row['bottomright'])
        bottomleft = eval(row['bottomleft'])

        # Check if any point is marked as out-of-bounds
        if 'topleft_out' in row and row['topleft_out']:
            topleft = (-1, -1)
        if 'topright_out' in row and row['topright_out']:
            topright = (-1, -1)
        if 'bottomright_out' in row and row['bottomright_out']:
            bottomright = (-1, -1)
        if 'bottomleft_out' in row and row['bottomleft_out']:
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

# Normalize the labels (assuming the images are 1024x768)
labels = labels / np.array([1024, 768, 1024, 768, 1024, 768, 1024, 768])

# Split the data into training and test sets
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, shuffle=True, random_state=42)

# Define the data augmentation
data_gen_args = dict(rotation_range=15,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     fill_mode='nearest')
image_datagen = ImageDataGenerator(**data_gen_args)

# Define the base model
base_model = EfficientNetB0(input_shape=(768, 1024, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Add custom layers on top
inputs = Input(shape=(768, 1024, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(8)(x)  # 8 outputs for the coordinates of the 4 corners

model = Model(inputs, x)

# Compile the model
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

# Train the model with data augmentation
train_gen = image_datagen.flow(images_train, labels_train, batch_size=16)
history = model.fit(train_gen, epochs=200)

# Save the model
model.save('tennis_court_model_efficientnetb0.h5')

# Plot training loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

print("Model training complete and saved as 'tennis_court_model_efficientnetb0.h5'.")
