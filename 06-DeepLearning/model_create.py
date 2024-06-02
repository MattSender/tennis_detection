import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

annotations_path = 'annotations.csv'
image_directory = 'images/train_data'

annotations = pd.read_csv(annotations_path)

def load_data(annotations):
    images = []
    labels = []
    for idx, row in annotations.iterrows():
        img_path = os.path.join(image_directory, row['filename'])
        image = cv2.imread(img_path)
        if image is None:
            print(f"Image {row['filename']} not found, skipping.")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        topleft = eval(row['topleft'])
        topright = eval(row['topright'])
        bottomright = eval(row['bottomright'])
        bottomleft = eval(row['bottomleft'])

        if 'topleft_out' in row and row['topleft_out']:
            topleft = (-50, 350)
        if 'topright_out' in row and row['topright_out']:
            topright = (1024, 0)
        if 'bottomright_out' in row and row['bottomright_out']:
            bottomright = (1100, 450)
        if 'bottomleft_out' in row and row['bottomleft_out']:
            bottomleft = (-50, 850)

        images.append(image)
        labels.append([topleft, topright, bottomright, bottomleft])

    return np.array(images), np.array(labels)

images, labels = load_data(annotations)

images = images / 255.0

labels = labels.reshape((labels.shape[0], -1))

labels = labels / np.array([1024, 768, 1024, 768, 1024, 768, 1024, 768])

for i in range(5):
    plt.imshow(images[i])
    coords = labels[i].reshape((4, 2)) * np.array([1024, 768])
    for coord in coords:
        plt.scatter(coord[0], coord[1], c='r')
    plt.show()

images_train, images_val, labels_train, labels_val = train_test_split(images, labels, test_size=0.2, shuffle=True, random_state=42)

data_gen_args = dict(rotation_range=15,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     fill_mode='nearest')
image_datagen = ImageDataGenerator(**data_gen_args)

base_model = ResNet50V2(input_shape=(768, 1024, 3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = Input(shape=(768, 1024, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(8)(x)

model = Model(inputs, x)

model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

train_gen = image_datagen.flow(images_train, labels_train, batch_size=16)
val_gen = image_datagen.flow(images_val, labels_val, batch_size=16)

history = model.fit(train_gen,
                    epochs=10000,
                    validation_data=val_gen)

model.save('tennis_court_model_resnet50v2.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

print("Model training complete and saved as 'tennis_court_model_resnet50v2.h5'.")
