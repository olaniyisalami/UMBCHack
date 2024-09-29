# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import json
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tkinter as tk

import os
import requests

# Load the JSON file
with open('data/annotations.json', 'r') as f:
    data = json.load(f)

# Dictionary to simulate the categories from another part of the annotations.json file
# Assuming there's a separate categories section or we assign categories based on image id
categories = {
    # Add more image id to category mappings if necessary
}

# Function to download images into subfolders based on category
def download_images_from_json(data, categories, download_path='~/trash'):
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Loop through images in the annotations
    for i in data['categories']:
        dirPath = os.path.join(download_path, i['supercategory'])
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        categories[i['supercategory']] = []
    
    imagesList = data['images']
    categoriesList = data['categories']
    last = -1
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        if (last != image_id):
            image = imagesList[image_id]
            image_url = image['flickr_640_url']
            category = categoriesList[annotation['category_id']]['supercategory']

        # Create category folder
            category_folder = os.path.join(download_path, category)
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)

        # Download the image
            try:
                response = requests.get(image_url)
                response.raise_for_status()  # Check for request errors

            # Save the image with a unique name
                file_name = f'{image_id}.jpg'
                file_path = os.path.join(category_folder, file_name)
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {file_name} into {category_folder}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {image_url}: {e}")
        last = image_id
def train_model():
    # Create the training dataset with a split
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory='trash',
        labels='inferred',
        label_mode='categorical',  # One-hot encoding since you are using categorical crossentropy
        image_size=(256, 256),
        validation_split=0.2,  # Splitting 20% for validation
        subset="training",
        seed=123  # Ensuring shuffling is deterministic across both train/val datasets
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory='trash',
        labels='inferred',
        label_mode='categorical',
        image_size=(256, 256),
        validation_split=0.2,
        subset="validation",
        seed=123 
    )

    class_names = train_ds.class_names
    print(class_names)

    # Batch and prefetch to optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Normalize images in both datasets
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Get one batch to check normalization
    for image_batch, labels_batch in train_ds.take(1):
        print(f'Image batch shape: {image_batch.shape}')
        print(f'Label batch shape: {labels_batch.shape}')
        first_image = image_batch[0]
        print(np.min(first_image), np.max(first_image))  # This should print (0.0, 1.0) after normalization
    # Get the number of classes
    num_classes = len(class_names)
    print(num_classes)

    # Define the model
    model = Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(256, 256, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Final layer for one-hot encoded output
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    # Print model summary
    model.summary()

    # Train the model
    epochs = 10
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # Plot training results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    return model

def main():
    train_model()
    window = tk.Tk()
    window.title("My Windowed Application")
    window.mainloop()


if __name__ == "__main__":
    main()