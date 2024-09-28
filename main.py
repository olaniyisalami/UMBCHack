import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pandas as pd
import json

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

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
def download_images_from_json(data, categories, download_path='trash'):
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
# Run the function
def main():
    download_images_from_json(data, categories)

if __name__ == "__main__":
    main()