import os

import tensorflow as tf
import numpy as np
import logging

# Path to the model
model_path = "model/model.h5"

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Get the current directory
current_directory = os.getcwd()
logging.info(f"Current Directory: {current_directory}")

# Log all files in the current directory
files_in_directory = os.listdir(current_directory)
logging.info(f"Files in Current Directory: {files_in_directory}")

# Check if the model file exists
if os.path.exists(model_path):
    logging.info(f"Model file found at: {model_path}")
else:
    logging.error(f"Model file NOT found at: {model_path}")
    # You can print the absolute path here as well
    logging.error(f"Absolute model path: {os.path.abspath(model_path)}")

# Attempt to load the model
try:
    model = tf.keras.models.load_model(model_path)
    logging.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logging.error(f"Error loading model from {model_path}: {e}")


def classify_image(img):
    img = img.convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    class_names = ["cats", "dogs"]  # Update to your model's classes
    predicted_class = class_names[class_idx]

    return predicted_class, confidence
