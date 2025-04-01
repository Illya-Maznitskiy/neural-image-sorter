import os

import requests
import tensorflow as tf
import numpy as np
import logging


# Path to the model
MODEL_PATH = os.path.join(os.getcwd(), "model", "model.h5")
MODEL_URL = "https://drive.google.com/uc?export=download&id=10Pg-LLFNIq6Gx_wHOQsauTe-2E-rGCcN"

print("Model path:", os.path.abspath(MODEL_PATH))

# Ensure model exists and is correct size
if (
    not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000
):  # Less than 1MB = corrupt
    print("\U0001f817 Model file is missing or corrupted. Downloading...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded successfully!")


# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Get the current directory
current_directory = os.getcwd()
logging.info(f"📂 Current Directory: {current_directory}")

# Log all files in the directory
files_in_directory = os.listdir(current_directory)
logging.info(f"📄 Files in Current Directory: {files_in_directory}")

# Check if the model file exists
if os.path.exists(MODEL_PATH):
    logging.info(f"✅ Model file found at: {MODEL_PATH}")
    logging.info(f"📏 Model file size: {os.path.getsize(MODEL_PATH)} bytes")
else:
    logging.error(f"❌ Model file NOT found at: {MODEL_PATH}")
    logging.error(f"🛠️ Absolute Path: {os.path.abspath(MODEL_PATH)}")
    model = None  # Prevent further errors
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Attempt to load the model
try:
    logging.info(f"Attempting to load model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("✅ Model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    model = None  # Prevent crashes


# Function to classify image
def classify_image(img):
    if model is None:
        logging.error("🚨 Model is not loaded! Classification cannot proceed.")
        raise RuntimeError("Model is not loaded.")
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
