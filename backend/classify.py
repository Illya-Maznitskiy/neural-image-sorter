import os
import tensorflow as tf
import numpy as np
import logging


# Path to the model
model_path = os.path.join(os.getcwd(), "model", "model.h5")

print("Model path:", os.path.abspath(model_path))


# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Get the current directory
current_directory = os.getcwd()
logging.info(f"📂 Current Directory: {current_directory}")

# Log all files in the directory
files_in_directory = os.listdir(current_directory)
logging.info(f"📄 Files in Current Directory: {files_in_directory}")

# Check if the model file exists
if os.path.exists(model_path):
    logging.info(f"✅ Model file found at: {model_path}")
    logging.info(f"📏 Model file size: {os.path.getsize(model_path)} bytes")
else:
    logging.error(f"❌ Model file NOT found at: {model_path}")
    logging.error(f"🛠️ Absolute Path: {os.path.abspath(model_path)}")
    model = None  # Prevent further errors
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Attempt to load the model
try:
    logging.info(f"Attempting to load model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
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
