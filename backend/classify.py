import os

import requests
import tensorflow as tf
import numpy as np
import logging


# Set up logging
logging.basicConfig(level=logging.DEBUG)

MODEL_PATH = os.path.join(os.getcwd(), "model", "model.h5")
MODEL_URL = (
    "https://drive.google.com/uc?export=download&"
    "id=10Pg-LLFNIq6Gx_wHOQsauTe-2E-rGCcN"
)


# Ensure model exists and is correct size
if (
    not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000
):  # Less than 1MB = corrupt
    logging.warning(
        "\U0001f817 Model file " "is missing or corrupted. Downloading..."
    )
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    logging.info("✅ Model downloaded successfully!")

# Attempt to load the model
try:
    logging.info(f"Attempting to load model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("✅ Model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    model = None  # Prevent crashes


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

    class_names = ["cats", "dogs"]
    predicted_class = class_names[class_idx]

    return predicted_class, confidence
