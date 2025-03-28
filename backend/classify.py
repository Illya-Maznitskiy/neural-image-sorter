import tensorflow as tf
import numpy as np
import logging

model_path = "model/model.h5"
logging.basicConfig(level=logging.DEBUG)

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
