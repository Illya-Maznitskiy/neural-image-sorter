import tensorflow as tf
import numpy as np
import logging


model_path = "model/model.h5"
# Set up logging
logging.basicConfig(level=logging.DEBUG)

try:
    model = tf.keras.models.load_model(model_path)
    logging.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logging.error(f"Error loading model from {model_path}: {e}")


def classify_image(img):
    # Convert the image to RGB to ensure it has 3 channels (no alpha)
    img = img.convert("RGB")

    # Resize the image to 128x128
    img = img.resize((128, 128))

    # Convert image to numpy array
    img_array = np.array(img) / 255.0

    # Expand the dimensions of the image to match the input format of the model
    img_array = np.expand_dims(img_array, axis=0)

    # Predict class
    predictions = model.predict(img_array)

    # Get the class with the highest probability
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]

    class_names = [
        "cats",
        "dogs",
    ]  # Make sure this matches the classes your model was trained with

    predicted_class = class_names[class_idx]

    return predicted_class, confidence
