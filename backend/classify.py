import tensorflow as tf
import numpy as np


model_path = "model/model.h5"
model = tf.keras.models.load_model(model_path)


def classify_image(img):
    # Debug print to check the image size before resizing
    print(f"Original image size: {img.size}")

    # Convert the image to RGB to ensure it has 3 channels (no alpha)
    img = img.convert("RGB")
    print(f"Image converted to RGB. New size: {img.size}")

    # Resize the image to 128x128
    img = img.resize((128, 128))
    print(f"Resized image size: {img.size}")

    # Convert image to numpy array
    img_array = np.array(img) / 255.0

    # Expand the dimensions of the image to match the input format of the model
    img_array = np.expand_dims(img_array, axis=0)

    # Predict class
    predictions = model.predict(img_array)
    print(f"Prediction result: {predictions}")

    # Get the class with the highest probability
    class_idx = np.argmax(predictions[0])

    class_names = [
        "cats",
        "dogs",
    ]  # Make sure this matches the classes your model was trained with

    predicted_class = class_names[class_idx]
    print(f"Predicted class: {predicted_class}")
    return predicted_class
