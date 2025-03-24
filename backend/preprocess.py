import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


DATASET_DIR = "dataset/"
OUTPUT_PATH = "model_training/processed_data.npz"
IMG_SIZE = 128


def load_and_preprocess_images():
    images, labels = [], []
    class_names = os.listdir(DATASET_DIR)

    for class_index, category in enumerate(class_names):
        category_path = os.path.join(DATASET_DIR, category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            images.append(img)
            labels.append(class_index)

    return np.array(images), np.array(labels), class_names


X, y, class_names = load_and_preprocess_images()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

np.savez(
    OUTPUT_PATH,
    X_train=X_train,
    X_val=X_val,
    y_train=y_train,
    y_val=y_val,
    class_names=class_names,
)

print(f"Preprocessed data saved to: {OUTPUT_PATH}")
