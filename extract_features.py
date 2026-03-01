import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

DATASET_DIR = r"bc\bach_3class"
CLASSES = ["Benign", "Malignant", "Normal"]

model = load_model("feature_extractor.keras")

def extract(split):
    X, y = [], []

    for label, cls in enumerate(CLASSES):
        folder = os.path.join(DATASET_DIR, split, cls)
        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            feature = model.predict(np.expand_dims(image, 0), verbose=0)[0]

            X.append(feature)
            y.append(label)

    return np.array(X), np.array(y)

X_train, y_train = extract("train")
X_val, y_val = extract("val")

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)

print("✅ Features extracted and saved")
print("Train:", X_train.shape, "Val:", X_val.shape)
