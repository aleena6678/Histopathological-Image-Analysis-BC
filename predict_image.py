import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

CLASSES = ["Benign", "Malignant", "Normal"]

svm = joblib.load("svm_bach.joblib")
feature_extractor = load_model("feature_extractor.keras")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = preprocess_input(img)

    features = feature_extractor.predict(np.expand_dims(img, 0), verbose=0)
    probs = svm.predict_proba(features)[0]

    for cls, p in zip(CLASSES, probs):
        print(f"{cls}: {p*100:.2f}%")

    best = np.argmax(probs)
    confidence = probs[best] * 100

    if confidence < 50:
        print("\n🟡 UNCERTAIN — Needs Expert Review")
    else:
        print(f"\n🔴 Prediction: {CLASSES[best]} ({confidence:.2f}%)")

predict_image(r"C:\bc\bc\bach_3class\val\Normal\n100.png")
