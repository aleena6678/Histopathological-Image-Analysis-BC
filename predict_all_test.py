import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

TEST_DIR = r"bc\bach_test_prepared"
CLASSES = ["Benign", "Malignant", "Normal"]

svm = joblib.load("svm_bach.joblib")
feature_extractor = load_model("feature_extractor.keras")

results = []

for file in os.listdir(TEST_DIR):
    img_path = os.path.join(TEST_DIR, file)

    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = preprocess_input(img)

    features = feature_extractor.predict(np.expand_dims(img, 0), verbose=0)
    probs = svm.predict_proba(features)[0]

    best = np.argmax(probs)
    confidence = probs[best] * 100

    results.append((file, CLASSES[best], confidence))

# Print summary
malignant_count = 0
benign_count = 0
normal_count = 0

for r in results:
    if r[1] == "Malignant":
        malignant_count += 1
    elif r[1] == "Benign":
        benign_count += 1
    else:
        normal_count += 1

print("\n📊 TEST SET SUMMARY")
print("Malignant:", malignant_count)
print("Benign:", benign_count)
print("Normal:", normal_count)

print("\n🔎 First 10 Predictions:")
for r in results[:10]:
    print(f"{r[0]} → {r[1]} ({r[2]:.2f}%)")