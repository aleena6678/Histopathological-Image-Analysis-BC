import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

svm = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    C=10,
    gamma="scale"
)

svm.fit(X_train, y_train)

y_pred = svm.predict(X_val)

print("\n📊 Classification Report:")
print(classification_report(y_val, y_pred, target_names=["Benign","Malignant","Normal"]))

joblib.dump(svm, "svm_bach.joblib")
print("💾 SVM model saved")
