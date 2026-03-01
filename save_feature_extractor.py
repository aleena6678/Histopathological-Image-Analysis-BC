from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model

# Load pretrained CNN (ImageNet)
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

# This IS the feature extractor
feature_extractor = Model(
    inputs=base_model.input,
    outputs=base_model.output
)

# Save it
feature_extractor.save("feature_extractor.keras")

print("✅ Feature extractor saved as feature_extractor.keras")
