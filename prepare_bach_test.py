import os
from PIL import Image

SOURCE_DIR = r"_TestDataset\ICIAR2018_BACH_Challenge_TestDataset\Photos"
TARGET_DIR = r"bc\bach_test_prepared"

os.makedirs(TARGET_DIR, exist_ok=True)

IMG_SIZE = (224, 224)

count = 0

for file in os.listdir(SOURCE_DIR):
    if file.lower().endswith(".tif"):
        src_path = os.path.join(SOURCE_DIR, file)
        dst_path = os.path.join(TARGET_DIR, file.replace(".tif", ".png"))

        try:
            img = Image.open(src_path).convert("RGB")
            img = img.resize(IMG_SIZE)
            img.save(dst_path)
            count += 1
        except Exception as e:
            print("❌ Failed:", file, e)

print(f"✅ Converted {count} test images")