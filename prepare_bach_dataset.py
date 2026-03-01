import os
import shutil
import random
from PIL import Image

SOURCE_DIR = r"ICIAR2018_BACH_Challenge\Photos"
TARGET_DIR = r"bc\bach_3class"

TRAIN_RATIO = 0.8
IMG_SIZE = (224, 224)

CLASSES = {
    "Benign": "Benign",
    "Normal": "Normal",
    "InSitu": "Malignant",
    "Invasive": "Malignant"
}

def convert_and_copy(src, dst):
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize(IMG_SIZE)
        img.save(dst.replace(".tif", ".png"))
        return True
    except:
        return False

for split in ["train", "val"]:
    for cls in ["Benign", "Normal", "Malignant"]:
        os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

for folder, target_class in CLASSES.items():
    images = os.listdir(os.path.join(SOURCE_DIR, folder))
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for img_list, split in [(train_imgs, "train"), (val_imgs, "val")]:
        for img_name in img_list:
            src_path = os.path.join(SOURCE_DIR, folder, img_name)
            dst_path = os.path.join(
                TARGET_DIR, split, target_class, img_name
            )
            convert_and_copy(src_path, dst_path)

print("✅ BACH dataset prepared successfully!")
