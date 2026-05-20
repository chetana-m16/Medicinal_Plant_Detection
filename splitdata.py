import os
import shutil
import random
from tqdm import tqdm


SOURCE_DIR = r"C:\Users\chetn\OneDrive\Desktop\Machine_Learning_Project\MedicinalPlants_Identification_and_Classification\dataset"

#  Output path for the split dataset
BASE_OUTPUT_DIR = r"C:\Users\chetn\OneDrive\Desktop\Machine_Learning_Project\MedicinalPlants_Identification_and_Classification\dataset2"
TRAIN_DIR = os.path.join(BASE_OUTPUT_DIR, "train")
TEST_DIR = os.path.join(BASE_OUTPUT_DIR, "test")

# Test size ratio
TEST_RATIO = 0.2  # 20% for testing

def create_split_folders():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    for class_name in os.listdir(SOURCE_DIR):
        class_path = os.path.join(SOURCE_DIR, class_name)

        # Only process actual folders
        if not os.path.isdir(class_path):
            continue

        images = [img for img in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img))]
        random.shuffle(images)

        split_idx = int(len(images) * (1 - TEST_RATIO))
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        # Create class folders
        os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, class_name), exist_ok=True)

        # Copy images
        for img in tqdm(train_images, desc=f"Train - {class_name}"):
            shutil.copy2(os.path.join(class_path, img), os.path.join(TRAIN_DIR, class_name, img))

        for img in tqdm(test_images, desc=f"Test - {class_name}"):
            shutil.copy2(os.path.join(class_path, img), os.path.join(TEST_DIR, class_name, img))

    print(f" Dataset split complete. Check '{BASE_OUTPUT_DIR}' for train/test folders.")

create_split_folders()
