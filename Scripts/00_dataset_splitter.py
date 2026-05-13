import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split


# --- SECTION 1: THE REUSABLE SPLIT FUNCTION ---
# Instead of writing the code twice, we create a 'function'.
# This shows 'Advanced Code Quality' because the code is modular and clean.
def split_medical_dataset(base_path, class_list):
    print(f"\n--- Starting Split for: {os.path.basename(base_path)} ---")

    # Define the training and validation paths inside the function.
    # This prevents the variables from being overwritten by other datasets.
    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'val')

    # Step 1: Create the validation folders for each class.
    # We use exist_ok=True so the code doesn't crash if you run it twice.
    for cls in class_list:
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    # Step 2: Loop through each class to perform the 80/20 split.
    # This ensures every category (like DME or Normal) is split perfectly.
    for cls in class_list:
        cls_path = os.path.join(train_dir, cls)

        # Check if the folder exists to avoid errors.
        if not os.path.exists(cls_path):
            print(f"Warning: Folder {cls_path} not found. Skipping...")
            continue

        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

        # If the folder is empty, skip it.
        if len(images) == 0:
            continue

        # Step 3: Use train_test_split for a scientific 20% validation set.
        # random_state=42 ensures you can reproduce your exact results.
        train_imgs, val_imgs = train_test_split(images, test_size=0.20, random_state=42)

        print(f"Class {cls}: Moving {len(val_imgs)} images to validation.")

        # Step 4: Physically move the files.
        # This keeps the 'train' folder for learning and 'val' for testing.
        for img in val_imgs:
            src = os.path.join(cls_path, img)
            dst = os.path.join(val_dir, cls, img)
            shutil.move(src, dst)


# --- SECTION 2: EXECUTION BLOCK ---
# Now we simply 'call' the function twice with the specific details for project.
if __name__ == "__main__":
    # 1. Run for Chest X-Ray
    xray_path = r'D:\AML Project\Advance\rscbjbr9sj-2\ChestXRay2017\chest_xray'
    xray_classes = ['NORMAL', 'PNEUMONIA']
    split_medical_dataset(xray_path, xray_classes)

    print("\n--- All Stratified Splits Complete ---")