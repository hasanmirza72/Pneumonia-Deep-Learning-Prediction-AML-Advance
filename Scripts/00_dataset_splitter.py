import os
import shutil
from sklearn.model_selection import train_test_split


def split_medical_dataset(base_path, class_list):
    """
    Executes a deterministic stratified partitioning of a clinical dataset
    to establish isolated training and validation cohorts.
    """
    print(f"\n--- Initializing Stratified Partitioning for: {os.path.basename(base_path)} ---")

    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'val')

    # Ensure validation subdirectories exist for each diagnostic target
    for cls in class_list:
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    # Process each class cohort independently to preserve distribution ratios
    for cls in class_list:
        cls_path = os.path.join(train_dir, cls)

        if not os.path.exists(cls_path):
            print(f"Warning: Directory target {cls_path} not found. Skipping cohort.")
            continue

        # Filter and log files with standard clinical graphics extensions
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

        if len(images) == 0:
            print(f"Notice: Cohort target {cls} contains no valid image data formats.")
            continue

        # Execute 80/20 stratified split. Seed random_state to guarantee experimental reproducibility.
        train_imgs, val_imgs = train_test_split(images, test_size=0.20, random_state=42)

        print(f"Class [{cls}]: Allocating {len(val_imgs)} samples to validation registry.")

        # Physically migrate validation vectors to isolate sets and prevent data leakage
        for img in val_imgs:
            src = os.path.join(cls_path, img)
            dst = os.path.join(val_dir, cls, img)
            shutil.move(src, dst)


if __name__ == "__main__":
    # Resolve absolute pathing context dynamically to support cross-platform deployment
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Priority 1: Check standard repository layout relative path
    xray_path = os.path.join(script_dir, 'ChestXRay2017', 'chest_xray')

    # Priority 2: Fallback to local host environment variables if run on local drive
    if not os.path.exists(xray_path):
        xray_path = r'D:\AML Project\Advance\rscbjbr9sj-2\ChestXRay2017\chest_xray'

    xray_classes = ['NORMAL', 'PNEUMONIA']

    if os.path.exists(xray_path):
        split_medical_dataset(xray_path, xray_classes)
        print("\n--- Stratified Dataset Splits Executed Successfully ---")
    else:
        print(f"Execution Error: Missing dataset directory profile at target path mapping: {xray_path}")