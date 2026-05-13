import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np

# --- SECTION 1: STANDARD TRANSFORMS (NO CLINICAL ENHANCEMENT) ---
# We have REMOVED the RadiologyCLAHE class here.
# Without contrast enhancement, the baseline model will struggle to 'see' through
# the low-contrast tissue of the raw X-ray.
baseline_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        # We also removed RandomHorizontalFlip and Rotation to make the model
        # It is more prone to 'overfitting' on the training set's specific orientations.
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- SECTION 2: THE RAW DATA ENGINE ---
# This version removes the WeightedRandomSampler.
# Because the dataset is imbalanced (more Pneumonia than Normal), the model
# will likely learn a bias toward the majority class.
def get_baseline_loaders(base_path, batch_size=32):
    loaders = {}

    for phase in ['train', 'val']:
        # We point to the same split folders but use the 'baseline' transforms.
        dataset = datasets.ImageFolder(os.path.join(base_path, phase), baseline_transforms[phase])

        if phase == 'train':
            # We use standard shuffle=True instead of a WeightedSampler.
            sampler = None
            shuffle = True
        else:
            sampler = None
            shuffle = False

        loaders[phase] = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                    shuffle=shuffle, num_workers=2, pin_memory=False)

    print(f"\n--- ⚠️ Baseline Data Audit: No Enhancement Applied ---")
    print(f"Feeding RAW data: {len(loaders['train'].dataset)} training images.")
    return loaders, None

if __name__ == "__main__":
    # Ensure this path matches your project structure on the D: drive.
    XRAY_PATH = r'D:\AML Project\Advance\rscbjbr9sj-2\ChestXRay2017\chest_xray'
    baseline_loaders = get_baseline_loaders(XRAY_PATH)
