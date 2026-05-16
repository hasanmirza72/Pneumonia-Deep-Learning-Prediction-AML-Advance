import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


class RadiologyCLAHE:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to isolate and optimize local contrast fields within the pulmonary structure
    without amplifying high-frequency background imaging artifacts.
    """
    def __call__(self, img):
        # Convert PIL asset to array configuration for OpenCV processing
        img_np = np.array(img)

        # Map to LAB color space to process the luminance (L) channel independently
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Enforce adaptive histogram equalization limiting over-exposure distortion
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Recompose color space architecture and restore original PIL data format
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final_img)


# --- RADIOLOGY PIPELINE TRANSFORMS ---
# Training targets utilize stochastic transformations to induce geometric invariance.
# Validation targets track static real-world metrics using non-augmented arrays.
xray_transforms = {
    'train': transforms.Compose([
        RadiologyCLAHE(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # Simulates spatial patient acquisition variances
        transforms.RandomRotation(10),           # Enhances network robust response to axis tilts
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        RadiologyCLAHE(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def get_xray_loaders(base_path, batch_size=32):
    """
    Constructs PyTorch DataLoaders optimized for skewed categorical allocations.
    Implements inverse class frequency sampling to stabilize gradient updates.
    """
    loaders = {}

    for phase in ['train', 'val']:
        phase_path = os.path.join(base_path, phase)
        if not os.path.exists(phase_path):
            raise FileNotFoundError(f"Data Target Error: Directory missing at: {phase_path}")

        dataset = datasets.ImageFolder(phase_path, xray_transforms[phase])

        if phase == 'train':
            # Compute inverse categorical frequencies to mitigate training distribution bias
            targets = np.array(dataset.targets)
            class_counts = np.bincount(targets)

            # Weight mapping formula: w = 1.0 / target_class_count
            weights = 1.0 / class_counts
            sample_weights = torch.from_numpy(weights[targets])

            # Instantiate sampler tracking rarity density profiles across mini-batches
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False  # Mutually exclusive parameter context with sampler deployment
        else:
            sampler = None
            shuffle = False

        # Allocate data tensor batches into system memory loaders
        loaders[phase] = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=False
        )

    print(f"\n--- Advanced Image Engine: Clinical Pipelines Instantiated ---")
    print(f"Cohort Breakdown: Training [{len(loaders['train'].dataset)}] | Validation [{len(loaders['val'].dataset)}]")
    return loaders, class_counts


if __name__ == "__main__":
    # Dynamically evaluate directory workspace parameters to achieve immediate cross-platform execution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    XRAY_PATH = os.path.join(script_dir, 'ChestXRay2017', 'chest_xray')

    # Emergency environment fallback structure for specific host local machines
    if not os.path.exists(XRAY_PATH):
        XRAY_PATH = r'D:\AML Project\Advance\rscbjbr9sj-2\ChestXRay2017\chest_xray'

    if os.path.exists(XRAY_PATH):
        xray_loaders, counts = get_xray_loaders(XRAY_PATH)
    else:
        print(f"Execution Failure: Target radiography data trees not located at pathing bounds: {XRAY_PATH}")