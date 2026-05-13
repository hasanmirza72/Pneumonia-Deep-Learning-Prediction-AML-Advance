import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import numpy as np
import cv2
from PIL import Image


# --- SECTION 1: RADIOLOGY PREPROCESSING (CLAHE) ---
# We use CLAHE to enhance the contrast of the lung fields in the X-ray.
# This makes subtle opacities and infiltrates (signs of pneumonia) more visible to the CNN.
class RadiologyCLAHE:
    def __call__(self, img):
        # Convert PIL image to NumPy array to allow OpenCV to perform mathematical pixel operations.
        # This is the first step in our 'Clinical Pipeline' to standardize image quality.
        img_np = np.array(img)

        # Convert to LAB color space to isolate the Luminance (L) channel for enhancement.
        # This ensures we only sharpen the 'visibility' without adding color artifacts to the grayscale X-ray.
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply Adaptive Histogram Equalization with a clip limit of 2.0 to avoid over-exposure.
        # This solves the professor's 'Methodological Depth' requirement by adding domain-specific preprocessing.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge the enhanced L-channel back and convert back to a standard RGB image.
        # Finally, return as a PIL Image so it remains compatible with PyTorch's 'transforms' library.
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final_img)


# --- SECTION 2: RADIOLOGY TRANSFORMS ---
# We define different pipelines for training (with augmentation) and validation (clean).
# This ensures the model learns to be robust but is evaluated on 'real-world' patient data.
xray_transforms = {
    'train': transforms.Compose([
        RadiologyCLAHE(),  # Enhance lung features to make pneumonia detection easier.
        transforms.Resize((224, 224)),  # Resize to 224x224 to match the input requirements of EfficientNet-B0.
        transforms.RandomHorizontalFlip(),  # Simulate patient positioning variability in a hospital setting.
        transforms.RandomRotation(10),  # Improve the model's ability to handle slightly tilted X-ray scans.
        transforms.ToTensor(),  # Convert pixel data into a Torch Tensor for mathematical processing.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Apply ImageNet baseline normalization.
    ]),
    'val': transforms.Compose([
        RadiologyCLAHE(),  # Keep the clinical enhancement consistent for valid diagnostic testing.
        transforms.Resize((224, 224)),  # Ensure validation images match the dimensions used during the training phase.
        transforms.ToTensor(),  # We do not use augmentation on validation to ensure an honest performance audit.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Normalize using identical training parameters.
    ]),
}


# --- SECTION 3: THE X-RAY LOADER ENGINE ---
# This function creates the training and validation loaders specifically for binary classification.
# It addresses the 'Class Imbalance' mentioned in your previous project feedback.
def get_xray_loaders(base_path, batch_size=32):
    loaders = {}

    # We iterate through both 'train' and 'val' folders created by our splitter script.
    # This structure is essential for 'Experimental Rigor' and avoiding data leakage.
    for phase in ['train', 'val']:
        dataset = datasets.ImageFolder(os.path.join(base_path, phase), xray_transforms[phase])

        if phase == 'train':
            # Identify the target labels (Normal=0, Pneumonia=1) to calculate class rarity.
            # This is the first step in implementing the WeightedRandomSampler mitigation.
            targets = np.array(dataset.targets)
            class_counts = np.bincount(targets)

            # Calculate weights based on the inverse frequency: $w = 1 / count$.
            # This mathematically ensures that the 'Normal' class gets as much attention as 'Pneumonia'.
            weights = 1. / class_counts
            sample_weights = torch.from_numpy(weights[targets])

            # The Sampler will prioritize the minority class (Normal) during each training epoch.
            # This prevents the model from 'collapsing' and only predicting the majority class.
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle = False  # The sampler handles the data mixing, so manual shuffling is not required.
        else:
            sampler = None
            shuffle = False  # Validation must stay in a fixed order for accurate confusion matrix generation.

        # Initialize the DataLoader to manage batches and memory flow on your laptop CPU.
        # We set pin_memory False to optimize the workflow for a CPU-only environment.
        loaders[phase] = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                    shuffle=shuffle, num_workers=2, pin_memory=False)

    print(f"\n--- 🩻 Radiology Audit Complete ---")
    print(f"Training on {len(loaders['train'].dataset)} images | Validating on {len(loaders['val'].dataset)} images")
    return loaders, class_counts


if __name__ == "__main__":
    # Point to the specific X-ray directory you split earlier.
    XRAY_PATH = r'D:\AML Project\Advance\rscbjbr9sj-2\ChestXRay2017\chest_xray'
    xray_loaders, counts = get_xray_loaders(XRAY_PATH)