import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- SECTION 1: BASELINE IMAGE TRANSFORMS (UNOPTIMIZED COHORT) ---
# Standard structural sizing pipeline omitting adaptive histogram contrast
# equalization and stochastic data augmentation variants.
baseline_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def get_baseline_loaders(base_path, batch_size=32):
    """
    Instantiates PyTorch DataLoaders using un-enhanced raw graphics matrices.
    Intentionally omits class-balancing samplers to establish an empirical
    performance floor baseline.
    """
    loaders = {}

    for phase in ['train', 'val']:
        phase_path = os.path.join(base_path, phase)
        if not os.path.exists(phase_path):
            raise FileNotFoundError(f"Missing empirical data target mapping for: {phase_path}")

        dataset = datasets.ImageFolder(phase_path, baseline_transforms[phase])

        if phase == 'train':
            # Implements basic standard uniform distribution parameters
            # without class rarity sampling corrections.
            sampler = None
            shuffle = True
        else:
            sampler = None
            shuffle = False

        loaders[phase] = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=False
        )

    print(f"\n--- Baseline Image Engine: Raw Data Pipelines Initialized ---")
    print(f"Dataset Audit: {len(loaders['train'].dataset)} training instances registered.")
    return loaders, None


if __name__ == "__main__":
    # Resolve execution directory pathing dynamically to ensure cross-platform compatibility
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Priority 1: Standard relative repository directory architecture
    XRAY_PATH = os.path.join(script_dir, 'ChestXRay2017', 'chest_xray')

    # Priority 2: Fallback to specific local environment variables
    if not os.path.exists(XRAY_PATH):
        XRAY_PATH = r'D:\AML Project\Advance\rscbjbr9sj-2\ChestXRay2017\chest_xray'

    if os.path.exists(XRAY_PATH):
        baseline_loaders, _ = get_baseline_loaders(XRAY_PATH)
    else:
        print(f"Execution Error: Missing target clinical directory mapping at: {XRAY_PATH}")