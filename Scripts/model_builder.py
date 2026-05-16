import torch
import torch.nn as nn
from torchvision import models


class RadiologyEfficientNet(nn.Module):
    """
    Advanced deep feature extraction and classification network.
    Integrates an EfficientNet-B0 backbone utilizing pre-trained ImageNet parameters
    with a custom multi-layer regularization classification head.
    """

    def __init__(self, num_classes=2, pretrained=True):
        super(RadiologyEfficientNet, self).__init__()

        # Feature extraction backbone initialization
        if pretrained:
            self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.base_model = models.efficientnet_b0(weights=None)

        # Extract output feature dimensions from final convolutional bottleneck layer
        num_ftrs = self.base_model.classifier[1].in_features

        # Custom diagnostic head design incorporating dual regularization layers
        # Stacking order matches existing weights parameters configuration precisely
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Executes structural forward pass propagation through the network pipeline.
        """
        return self.base_model(x)


def get_clinical_criterion(class_counts, device=None):
    """
    Computes a class-weighted Cross-Entropy Loss optimization target.
    Utilizes inverse frequency scaling to balance data distribution asymmetry.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inverse frequency allocation formula: Weight = 1.0 / Count
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

    # Normalize weight parameters to prevent erratic gradient variances
    weights = weights / weights.sum()
    weights = weights.to(device)

    return nn.CrossEntropyLoss(weight=weights)


def initialize_advanced_model(num_classes=2, device=None):
    """
    Orchestrates architectural instantiation, parameter allocation,
    and target runtime device mapping.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RadiologyEfficientNet(num_classes=num_classes)
    model = model.to(device)

    print(f"Pipeline System: Advanced architecture initialized successfully on hardware target: {device}")
    return model


if __name__ == "__main__":
    # Functional validation loop block
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_model = initialize_advanced_model(num_classes=2, device=current_device)