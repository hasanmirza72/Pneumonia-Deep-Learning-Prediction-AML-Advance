import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    A standard 3-layer Convolutional Neural Network (CNN) architecture.
    Serves as an unoptimized performance floor baseline to evaluate the
    clinical gain of advanced transfer-learning networks.
    """

    def __init__(self, num_classes=2):
        super(BaselineCNN, self).__init__()

        # Spatial Feature Extraction Layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Linear Classification Head
        # Input dimension (64 * 28 * 28) corresponds to a standard 224x224x3 image
        # down-sampled sequentially through three 2x2 max-pooling operations.
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Executes the forward pass through the feature extraction stages
        and fully connected classification head layers.
        """
        # Sequential feature map processing
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten multi-dimensional feature tensors into a 1D vector
        x = x.view(x.size(0), -1)

        # Forward pass through classification layers
        # Uses standard functional non-linearity between linear transforms
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x