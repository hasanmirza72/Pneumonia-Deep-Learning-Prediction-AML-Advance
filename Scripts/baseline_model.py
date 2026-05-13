import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=2):
# This class defines a standard 3-layer Convolutional Neural Network.
# It serves as the 'Naive Baseline' to compare against the Advanced EfficientNet.
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
# The first layer extracts 16 basic visual features like edges and lines.
# MaxPool reduces the image size by half to keep the computation manageable on a CPU.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
# These layers gradually increase complexity, identifying shapes and textures.
# We cap the depth at 64 filters to keep the model lightweight and fast.
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
# The final layers flatten the 2D features into a 1D vector for classification.
# It maps the identified visual patterns to the 'Normal' or 'Pneumonia' labels.

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
# Data flows through the layers, applying the ReLU activation to introduce non-linearity.
# The pooling layers ensure the model focuses on the most prominent visual signals.
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
# The 3rd layer finishes the feature extraction before the data is flattened.
# This 'view' command prepares the multi-dimensional image data for the linear classifier.
        x = self.fc2(self.fc1(x))
        return x
# The final output returns the raw 'logits' which represent the prediction scores.
# These will be converted into probabilities by the training engine later.