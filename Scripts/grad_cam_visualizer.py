import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from data_loader import xray_transforms
from model_builder import initialize_advanced_model


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) engine.
    Registers forward and backward hooks onto a targeted convolutional layer
    to capture activation states and spatial gradients for feature localization.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register structural hooks to extract feature map tensors during passes
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, class_idx):
        """
        Backpropagates the target score gradient to compute spatial importance maps.
        """
        output = self.model(input_image)
        score = output[:, class_idx]

        self.model.zero_grad()
        score.backward()

        # Global average pool the gradients to determine channel-wise importance coefficients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Compute the weighted linear combination of forward activation maps
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()

        # Apply ReLU to isolate features with positive clinical correlation to the target class
        heatmap = F.relu(heatmap)

        # Normalize the localization bounds
        heatmap_max = torch.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max

        return heatmap.detach().cpu().numpy()


if __name__ == "__main__":
    print("🚀 Initializing Grad-CAM Multi-Case Clinical Localization Phase...")

    # Dynamically resolve directory tree boundaries for absolute tracking
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(script_dir, 'pneumonia_classifier_v1.pth')
    test_path = os.path.join(script_dir, 'ChestXRay2017', 'chest_xray', 'test')
    output_plot = os.path.join(script_dir, 'gradcam_clinical_analysis.png')

    # Environment fallback checkpoints for specific local drive paths
    if not os.path.exists(test_path):
        test_path = r'D:\AML Project\Advance\rscbjbr9sj-2\ChestXRay2017\chest_xray\test'

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Execution Error: Target test directory matrix not resolved: {test_path}")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Execution Error: Missing optimization weights checkpoint: {model_file}")

    # Explicitly configure target hardware acceleration frameworks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define validation tracking targets across categorical image instances
    images_to_analyze = [
        ('PNEUMONIA', 'person1_virus_6.jpeg'),
        ('PNEUMONIA', 'person31_virus_70.jpeg'),
        ('NORMAL', 'NORMAL2-IM-0035-0001.jpeg'),
        ('NORMAL', 'NORMAL2-IM-0335-0001.jpeg')
    ]

    # Instantiate structural layout mapping matrices
    fig, axes = plt.subplot_mosaic([
        ['original_1', 'gradcam_1'],
        ['original_2', 'gradcam_2'],
        ['original_3', 'gradcam_3'],
        ['original_4', 'gradcam_4']
    ], figsize=(12, 16), constrained_layout=True)

    plt.suptitle('Clinical Interpretability: Grad-CAM Feature Localization Audit', fontsize=20, y=1.02,
                 fontweight='bold')

    # Initialize fine-tuned model and map parameters to target device
    model = initialize_advanced_model(num_classes=2)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # Isolate the final convolutional feature extractor block of the EfficientNet-B0 backbone
    target_layer = model.base_model.features[-1]
    cam = GradCAM(model, target_layer)

    for i, (actual_class, file_name) in enumerate(images_to_analyze, 1):
        image_full_path = os.path.join(test_path, actual_class, file_name)

        if not os.path.exists(image_full_path):
            print(f"Notice: Verification case vector {file_name} not found in directory. Skipping slot.")
            continue

        # Load and resize matrix for standard spatial alignment
        original_img_cv = cv2.imread(image_full_path)
        original_img_cv = cv2.resize(original_img_cv, (224, 224))
        original_img_rgb = cv2.cvtColor(original_img_cv, cv2.COLOR_BGR2RGB)

        # Preprocess asset using standard clinical evaluation pipelines
        img_pil = Image.open(image_full_path).convert('RGB')
        input_tensor = xray_transforms['val'](img_pil).unsqueeze(0).to(device)

        # Generate localization heatmap focused on the PNEUMONIA classification index (1)
        heatmap = cam.generate_heatmap(input_tensor, class_idx=1)

        # Blend localization overlay with original grayscale dimensions
        original_img_gray = cv2.cvtColor(original_img_cv, cv2.COLOR_RGB2GRAY)
        original_img_gray_cv = cv2.merge([original_img_gray, original_img_gray, original_img_gray])
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # Linear combination formulation enforcing a 40% transparency layer overlay
        superimposed_img = heatmap_color * 0.4 + original_img_gray_cv * 0.6

        # Plot Sub panel component 1: Structural Original Frame
        ax_orig = axes[f'original_{i}']
        ax_orig.imshow(original_img_rgb)
        ax_orig.set_title(f"Case {i}: {actual_class} (Ground Truth Reference)", fontsize=12)
        ax_orig.axis('off')

        # Plot Sub panel component 2: Superimposed Grad-CAM Activation Array
        ax_grad = axes[f'gradcam_{i}']
        ax_grad.imshow(cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB))
        ax_grad.set_title(f"Case {i}: Pneumonia Feature Activation Mapping", fontsize=12, color='darkblue')
        ax_grad.axis('off')

    # Serialize composite visual audit matrix to disk asset files
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Success: Explainable AI Interpretability Gallery exported cleanly to: {output_plot}")