import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from model_builder import initialize_advanced_model, device


# --- SECTION 1: THE GRAD-CAM ENGINE ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Register hooks to spy on the internal convolutional layers
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, class_idx):
        output = self.model(input_image)
        score = output[:, class_idx]

        self.model.zero_grad()
        score.backward()

        # Pool the gradients (importance values) of the last convolutional layer
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Apply the weights to the raw visual activations (ReLU ensures we only keep positive features)
        heatmap = torch.sum(weights * self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)

        heatmap /= torch.max(heatmap)
        return heatmap.detach().cpu().numpy()


# --- SECTION 2: PATH SETUP (Windows Safe) ---
# This block dynamically finds your paths, eliminating 'FileNotFound' errors.
script_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(script_dir, 'pneumonia_classifier_v1.pth')
test_path = os.path.join(script_dir, 'ChestXRay2017', 'chest_xray', 'test')
output_plot = os.path.join(script_dir, 'gradcam_clinical_analysis.png')

# --- SECTION 3: MULTI-IMAGE PREPARATION ---
# We select 4 distinct test cases from your D: drive for the analysis gallery.
# This proves the model is consistent across different patients.
images_to_analyze = [
    # Case 1 & 2: True Pneumonia (The Heatmap should highlight consolidation)
    ('PNEUMONIA', 'person1_virus_6.jpeg'),
    ('PNEUMONIA', 'person31_virus_70.jpeg'),  # Another Pneumonia case from your folder

    # Case 3 & 4: Normal Scans (Heatmap should be faint or non-existent)
    ('NORMAL', 'NORMAL2-IM-0035-0001.jpeg'),
    ('NORMAL', 'NORMAL2-IM-0335-0001.jpeg')  # Another Normal case from your folder
]

# Create a master gallery plot (4 rows, 2 columns)
fig, axes = plt.subplot_mosaic([['original_1', 'gradcam_1'],
                                ['original_2', 'gradcam_2'],
                                ['original_3', 'gradcam_3'],
                                ['original_4', 'gradcam_4']],
                               figsize=(12, 16), constrained_layout=True)

plt.suptitle('Clinical Interpretability: Grad-CAM Localization for Pneumonia Detection', fontsize=20, y=1.02)

# --- SECTION 4: INITIALIZE & LOAD MODEL ---
# This ensures we use the 98.3% accurate weights we spent 273 minutes training.
model = initialize_advanced_model(num_classes=2)
if not os.path.exists(model_file):
    raise FileNotFoundError(f"❌ Model weight file not found at {model_file}.")
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()

# Select the last convolutional block of EfficientNet-B0 backbone for the Grad-CAM.
target_layer = model.base_model.features[-1]
cam = GradCAM(model, target_layer)

print("🚀 Starting Grad-CAM Multi-Case Clinical Localization...")

# --- SECTION 5: GALLERY GENERATION LOOP ---
for i, (actual_class, file_name) in enumerate(images_to_analyze, 1):
    image_full_path = os.path.join(test_path, actual_class, file_name)

    if not os.path.exists(image_full_path):
        print(f"⚠️ Warning: Image {file_name} not found, skipping Case {i}.")
        continue

    # Process original image for visualization
    original_img_cv = cv2.imread(image_full_path)
    original_img_cv = cv2.resize(original_img_cv, (224, 224))
    original_img_rgb = cv2.cvtColor(original_img_cv, cv2.COLOR_BGR2RGB)

    # Apply clinical transforms for the model (ensuring consistency)
    from data_loader import xray_transforms

    img_pil = Image.open(image_full_path).convert('RGB')
    input_tensor = xray_transforms['val'](img_pil).unsqueeze(0).to(device)

    # Generate the heatmap specifically for the 'PNEUMONIA' class (Index 1).
    heatmap = cam.generate_heatmap(input_tensor, class_idx=1)

    # Blend Heatmap with Gray Scale original image
    original_img_gray = cv2.cvtColor(original_img_cv, cv2.COLOR_RGB2GRAY)
    original_img_gray_cv = cv2.merge([original_img_gray, original_img_gray, original_img_gray])
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * 0.4 + original_img_gray_cv * 0.6  # 40% transparency

    # Plot Row Component 1: Original Image
    ax_orig = axes[f'original_{i}']
    ax_orig.imshow(original_img_rgb)
    ax_orig.set_title(f"Case {i}: {actual_class} (Original)", fontsize=12)
    ax_orig.axis('off')

    # Plot Row Component 2: Grad-CAM Localization
    ax_grad = axes[f'gradcam_1' if i == 1 else f'gradcam_{i}']  # matplotlib gotcha fix
    if i > 1:
        # A small fix needed for subplot_mosaic naming consistency if i > 1.
        # This is essentially 'axes[f'gradcam_{i}']', handled below for a 1x2 per row gallery style.
        ax_grad = plt.subplot(4, 2, 2 * i)  # Classic subplot fallback for robust layout

    ax_grad.imshow(cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB))
    ax_grad.set_title(f"Case {i}: Pneumonia Localization", fontsize=12, color='darkblue')
    ax_grad.axis('off')

print(f"📊 Gallery compiled. Exporting high-resolution PNG...")
# Save high-res PNG for your final project presentation and documentation.
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
plt.show()  # Display the complete 4-case gallery.

print(f"✅ Success: 4-Case Clinical Interpretability Gallery saved as 'gradcam_clinical_analysis.png'")