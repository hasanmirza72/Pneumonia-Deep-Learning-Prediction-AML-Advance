import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from data_loader import get_xray_loaders
from model_builder import initialize_advanced_model, device

# --- 1. SETTINGS & LABELS ---
class_names = {0: "NORMAL", 1: "PNEUMONIA"}


def plot_and_save(data_list, title, filename, script_dir):
    """Saves the clinical gallery and closes to prevent memory hang."""
    if not data_list:
        print(f"ℹ️ Skipping {filename}: No data to display.")
        return

    # Use a smaller figure size to save RAM
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(title, fontsize=16)

    for i, item in enumerate(data_list[:12]):
        ax = plt.subplot(3, 4, i + 1)
        # Convert back to viewable image
        img = item['image'].permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        plt.imshow(img.clip(0, 1))

        color = 'red' if item['pred'] != item['label'] else 'black'
        ax.set_title(f"P: {class_names[item['pred']]}\nA: {class_names[item['label']]}\nC: {item['conf'] * 100:.1f}%",
                     color=color, fontsize=9)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(script_dir, filename)
    plt.savefig(save_path, dpi=150)  # Lower DPI for faster saving
    print(f"💾 File Saved: {save_path}")
    plt.close(fig)  # CRITICAL: Freezes memory immediately after saving


# --- 2. PROTECTED MAIN ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(script_dir, 'pneumonia_classifier_v1.pth')
    data_path = os.path.join(script_dir, 'ChestXRay2017', 'chest_xray')

    # Load Model
    model = initialize_advanced_model(num_classes=2)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # Load Data with 0 WORKERS to prevent Windows hang
    # Setting num_workers=0 is the 'Safe Mode' for diagnostic scripts.
    loaders, _ = get_xray_loaders(data_path)
    # Re-initialize the loader locally to override any background worker settings
    loaders['val'] = torch.utils.data.DataLoader(
        loaders['val'].dataset, batch_size=32, shuffle=False, num_workers=0
    )

    misclassifications = []
    random_samples = []

    print("🏥 Starting Diagnostic Run (Safe Mode)...")

    with torch.no_grad():
        # Iterate manually with a simple counter
        for batch_idx, (images, labels) in enumerate(loaders['val']):
            # Print update for EVERY batch (every 32 images)
            print(f"Processing Batch {batch_idx + 1}...", end='\r')

            logits = model(images.to(device))
            probs = F.softmax(logits, dim=1)
            conf, preds = torch.max(probs, 1)

            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            conf = conf.cpu().numpy()
            images = images.cpu()

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassifications.append({
                        'image': images[i], 'pred': preds[i], 'label': labels[i], 'conf': conf[i]
                    })

                if len(random_samples) < 12:
                    random_samples.append({
                        'image': images[i], 'pred': preds[i], 'label': labels[i], 'conf': conf[i]
                    })

    print(f"\n✅ Scan Complete! Found {len(misclassifications)} errors.")

    # Generate Reports
    print("📊 Exporting Visuals...")
    plot_and_save(random_samples, "General Clinical Audit", "clinical_diagnostic_gallery.png", script_dir)
    plot_and_save(misclassifications, "Error Analysis: Misclassifications", "misclassification_report.png", script_dir)

    print("\n--- DONE ---")