import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data_loader import get_xray_loaders
from model_builder import initialize_advanced_model

# Categorical mapping array for evaluation display
class_names = {0: "NORMAL", 1: "PNEUMONIA"}


def plot_and_save(data_list, title, filename, script_dir):
    """
    Constructs and serializes a multi-panel grid visualization showing
    clinical samples, target classes, and associated model confidence intervals.
    Automatically deallocates figure contexts to prevent memory overhead leaks.
    """
    if not data_list:
        print(f"ℹ️ Content Notice: Skipping {filename} generation due to empty dataset stream.")
        return

    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(title, fontsize=16, fontweight='bold')

    for i, item in enumerate(data_list[:12]):
        ax = plt.subplot(3, 4, i + 1)

        # Permute tensor dimensions back to standard image spatial dimensions (H, W, C)
        img = item['image'].permute(1, 2, 0).numpy()

        # De-normalize vectors using standard ImageNet distribution constants
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        plt.imshow(img.clip(0, 1))

        # Enforce distinct color bounds to visually highlight diagnostic anomalies
        color = 'red' if item['pred'] != item['label'] else 'black'
        ax.set_title(
            f"P: {class_names[item['pred']]}\n"
            f"A: {class_names[item['label']]}\n"
            f"C: {item['conf'] * 100:.1f}%",
            color=color,
            fontsize=9,
            fontweight='bold'
        )
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(script_dir, filename)

    # Serialize dashboard panel matrix to disk and flush figure thread resources
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 File Serialized Successfully: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    print("🏥 Starting Advanced Model Visual Diagnostic Evaluation Loop...")

    # Dynamically resolve host execution boundaries
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(script_dir, 'pneumonia_classifier_v1.pth')
    data_path = os.path.join(script_dir, 'ChestXRay2017', 'chest_xray')

    # Environment fallback checkpoint for specific host execution setups
    if not os.path.exists(data_path):
        data_path = r'D:\AML Project\Advance\rscbjbr9sj-2\ChestXRay2017\chest_xray'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Execution Error: Core data directory tree not resolved at target: {data_path}")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Execution Error: Missing fine-tuned weights file checkpoint at target: {model_file}")

    # Synchronize execution hardware targets
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model architecture and overlay optimization checkpoint parameters
    model = initialize_advanced_model(num_classes=2)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # Instantiate dataloaders with single-process constraints to ensure zero-hang pipeline profiling
    loaders, _ = get_xray_loaders(data_path)
    loaders['val'] = torch.utils.data.DataLoader(
        loaders['val'].dataset, batch_size=32, shuffle=False, num_workers=0
    )

    misclassifications = []
    random_samples = []

    # Deactivate gradient history engine for validation profiling passes
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loaders['val']):
            print(f"Processing Batch {batch_idx + 1}...", end='\r')

            logits = model(images.to(device))
            probs = F.softmax(logits, dim=1)
            conf, preds = torch.max(probs, 1)

            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()
            conf = conf.cpu().numpy()
            images = images.cpu()

            for i in range(len(labels)):
                # Segregate classification anomalies for targeted error analysis
                if preds[i] != labels[i]:
                    misclassifications.append({
                        'image': images[i], 'pred': preds[i], 'label': labels[i], 'conf': conf[i]
                    })

                # Maintain a randomized distribution baseline gallery reference
                if len(random_samples) < 12:
                    random_samples.append({
                        'image': images[i], 'pred': preds[i], 'label': labels[i], 'conf': conf[i]
                    })

    print(f"\n✅ Diagnostic Scan Complete! Detected {len(misclassifications)} classification errors.")

    # Orchestrate reporting asset serialization sequences
    print("📊 Exporting Evaluative Visual Dashboards...")
    plot_and_save(random_samples, "General Clinical Audit", "clinical_diagnostic_gallery.png", script_dir)
    plot_and_save(misclassifications, "Error Analysis: Misclassifications", "misclassification_report.png", script_dir)

    print("\n--- Pipeline Diagnostic Evaluation Module Terminated Cleanly ---")