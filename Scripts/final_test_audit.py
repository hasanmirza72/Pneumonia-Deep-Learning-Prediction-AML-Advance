import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torchvision import datasets
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
from model_builder import initialize_advanced_model, device
from data_loader import xray_transforms

# --- SECTION 1: STABLE PATH CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(script_dir, 'pneumonia_classifier_v1.pth')
test_data_path = os.path.join(script_dir, 'ChestXRay2017', 'chest_xray', 'test')

# --- SECTION 2: PROTECTED EXECUTION ---
if __name__ == "__main__":
    print("🏥 Initializing Final Clinical Test Audit...")

    # 1. Load the Test Dataset (Unseen Data)
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=xray_transforms['val'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 2. Load the Trained Model
    model = initialize_advanced_model(num_classes=2)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    # 3. Perform Inference
    print(f"🔍 Analyzing {len(test_dataset)} images in the Test folder...")
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Calculate Comprehensive Clinical Metrics
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'Accuracy': accuracy_score(all_labels, all_preds),
        'Precision': precision_score(all_labels, all_preds),
        'Recall (Sens.)': recall_score(all_labels, all_preds),
        'Specificity': tn / (tn + fp),
        'F1-Score': f1_score(all_labels, all_preds),
        'MCC': matthews_corrcoef(all_labels, all_preds)
    }

    # 5. Generate Visual Performance Dashboard
    # We use 'Agg' backend implicitly by calling close() to prevent the script from hanging.
    fig = plt.figure(figsize=(16, 8), facecolor='#f8f9fa')
    plt.suptitle('FINAL TEST AUDIT: Unseen Clinical Data Performance',
                 fontsize=22, fontweight='bold', color='#2c3e50', y=0.96)

    # Subplot 1: Confusion Matrix
    ax1 = plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                annot_kws={'size': 16, 'fontweight': 'bold'},
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    ax1.set_title('Test Set: Confusion Matrix', fontsize=16, pad=20)
    ax1.set_xlabel('Predicted Diagnosis', fontsize=12)
    ax1.set_ylabel('Actual Clinical Label', fontsize=12)

    # Subplot 2: Metric Scorecard
    ax2 = plt.subplot(1, 2, 2)
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = ['#27ae60' if v > 0.85 else '#e67e22' for v in values]

    bars = ax2.barh(names, values, color=colors, height=0.6)
    ax2.set_xlim(0, 1.1)
    ax2.set_title('Diagnostic Performance Metrics (Test Set)', fontsize=16, pad=20)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{width * 100:.1f}%' if names[i] != 'MCC' else f'{width:.3f}',
                 va='center', fontweight='bold', fontsize=12)

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.92])

    # SAVE AND CLOSE (The fix for the hanging issue)
    output_path = os.path.join(script_dir, 'final_test_scorecard.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n✅ Audit Complete! Results saved to: {output_path}")
    print("--- 🏁 Script Terminated Successfully ---")