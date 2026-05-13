import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torchvision import datasets
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
# Import your baseline architecture and clinical data settings
from baseline_model import BaselineCNN
from data_loader import xray_transforms

# --- SECTION 1: TARGETING THE TEST DATA ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(script_dir, 'baseline_model.pth')
test_data_path = os.path.join(script_dir, 'ChestXRay2017', 'chest_xray', 'test')

if __name__ == "__main__":
    # 1. Load the same UNSEEN data used for the model
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=xray_transforms['val'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 2. Load the Baseline 'Brain'
    model = BaselineCNN(num_classes=2)
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()

    all_preds, all_labels = [], []
    print(f"🏥 Auditing Baseline Model on {len(test_dataset)} test images...")

    # 3. Inference Loop
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Calculate Metrics
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

    # 5. Generate the Comparison Scorecard
    fig = plt.figure(figsize=(16, 8), facecolor='#fdfcf0')  # Different color to distinguish from Advanced
    plt.suptitle('BASELINE AUDIT: Standard CNN Performance (Unseen Data)',
                 fontsize=22, fontweight='bold', color='#7f8c8d', y=0.96)

    # Subplot 1: Confusion Matrix
    ax1 = plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
                annot_kws={'size': 16, 'fontweight': 'bold'},
                xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'])
    ax1.set_title('Baseline: Confusion Matrix', fontsize=16, pad=20)

    # Subplot 2: Metric Dashboard
    ax2 = plt.subplot(1, 2, 2)
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = ['#e67e22' for _ in values]  # Orange theme for Baseline

    bars = ax2.barh(names, values, color=colors, height=0.6)
    ax2.set_xlim(0, 1.1)
    ax2.set_title('Baseline Performance Metrics', fontsize=16, pad=20)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{width * 100:.1f}%' if names[i] != 'MCC' else f'{width:.3f}',
                 va='center', fontweight='bold', fontsize=12)

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.92])
    output_path = os.path.join(script_dir, 'baseline_test_scorecard.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Ensures script finishes cleanly

    print(f"✅ Baseline Audit Complete. Scorecard saved as: {output_path}")
