import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
from model_builder import initialize_advanced_model
from data_loader import xray_transforms

if __name__ == "__main__":
    print("🏥 Initializing Advanced Model Final Clinical Test Audit...")

    # Dynamically resolve directory tree boundaries for tracking validation resources
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(script_dir, 'pneumonia_classifier_v1.pth')
    test_data_path = os.path.join(script_dir, 'ChestXRay2017', 'chest_xray', 'test')

    # Environment fallback check for specific host execution setups
    if not os.path.exists(test_data_path):
        test_data_path = r'D:\AML Project\Advance\rscbjbr9sj-2\ChestXRay2017\chest_xray\test'

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Execution Error: Clinical test partition missing at target: {test_data_path}")

    # Synchronize targeted hardware acceleration frameworks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load contrast-enhanced test vectors using validation transformation definitions
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=xray_transforms['val'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Instantiate advanced architecture and transfer fine-tuned parameter weights
    model = initialize_advanced_model(num_classes=2)
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Execution Error: Missing fine-tuned weights file at path: {model_file}")

    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    print(f"🔍 Executing inference evaluation over {len(test_dataset)} out-of-sample images...")

    # Deactivate gradient history updates to optimize system resources during prediction passes
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compile categorical confusion components
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # Structure target metrics summary log entries
    metrics = {
        'Accuracy': accuracy_score(all_labels, all_preds),
        'Precision': precision_score(all_labels, all_preds),
        'Recall (Sens.)': recall_score(all_labels, all_preds),
        'Specificity': tn / (tn + fp),
        'F1-Score': f1_score(all_labels, all_preds),
        'MCC': matthews_corrcoef(all_labels, all_preds)
    }

    # Generate professional statistical performance dashboard panel
    fig = plt.figure(figsize=(16, 8), facecolor='#f8f9fa')
    plt.suptitle('FINAL TEST AUDIT: Unseen Clinical Data Performance',
                 fontsize=22, fontweight='bold', color='#2c3e50', y=0.96)

    # Subplot 1: Confusion Matrix Visualization
    ax1 = plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                annot_kws={'size': 16, 'fontweight': 'bold'},
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    ax1.set_title('Test Set: Confusion Matrix', fontsize=16, pad=20)
    ax1.set_xlabel('Predicted Diagnosis', fontsize=12)
    ax1.set_ylabel('Actual Clinical Label', fontsize=12)

    # Subplot 2: Metric Performance Scorecard Horizontal Dashboard
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

    # Save target visualizations and clear plot resources to maintain memory bounds
    output_path = os.path.join(script_dir, 'final_test_scorecard.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n✅ Audit Complete! Performance scorecard exported to: {output_path}")
    print("--- 🏁 Evaluation Module Safely Terminated ---")