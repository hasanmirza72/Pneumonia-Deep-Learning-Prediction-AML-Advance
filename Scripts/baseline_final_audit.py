import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
from baseline_model import BaselineCNN
from data_loader_baseline import baseline_transforms

if __name__ == "__main__":
    # Dynamically resolve host execution paths for absolute file location tracking
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(script_dir, 'baseline_model.pth')
    test_data_path = os.path.join(script_dir, 'ChestXRay2017', 'chest_xray', 'test')

    # Fallback to local drive context if directory tree deviates from standard layout
    if not os.path.exists(test_data_path):
        test_data_path = r'D:\AML Project\Advance\rscbjbr9sj-2\ChestXRay2017\chest_xray\test'

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Execution Error: Test data target matrix not located at: {test_data_path}")

    print(f"🏥 Initializing Empirical Baseline Evaluation on Unseen Test Partition...")

    # Hardware resource synchronization configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load un-enhanced test partition vectors to enforce clean experimental boundaries
    test_dataset = datasets.ImageFolder(root=test_data_path, transform=baseline_transforms['val'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Instantiate baseline neural network topology and assign serialized weights
    model = BaselineCNN(num_classes=2)
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Execution Error: Serialized weight parameters missing at path target: {model_file}")

    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    # Deactivate autograd engine to optimize memory during inference tracking
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compile categorical diagnostic outcomes
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    # Structure performance scorecard parameters
    metrics = {
        'Accuracy': accuracy_score(all_labels, all_preds),
        'Precision': precision_score(all_labels, all_preds),
        'Recall (Sens.)': recall_score(all_labels, all_preds),
        'Specificity': tn / (tn + fp),
        'F1-Score': f1_score(all_labels, all_preds),
        'MCC': matthews_corrcoef(all_labels, all_preds)
    }

    # Generate high-resolution visual performance dashboard panel
    fig = plt.figure(figsize=(16, 8), facecolor='#fdfcf0')
    plt.suptitle('BASELINE AUDIT: Standard CNN Performance (Unseen Data)',
                 fontsize=22, fontweight='bold', color='#7f8c8d', y=0.96)

    # Panel Component 1: Visual Heatmap Confusion Matrix
    ax1 = plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
                annot_kws={'size': 16, 'fontweight': 'bold'},
                xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'])
    ax1.set_title('Baseline: Confusion Matrix', fontsize=16, pad=20)
    ax1.set_xlabel('Predicted Diagnostic Class', fontsize=12)
    ax1.set_ylabel('Ground Truth Label', fontsize=12)

    # Panel Component 2: Quantitative Scorecard Dashboard
    ax2 = plt.subplot(1, 2, 2)
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = ['#e67e22' for _ in values]

    bars = ax2.barh(names, values, color=colors, height=0.6)
    ax2.set_xlim(0, 1.1)
    ax2.set_title('Baseline Performance Metrics Summary', fontsize=16, pad=20)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{width * 100:.1f}%' if names[i] != 'MCC' else f'{width:.3f}',
                 va='center', fontweight='bold', fontsize=12)

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.92])
    output_path = os.path.join(script_dir, 'baseline_test_scorecard.png')

    # Serialize diagnostic visualization and terminate plot thread memory allocations
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Baseline Statistical Audit Complete. Results exported to: {output_path}")