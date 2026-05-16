import os
import torch
import matplotlib.pyplot as plt
from baseline_model import BaselineCNN
from training_engine_baseline import train_radiology_model
from data_loader_baseline import get_baseline_loaders as get_xray_loaders


def plot_baseline_results(history, save_path):
    """
    Generates and serializes a high-resolution performance dashboard detailing
    cross-entropy loss reduction and diagnostic accuracy optimization curves.
    """
    plt.figure(figsize=(12, 5))

    # Subplot 1: Cross-Entropy Loss Convergence Trace
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange', linestyle='--')
    plt.title('Baseline: Loss Evolution', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Metrics')
    plt.legend()

    # Subplot 2: Categorical Accuracy Evolution Trace
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy', color='green')
    plt.plot(history['val_acc'], label='Validation Accuracy', color='red', linestyle='--')
    plt.title('Baseline: Accuracy Evolution', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Explicitly serialize to disk and terminate figure memory allocation to prevent runtime leaks
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics Evaluation: Optimization curves successfully saved to {save_path}")


if __name__ == "__main__":
    # Dynamically resolve directory path contexts to ensure multi-platform validation safety
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'ChestXRay2017', 'chest_xray')

    # Priority Fallback: Attempt local directory map if primary path mapping fails
    if not os.path.exists(data_path):
        data_path = r'D:\AML Project\Advance\rscbjbr9sj-2\ChestXRay2017\chest_xray'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Execution Error: Missing target data directory context at: {data_path}")

    # Initialize data engineering pipeline and structural model configurations
    loaders, _ = get_xray_loaders(data_path)
    model = BaselineCNN(num_classes=2)

    # Standardize optimization parameters and empirical loss constraints
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Pipeline Core: Initializing Empirical Baseline Optimization Phase.")

    # Orchestrate the supervised learning loop execution
    model, history = train_radiology_model(model, criterion, optimizer, loaders, num_epochs=15)

    # Resolve deterministic absolute output paths for tracking artifacts safely
    curves_output_path = os.path.join(script_dir, 'baseline_learning_curves.png')
    weights_output_path = os.path.join(script_dir, 'baseline_model.pth')

    # Serialize evaluation logs and structural parameter weights
    plot_baseline_results(history, curves_output_path)
    torch.save(model.state_dict(), weights_output_path)

    print("Execution Success: Baseline deployment loop executed without terminal errors.")