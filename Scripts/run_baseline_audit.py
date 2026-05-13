import torch
import os
import matplotlib.pyplot as plt
from baseline_model import BaselineCNN
# --- CRITICAL CHANGE: Import from your NEW file name ---
from training_engine_baseline import train_radiology_model
from data_loader_baseline import get_baseline_loaders as get_xray_loaders


def plot_baseline_results(history, save_path):
    # This function solves the problem you had last time.
    # It takes the results and writes them directly to a PNG file on your D: drive.
    plt.figure(figsize=(12, 5))

    # 1. Plotting the Loss (Error Rate)
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='orange', linestyle='--')
    plt.title('Baseline: Loss Evolution', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()

    # 2. Plotting the Accuracy (Success Rate)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', color='green')
    plt.plot(history['val_acc'], label='Val Acc', color='red', linestyle='--')
    plt.title('Baseline: Accuracy Evolution', fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # This line 'burns' the image into your folder so you don't have to copy anything.
    plt.savefig(save_path, dpi=300)
    plt.close()  # <--- This prevents the script from hanging/freezing!
    print(f"🖼️ Learning curves saved automatically as: {save_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'ChestXRay2017', 'chest_xray')

    loaders, _ = get_xray_loaders(data_path)
    model = BaselineCNN(num_classes=2)

    # Define your clinical loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(" Baseline work started. ")

    # The engine runs for 15 epochs and gathers all the numbers for you.
    model, history = train_radiology_model(model, criterion, optimizer, loaders, num_epochs=15)

    # This automatically creates your PNG chart.
    plot_baseline_results(history, 'baseline_learning_curves.png')

    # This automatically saves your model weights.
    torch.save(model.state_dict(), 'baseline_model.pth')

    print("✅ Process Finished. Charts and weights are ready in your folder.")