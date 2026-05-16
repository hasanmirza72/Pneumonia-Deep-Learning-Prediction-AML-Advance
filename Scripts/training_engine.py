import os
import time
import copy
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim import lr_scheduler


def train_radiology_model(model, criterion, optimizer, scheduler, loaders, num_epochs=20):
    """
    Executes the deep learning optimization cycle over a specified number of epochs.
    Orchestrates dual-phase training and validation passes, logs cross-entropy convergence,
    and checkpoint-serializes the model parameters achieving top diagnostic accuracy.
    """
    since = time.time()

    # Enforce hardware-agnostic initialization outside the processing loops to minimize overhead
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Maintain a deep copy of peak parameter weights to safeguard against overfitting convergence
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}/{num_epochs - 1}')
        print('-' * 30)

        for phase in ['train', 'val']:
            # Toggle structural regularization layers based on operational phase
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Instantiate real-time monitoring progress bar over data matrix batches
            pbar = tqdm(loaders[phase], desc=f"{phase.upper()} Phase")

            for inputs, labels in pbar:
                # Synchronize input tensors and targets onto the target hardware memory
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Reset parameter gradients to zero to prevent historical accumulation
                optimizer.zero_grad()

                # Track feature map operations strictly during the supervised learning pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Execute backward propagation step and update structural weights
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Accumulate normalization coefficients across current batch limits
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Compute normalized metrics over the entire data split cohort
            epoch_loss = running_loss / len(loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(loaders[phase].dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Update the plateau scheduler dynamically using the validation loss matrix
            if phase == 'val':
                scheduler.step(epoch_loss)

            # Register parameters if validation accuracy outperforms prior epoch horizons
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\nTraining Phase Complete in: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Peak Validation Accuracy Registered: {best_acc:.4f}')

    # Load peak parameter distribution weights back to the model instance before exporting
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    from data_loader import get_xray_loaders
    from model_builder import initialize_advanced_model, get_clinical_criterion

    # Resolve execution paths dynamically to allow zero-configuration cross-platform deployments
    script_dir = os.path.dirname(os.path.abspath(__file__))
    XRAY_PATH = os.path.join(script_dir, 'ChestXRay2017', 'chest_xray')

    # Environment fallback checkpoint for specific local host setups
    if not os.path.exists(XRAY_PATH):
        XRAY_PATH = r'D:\AML Project\Advance\rscbjbr9sj-2\ChestXRay2017\chest_xray'

    if not os.path.exists(XRAY_PATH):
        raise FileNotFoundError(f"Execution Error: Core data directory tree not resolved at target: {XRAY_PATH}")

    # Initialize data loaders and categorical distribution counts
    loaders, train_counts = get_xray_loaders(XRAY_PATH)

    # Initialize neural network architecture using dynamic target resource lookups
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_advanced_model(num_classes=2, device=current_device)

    # Incorporate inverse class frequency weights into cross-entropy loss optimizations
    criterion = get_clinical_criterion(class_counts=train_counts, device=current_device)

    # Optimize using the Adam algorithm for adaptive momentum feature updates
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Attenuate the learning rate parameter dynamically when validation loss stagnates
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    print("Pipeline Execution: Starting Advanced EfficientNet Optimization Sequences.")

    # Execute supervised learning cycle optimization
    trained_model = train_radiology_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=exp_lr_scheduler,
        loaders=loaders,
        num_epochs=20
    )

    # Securely serialize the resulting champion parameter states to an absolute script destination
    output_checkpoint_path = os.path.join(script_dir, 'pneumonia_classifier_v1.pth')
    torch.save(trained_model.state_dict(), output_checkpoint_path)
    print(f"Success: Optimization artifacts exported cleanly to: {output_checkpoint_path}")