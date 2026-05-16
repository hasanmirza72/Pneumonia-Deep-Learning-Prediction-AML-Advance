import torch
from tqdm import tqdm


def train_radiology_model(model, criterion, optimizer, loaders, num_epochs=15):
    """
    Executes the supervised learning optimization loop for the baseline CNN.
    Records cross-entropy loss convergence and diagnostic accuracy parameters
    across training and validation partitions.
    """
    # Enforce hardware-agnostic environment synchronization to prevent runtime device mismatches
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize tracking registries for subsequent learning curve analysis
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            # Toggle internal structural tracking layers depending on operational state
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Instantiate mini-batch iterator bar for real-time monitoring
            pbar = tqdm(loaders[phase], desc=f"Epoch {epoch + 1}/{num_epochs} [{phase.upper()}]")

            for images, labels in pbar:
                # Mathematically synchronize tensor elements to the active hardware target
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Condition gradient history compilation on operational mode
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # Backpropagate error gradients and apply network weight optimizations
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Accumulate quantitative metadata across mini-batch matrices
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Compute normalized epoch metrics based on total partition scale
            epoch_loss = running_loss / len(loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(loaders[phase].dataset)

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

    return model, history