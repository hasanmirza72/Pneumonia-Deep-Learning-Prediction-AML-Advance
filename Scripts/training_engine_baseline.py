from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


def train_radiology_model(model, criterion, optimizer, loaders, num_epochs=15):
    # This function handles the heavy lifting of moving data through the network.
    # It uses 'tqdm' to show you a live progress bar for every batch.
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    # We create a history dictionary to store every percentage for your report.
    # This data is what allows us to draw the 'Learning Curve' PNG later.

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            # We loop through training to learn, and validation to check for mistakes.
            # The model enters 'eval' mode during validation to keep results honest.
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, corrects = 0.0, 0
            pbar = tqdm(loaders[phase], desc=f"Epoch {epoch + 1} {phase.upper()}")
            # The 'tqdm' bar will appear in your PyCharm terminal with a timer.
            # It tells you exactly how many seconds are left until the epoch ends.

            for images, labels in pbar:
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # We process the images and calculate how far off the prediction was.
                    # If training, we update the model's 'brain' to be smarter next time.
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # The math flows backward to correct the errors identified in the lung scan.
                # This is how the baseline model slowly learns the difference from 'Normal'.

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
            # We track every single correct guess to build the final accuracy score.
            # This is updated in real-time so the progress bar stays accurate.

            epoch_loss = running_loss / len(loaders[phase].dataset)
            epoch_acc = corrects.double() / len(loaders[phase].dataset)
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
    # We save the final stats for the epoch into our history tracking system.
    # This ensures no data is lost during the long processing run.

    return model, history
# Once all epochs are done, the engine returns the trained model and data.
# You now have a 'Certified' baseline model ready for the final audit.
