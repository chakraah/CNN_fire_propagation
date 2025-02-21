import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 

from torch.utils.data import Dataset

# CNN Model for Predicting Fire Spread
class FirePropagationCNN(nn.Module):
    def __init__(self):
        super(FirePropagationCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 2, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.sigmoid(self.conv4(x))
        return x
 
# Dataset for training
class FirePropagationDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

# Model training
class ModelTrainer:
    def __init__(self, model, dataloader, criterion, optimizer, num_epochs, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
    
    def train(self):
        self.model.train()  # Set model to training mode
        all_losses = []  # Store loss values for monitoring

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(self.dataloader):
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                fire_outputs, fuel_outputs = outputs[:, 0], outputs[:, 1]  # Assuming the first channel is fire, second is fuel
                fire_targets, fuel_targets = targets[:, 0], targets[:, 1]
                
                fire_loss = self.loss_function(fire_outputs, fire_targets)
                fuel_loss = self.loss_function(fuel_outputs, fuel_targets)
                loss = fire_loss + fuel_loss

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Log running loss
                running_loss += loss.item()

                # Optionally, print loss every 100 batches
                if i % 100 == 99:  # Print every 100 batches
                    print(f"Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{len(self.dataloader)}], Loss: {running_loss / 100:.4f}")
                    running_loss = 0.0

            # After each epoch, store the average loss
            avg_loss = running_loss / len(self.dataloader)
            all_losses.append(avg_loss)

            # Visualize the intermediate predictions after a few epochs
            if epoch % 10 == 0:  # Visualize every 10 epochs (adjust as needed)
                self.visualize_predictions(inputs, fire_targets, fuel_targets, epoch)

        # Plot loss graph after training is complete
        self.plot_loss_curve(all_losses)

    def visualize_predictions(self, inputs, fire_targets, fuel_targets, epoch):
        # Select a few images from the batch
        sample_input = inputs[0].cpu().detach().numpy()
        sample_fire_target = fire_targets[0].cpu().detach().numpy()
        sample_fuel_target = fuel_targets[0].cpu().detach().numpy()

        # Model prediction
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(inputs.unsqueeze(0).to(self.device))
            fire_pred = outputs[0, 0].cpu().detach().numpy()
            fuel_pred = outputs[0, 1].cpu().detach().numpy()

        # Plot the ground truth and predicted outputs
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))

        axes[0].imshow(sample_fire_target, cmap='hot')
        axes[0].set_title('Fire Ground Truth')

        axes[1].imshow(sample_fuel_target, cmap='Greens')
        axes[1].set_title('Fuel Ground Truth')

        axes[2].imshow(fire_pred, cmap='hot')
        axes[2].set_title(f'Fire Prediction (Epoch {epoch})')

        axes[3].imshow(fuel_pred, cmap='Greens')
        axes[3].set_title(f'Fuel Prediction (Epoch {epoch})')

        plt.show()

    def plot_loss_curve(self, loss_values):
        # Plot the loss curve after training
        plt.plot(range(1, len(loss_values) + 1), loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.show()