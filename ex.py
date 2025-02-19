import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from fire_propagation_model.fire_propagation_model import WildfireSimulation


class FirePropagationCNN(nn.Module):
    def __init__(self):
        super(FirePropagationCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class FireDataset(Dataset):
    def __init__(self, fire_matrices, fuel_matrices, next_fire_matrices):
        """
        Args:
            fire_matrices: Fire states at time t.
            fuel_matrices: Fuel amounts at time t.
            next_fire_matrices: Fire states at time t+1.
        """
        self.fire_matrices = torch.tensor(fire_matrices, dtype=torch.float32) / 1.0
        self.fuel_matrices = torch.tensor(fuel_matrices, dtype=torch.float32) / 255.0
        self.next_fire_matrices = torch.tensor(next_fire_matrices, dtype=torch.float32)

    def __len__(self):
        return len(self.fire_matrices)

    def __getitem__(self, idx):
        fire = self.fire_matrices[idx].unsqueeze(0)  # Shape (1, H, W)
        fuel = self.fuel_matrices[idx].unsqueeze(0)  # Shape (1, H, W)
        input_data = torch.cat([fire, fuel], dim=0)  # Shape (2, H, W)
        target = self.next_fire_matrices[idx].unsqueeze(0)  # Shape (1, H, W)
        return input_data, target

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()  # Binary classification
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, epochs=10):
        best_val_loss = float('inf')
        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_fire_model.pth")
                print("Model saved!")

# Generate data using your function
wildfire_simulation = WildfireSimulation(20, 0.9, 50, 5, 0.2, [(10,10)], 0)
fire_data, fuel_history = wildfire_simulation.run_simulation()

# Convert to numpy arrays for PyTorch processing
fire_data = np.array(fire_data)  # Shape: (num_samples, H, W)
fuel_history = np.array(fuel_history)  # Shape: (num_samples, H, W)

# Define input (current state) and target (next step)
input_fire = fire_data[:-1]  # Fire state at time t
input_fuel = fuel_history[:-1]  # Fuel state at time t
target_fire = fire_data[1:]  # Fire state at time t+1 (label)

# Create dataset
dataset = FireDataset(input_fire, input_fuel, target_fire)

# Create data loader
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and trainer
model = FirePropagationCNN()
trainer = Trainer(model, train_loader, train_loader, device)  # Using the same loader for now

# Train the model
trainer.train(epochs=20)

import matplotlib.pyplot as plt

def visualize_predictions(model, dataset, num_samples=5):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))
    
    with torch.no_grad():
        for i in range(num_samples):
            input_data, true_fire = dataset[i]  # Get sample
            input_data = input_data.unsqueeze(0).to(device)  # Add batch dim
            pred_fire = model(input_data).squeeze().cpu().numpy()  # Forward pass


            
            true_fire = true_fire.squeeze().numpy()
            input_fire = input_data[0, 0].cpu().numpy()  # Fire at t
            fuel_map = input_data[0, 1].cpu().numpy()  # Fuel map

            active_fire_pixels = np.sum(true_fire == 1)
            threshold_value = np.sort(pred_fire.flatten())[-active_fire_pixels]
            pred_fire = np.where(pred_fire >= threshold_value, 1, 0)

            # Plot original fire state
            axes[i, 0].imshow(input_fire, cmap="hot", interpolation="nearest")
            axes[i, 0].set_title("Fire at t")

            # Plot predicted fire state
            axes[i, 1].imshow(pred_fire, cmap="hot", interpolation="nearest")
            axes[i, 1].set_title("Predicted Fire at t+1")

            # Plot ground truth fire state
            axes[i, 2].imshow(true_fire, cmap="hot", interpolation="nearest")
            axes[i, 2].set_title("Actual Fire at t+1")

            for ax in axes[i]:
                ax.axis("off")

            plt.pause(0.01)

    plt.tight_layout()
    plt.show()

# Run visualization
visualize_predictions(model, dataset)
