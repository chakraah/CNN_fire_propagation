import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from fire_propagation_model.fire_propagation_model import WildfireSimulation

# Attention mechanism class
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1))
        return x * psi

# U-Net with Attention
class UNetWithAttention(nn.Module):
    def __init__(self):
        super(UNetWithAttention, self).__init__()

        # Encoder layers
        self.enc1 = self.conv_block(2, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Attention and Decoder layers
        self.att4 = AttentionBlock(512, 1024, 512)
        self.dec4 = self.conv_block(1024, 512)

        self.att3 = AttentionBlock(256, 512, 256)
        self.dec3 = self.conv_block(512, 256)

        self.att2 = AttentionBlock(128, 256, 128)
        self.dec2 = self.conv_block(256, 128)

        self.att1 = AttentionBlock(64, 128, 64)
        self.dec1 = self.conv_block(128, 64)

        # Output layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        """ Convolutional block: Conv2D + BatchNorm + ReLU """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with Attention
        d4 = self.dec4(torch.cat([self.att4(e4, b), F.interpolate(b, scale_factor=2)], dim=1))
        d3 = self.dec3(torch.cat([self.att3(e3, d4), F.interpolate(d4, scale_factor=2)], dim=1))
        d2 = self.dec2(torch.cat([self.att2(e2, d3), F.interpolate(d3, scale_factor=2)], dim=1))
        d1 = self.dec1(torch.cat([self.att1(e1, d2), F.interpolate(d2, scale_factor=2)], dim=1))

        # Output
        return self.sigmoid(self.final(d1))

# Dataset for loading fire simulation data
class FireDataset(Dataset):
    def __init__(self, fire_matrices, fuel_matrices, next_fire_matrices):
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

# Training class
class Trainer:
    def __init__(self, model, train_loader, val_loader, device, lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_fire_model.pth")
                print("🔥 Model saved!")

# Visualization of predictions
def visualize_predictions(model, dataset, num_samples=5):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))

    with torch.no_grad():
        for i in range(num_samples):
            input_data, true_fire = dataset[i]
            input_data = input_data.unsqueeze(0).to(device)
            pred_fire = model(input_data).squeeze().cpu().numpy()

            true_fire = true_fire.squeeze().numpy()
            input_fire = input_data[0, 0].cpu().numpy()
            fuel_map = input_data[0, 1].cpu().numpy()

            axes[i, 0].imshow(input_fire, cmap="hot", interpolation="nearest")
            axes[i, 0].set_title("🔥 Fire at t")
            axes[i, 1].imshow(pred_fire, cmap="hot", interpolation="nearest")
            axes[i, 1].set_title("🔮 Predicted Fire at t+1")
            axes[i, 2].imshow(true_fire, cmap="hot", interpolation="nearest")
            axes[i, 2].set_title("✅ Actual Fire at t+1")

            for ax in axes[i]:
                ax.axis("off")

    plt.tight_layout()
    plt.show()

# Sample prediction printing
def print_sample_prediction(model, dataset, sample_idx=0):
    model.eval()
    with torch.no_grad():
        input_data, true_fire = dataset[sample_idx]
        input_data = input_data.unsqueeze(0).to(device)
        pred_fire = model(input_data).squeeze().cpu().numpy()

        print("🔥 Fire at t (Input):")
        print(input_data[0, 0].cpu().numpy())

        print("\n🌲 Fuel Map (Input):")
        print(input_data[0, 1].cpu().numpy())

        print("\n🔮 Predicted Fire at t+1:")
        print(pred_fire)

        print("\n✅ Actual Fire at t+1:")
        print(true_fire.squeeze().numpy())

# Sample usage (assuming your simulation data is generated):
wildfire_simulation = WildfireSimulation(20, 0.9, 20, 5, 0.2, [(10,10)], 0)
fire_data, fuel_history = wildfire_simulation.run_simulation()

# Convert data
fire_data = np.array(fire_data)
fuel_history = np.array(fuel_history)

# Define input & target
input_fire = fire_data[:-1]
input_fuel = fuel_history[:-1]
target_fire = fire_data[1:]

# Create dataset and dataloader
dataset = FireDataset(input_fire, input_fuel, target_fire)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model and trainer
model = UNetWithAttention()
trainer = Trainer(model, train_loader, train_loader, device)

# Train the model
trainer.train(epochs=20)

# Visualize predictions on the trained model
visualize_predictions(model, dataset)
