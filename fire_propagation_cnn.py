import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 

from torch.utils.data import Dataset

# CNN Model for Predicting Fire Spread
class FirePropagationCNN(nn.Module):
    def __init__(self):
        super(FirePropagationCNN, self).__init__()
        
        # Encoding path (downsampling)
        self.enc1 = self.conv_block(2, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)

        # Decoding path (upsampling)
        self.upconv5 = self.upconv_block(1024, 512)
        self.upconv4 = self.upconv_block(512, 256)
        self.upconv3 = self.upconv_block(256, 128)
        self.upconv2 = self.upconv_block(128, 64)

        # Final layer
        self.final = nn.Conv2d(64, 2, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoding path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        # Decoding path with skip connections
        upconv5 = self.upconv5(enc5)
        upconv4 = self.upconv4(upconv5 + enc4)  # Skip connection
        upconv3 = self.upconv3(upconv4 + enc3)  # Skip connection
        upconv2 = self.upconv2(upconv3 + enc2)  # Skip connection

        # Final output
        out = self.final(upconv2 + enc1)  # Skip connection
        return out
 
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
        # Trains the CNN model
        self.model.to(self.device)

        loss_function = nn.BCEWithLogitsLoss()

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            for inputs, labels in self.dataloader:
                # Move data to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                fire_outputs, fuel_outputs = outputs[:, 0], outputs[:, 1]  # Assuming the first channel is fire, second is fuel
                fire_targets, fuel_targets = labels[:, 0], labels[:, 1]
                
                fire_loss = self.criterion(fire_outputs, fire_targets)
                fuel_loss = loss_function(fuel_outputs, fuel_targets)
                loss = fire_loss + fuel_loss

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {total_loss / len(self.dataloader):.4f}")