import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 

from torch.utils.data import Dataset

# CNN Model for Predicting Fire Spread
class FirePropagationCNN(nn.Module):
    def __init__(self):
        super(FirePropagationCNN, self).__init__()
        
# Encoder: Downsampling path
        self.enc1 = self.contract_block(2, 32)
        self.enc2 = self.contract_block(32, 64)
        self.enc3 = self.contract_block(64, 128)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)

        # Decoder: Upsampling path
        self.upconv3 = self.upconv_block(256, 128)  # After bottleneck, 256 channels
        self.upconv2 = self.upconv_block(128 + 128, 64)  # Input 128 from upconv3 + 128 from enc3
        self.upconv1 = self.upconv_block(64 + 64, 32)  # Input 64 from upconv2 + 64 from enc2

        # Final convolution layer
        self.final_conv = nn.Conv2d(32 + 32, 2, kernel_size=1)  # Input 32 from upconv1 + 32 from enc1

    def contract_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

    def upconv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        return block

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)   # (Batch, 32, H/2, W/2)
        enc2 = self.enc2(enc1) # (Batch, 64, H/4, W/4)
        enc3 = self.enc3(enc2) # (Batch, 128, H/8, W/8)

        # Bottleneck
        bottleneck = self.bottleneck(enc3)  # (Batch, 256, H/16, W/16)

        # Decoder
        upconv3 = self.upconv3(bottleneck)  # (Batch, 128, H/8, W/8)
        upconv3 = F.interpolate(upconv3, size=enc3.shape[2:], mode='bilinear', align_corners=False)  # Resize to match enc3
        upconv3 = torch.cat([upconv3, enc3], dim=1)  # Skip connection

        upconv2 = self.upconv2(upconv3)  # (Batch, 64, H/4, W/4)
        upconv2 = F.interpolate(upconv2, size=enc2.shape[2:], mode='bilinear', align_corners=False)  # Resize to match enc2
        upconv2 = torch.cat([upconv2, enc2], dim=1)  # Skip connection

        upconv1 = self.upconv1(upconv2)  # (Batch, 32, H/2, W/2)
        upconv1 = F.interpolate(upconv1, size=enc1.shape[2:], mode='bilinear', align_corners=False)  # Resize to match enc1
        upconv1 = torch.cat([upconv1, enc1], dim=1)  # Skip connection

        # Final convolution layer
        out = self.final_conv(upconv1)  # (Batch, 2, H, W)

        # Resize the output to match the target size (e.g., 200x200)
        out = F.interpolate(out, size=(200, 200), mode='bilinear', align_corners=False)

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

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            for inputs, labels in self.dataloader:
                # Move data to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {total_loss / len(self.dataloader):.4f}")