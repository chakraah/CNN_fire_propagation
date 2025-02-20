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

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 2, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.sigmoid(self.conv6(x))
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