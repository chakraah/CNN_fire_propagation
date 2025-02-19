import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 

from torch.utils.data import Dataset

# CNN Model for Predicting Fire Spread
class FirePropagationCNN(nn.Module):
    def __init__(self):
        super(FirePropagationCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 2, out_channels = 12, kernel_size = 5) 
        # [4,12,220,220]
        self.pool1 = nn.MaxPool2d(2,2) #reduces the images by a factor of 2
        # [4,12,110,110]
        self.conv2 = nn.Conv2d(in_channels = 12, out_channels = 24, kernel_size = 5)
        # [4,24,106,106]
        self.pool2 = nn.MaxPool2d(2,2)
        # [4,24,53,53] which becomes the input of the fully connected layer 
        self.fc1 = nn.Linear(in_features = (24 * 53 * 53), out_features = 120) 
        self.fc2 = nn.Linear(in_features = 120, out_features = 84) 
        self.fc3 = nn.Linear(in_features = 84, out_features = 2) #final layer, output will be the number of classes 

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  
        x = self.pool2(F.relu(self.conv2(x)))  
        x = F.relu(self.fc1(x))               
        x = F.relu(self.fc2(x))              
        x = self.fc3(x)                       
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