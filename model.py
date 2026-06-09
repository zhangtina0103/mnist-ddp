import torch
import torch.nn as nn


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        # each image 7x7, 64 channels
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        self.relu  = nn.ReLU()

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = self.pool(self.relu(self.conv1(x))) # (batch, 32, 14, 14)
        x = self.pool(self.relu(self.conv2(x))) # (batch, 64, 7, 7)
        # flatten
        x = x.view(x.size(0), -1) # (batch, 64*7*7)
        x = self.relu(self.fc1(x)) # (batch, 128)
        x = self.fc2(x) # (batch, 10)
        return x
