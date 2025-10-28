import torch.nn as nn
import torch.nn.functional as F
import torch
from torchinfo import summary

class CNN(nn.Module):
    def __init__(self,no_classes=1):
        super().__init__()
        self.channels=channels
        Conv_output_size=int((((input_size-4)/2)-4)/2)
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(Conv_output_size*Conv_output_size*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, no_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x