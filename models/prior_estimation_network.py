import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorEstimationNetwork(nn.Module):
    """Prior Estimation Network (PEN) module"""
    def __init__(self, in_channels, verbose=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.verbose = verbose
        
        # Initialize weights for faster convergence
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.conv4.weight)

    def forward(self, x):
        if self.verbose:
            print("Passing features through PriorEstimationNetwork")
            print(f"Input shape: {x.shape}")
            
        # Use inplace operations where possible for memory efficiency
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = torch.tanh(self.conv4(x))
        
        if self.verbose:
            print(f"Output shape: {x.shape}")
        return x
