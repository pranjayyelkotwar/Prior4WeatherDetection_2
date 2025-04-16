import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualFeatureRecoveryBlock(nn.Module):
    """Residual Feature Recovery Block (RFRB)"""
    def __init__(self, in_channels, out_channels, verbose=False):
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, 
                              kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, 
                              kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, stride=1, padding=1)
        self.verbose = verbose
        
    def forward(self, x):
        if self.verbose:
            print("Passing features through ResidualFeatureRecoveryBlock")
            print(f"Input shape: {x.shape}, Output shape: {x.shape}")
        x = self.maxpool1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
