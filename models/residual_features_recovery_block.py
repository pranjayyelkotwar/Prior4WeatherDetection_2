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
        self.bn1 = nn.BatchNorm2d(out_channels//2)  # Add batch normalization
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, 
                              kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)     # Add batch normalization
        self.conv3 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, stride=1, padding=1)
        self.verbose = verbose
        
        # Initialize weights for faster convergence
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        if self.verbose:
            print("Passing features through ResidualFeatureRecoveryBlock")
            print(f"Input shape: {x.shape}")
        
        # Use maxpool with ceil_mode for better size compatibility
        x = self.maxpool1(x)
        
        # Use inplace operations for memory efficiency
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.conv3(x)
        
        if self.verbose:
            print(f"Output shape: {x.shape}")
        return x
