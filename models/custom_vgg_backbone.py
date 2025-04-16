import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict

class CustomVGGBackbone(nn.Module):
    def __init__(self, verbose=False, freeze_conv3=True):
        super().__init__()
        vgg = torchvision.models.vgg16(pretrained=True).features

        # Layer breakdown based on torchvision.models.vgg16 structure:
        # conv1_1 to conv1_2 → 0 to 4
        # conv2_1 to conv2_2 → 5 to 9
        # conv3_1 to conv3_3 → 10 to 16
        # conv4_1 to conv4_3 → 17 to 23
        # conv5_1 to conv5_3 → 24 to 30

        self.c1_c3 = nn.Sequential(*list(vgg.children())[:17])   # conv1_1 to conv3_3
        self.c4 = nn.Sequential(*list(vgg.children())[17:24])    # conv4_1 to conv4_3
        self.c5 = nn.Sequential(*list(vgg.children())[24:31])    # conv5_1 to conv5_3

        # Freeze weights up to conv3 if requested
        if freeze_conv3:
            for param in self.c1_c3.parameters():
                param.requires_grad = False
            if verbose:
                print("Frozen weights up to conv3 layer in VGG backbone")

        self.out_channels = 512
        self.verbose = verbose
        
        # Store C3 features for access but not directly returned to FasterRCNN
        self.c3_features = None

    def forward(self, x):
        if self.verbose:
            print("Extracting features using CustomVGGBackbone")
            print(f"Input shape: {x.shape}")
        c3 = self.c1_c3(x)
        if self.verbose:
            print(f"C3 shape: {c3.shape}")
            
        # Store C3 features for later access
        self.c3_features = c3

        c4 = self.c4(c3)
        if self.verbose:
            print(f"C4 shape: {c4.shape}")

        c5 = self.c5(c4)
        if self.verbose:
            print(f"C5 shape: {c5.shape}")

        # Return only C4 and C5 features to FasterRCNN
        return {'0': c4, '1': c5}
        
    def get_c3_features(self):
        """Get the stored C3 features from the last forward pass"""
        return self.c3_features