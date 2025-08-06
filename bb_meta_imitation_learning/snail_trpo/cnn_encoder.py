# snail_trpo/cnn_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    CNN encoder for images.
    Preprocesses the image using the same convolutional architecture as Duan et al. (2016):
      - Two layers with kernel size 5×5, 16 filters, stride 2, and ReLU nonlinearity,
      - Followed by flattening and a fully-connected layer to produce a 256-dimensional feature vector.
    
    This module expects a 3-channel input.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 7))
        self.fc = nn.Linear(16 * 5 * 7, 256)

    def forward(self, obs):
        """
        Args:
            obs (Tensor): shape (B, in_channels, H, W)
        Returns:
            Tensor: shape (B, 256) embedding of the image.
        """
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = x.reshape(x.size(0), -1)
        return F.relu(self.fc(x))
