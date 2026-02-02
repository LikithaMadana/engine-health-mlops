"""
PyTorch 1D CNN model for engine health prediction.
Aligned with baseline conventions.
"""

import torch
import torch.nn as nn
from typing import Optional


class EngineHealthCNN(nn.Module):
    """
    1D CNN for engine health classification.
    Follows baseline architecture: 2 conv layers, simple structure.
    """
    
    def __init__(self, 
                 input_channels: int = 6,
                 window_size: int = 10,
                 hidden_channels: int = 16):
        """
        Initialize the CNN model (baseline-aligned).
        
        Args:
            input_channels: Number of input features (default: 6)
            window_size: Size of the input window (default: 10)
            hidden_channels: Number of channels in first layer (default: 16)
        """
        super(EngineHealthCNN, self).__init__()
        
        # Baseline architecture: 2 conv layers (16 -> 32 channels)
        self.conv_stack = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, 
                     out_channels=hidden_channels, 
                     kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_channels, 
                     out_channels=hidden_channels * 2, 
                     kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            # Linear input size: (window_size - 4) * (hidden_channels * 2)
            # window_size=10: (10-4) * 32 = 192
            nn.Linear((hidden_channels * 2) * (window_size - 4), 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, window, features)
            
        Returns:
            Output tensor of shape (batch, 1) with sigmoid activation
        """
        # Transpose: (batch, window, features) -> (batch, features, window)
        x = x.transpose(1, 2)
        return self.conv_stack(x)


# Backward compatibility alias
EngineCNN = EngineHealthCNN
