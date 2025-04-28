import torch
import torch.nn as nn

class VectorNet(nn.Module):
    """
    VectorNet model for feature extraction from fragment edges
    
    This network converts raw edge features into a latent representation.
    """
    
    def __init__(self):
        """Initialize VectorNet with convolutional layers"""
        super(VectorNet, self).__init__()
        # 64 -> 55
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=10, padding="valid")
        self.a1 = nn.PReLU()
        # 55 -> 51
        self.conv2 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=5, padding="valid")
        self.a2 = nn.PReLU()
        self.fc1 = nn.Linear(5 * 51, 32)
        self.a3 = nn.PReLU()

    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, 1, 64]
            
        Returns:
            Output tensor of shape [batch_size, 32]
        """
        x = self.conv1(x)
        x = self.a1(x)
        x = self.conv2(x)
        x = self.a2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.a3(x)
        return x