import torch
import torch.nn as nn

class CompareNet(nn.Module):
    """
    CompareNet model for comparing two fragment representations
    
    This network takes two fragment representations and outputs a similarity score.
    """
    
    def __init__(self):
        """Initialize CompareNet with convolutional layers"""
        super(CompareNet, self).__init__()
        # 32 -> 23
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=10, padding="valid")
        self.a1 = nn.PReLU()
        # 23 -> 19
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=2, kernel_size=5, padding="valid")
        self.a2 = nn.PReLU()
        self.fc1 = nn.Linear(2 * 19, 1)
        self.a3 = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, 2, 32]
            
        Returns:
            Output tensor of shape [batch_size, 1]
        """
        x = self.conv1(x)
        x = self.a1(x)
        x = self.conv2(x)
        x = self.a2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.a3(x)
        return x


def inference(f_model, c_model, vector_1, vector_2):
    """
    Calculate the distance/dissimilarity between two vectors using the models
    
    Lower values indicate higher similarity
    
    Args:
        f_model: VectorNet model
        c_model: CompareNet model
        vector_1: First fragment vector
        vector_2: Second fragment vector
        
    Returns:
        Dissimilarity score (lower means more similar)
    """
    f_model.eval()
    c_model.eval()
    feature_1 = f_model(vector_1)
    feature_2 = f_model(vector_2)
    dis = c_model(torch.stack((feature_1, feature_2), 1))
    return dis