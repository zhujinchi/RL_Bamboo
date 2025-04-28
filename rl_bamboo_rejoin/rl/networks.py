import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Actor(nn.Module):
    """
    Actor Network for SAC
    
    Implements a stochastic policy that outputs a distribution over actions.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, log_std_min=-20, log_std_max=2):
        """
        Initialize the Actor network
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """
        Forward pass through the network
        
        Args:
            state: State tensor
            
        Returns:
            mu: Mean of the action distribution
            std: Standard deviation of the action distribution
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mu = self.mu(x)
        
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mu, std
    
    def sample(self, state):
        """
        Sample an action from the policy
        
        Args:
            state: State tensor
            
        Returns:
            action: Sampled action
            log_prob: Log probability of the action
        """
        mu, std = self.forward(state)
        normal = Normal(mu, std)
        
        # Use reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Calculate log_prob, accounting for tanh transformation
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """
    Critic Network for SAC
    
    Implements two Q-networks to reduce overestimation bias.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize the Critic network
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
        """
        super(Critic, self).__init__()
        
        # Q1 network
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 network
        self.fc3 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        """
        Forward pass through both Q-networks
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            q1: Q-value from first network
            q2: Q-value from second network
        """
        # Concatenate state and action
        x = torch.cat([state, action], 1)
        
        # Q1
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)
        
        # Q2
        q2 = F.relu(self.fc3(x))
        q2 = F.relu(self.fc4(q2))
        q2 = self.q2(q2)
        
        return q1, q2