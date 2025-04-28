import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .networks import Actor, Critic

class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent
    
    Implements the SAC algorithm for continuous control tasks.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, 
                 alpha=0.2, gamma=0.99, tau=0.005, lr=3e-4):
        """
        Initialize the SAC agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
            alpha: Temperature parameter for entropy
            gamma: Discount factor
            tau: Soft update coefficient
            lr: Learning rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # Discount factor
        self.tau = tau      # Soft update coefficient
        self.alpha = alpha  # Entropy coefficient
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Initialize target network parameters to match main network
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
    def select_action(self, state, evaluate=False):
        """
        Select an action given the current state
        
        Args:
            state: Current state
            evaluate: If True, use deterministic policy
            
        Returns:
            Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if evaluate:
            mu, _ = self.actor(state)
            return torch.tanh(mu).detach().cpu().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            return action.detach().cpu().numpy()[0]
    
    def update(self, replay_buffer, batch_size=128):
        """
        Update the agent's networks using a batch from the replay buffer
        
        Args:
            replay_buffer: Replay buffer to sample from
            batch_size: Batch size for updates
            
        Returns:
            critic_loss: Critic loss value
            actor_loss: Actor loss value
        """
        if replay_buffer.size() < batch_size:
            return 0, 0
        
        # Sample from replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        new_action, log_prob = self.actor.sample(state)
        q1, q2 = self.critic(state, new_action)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_prob - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1 - self.tau) + param.data * self.tau
            )
        
        return critic_loss.item(), actor_loss.item()