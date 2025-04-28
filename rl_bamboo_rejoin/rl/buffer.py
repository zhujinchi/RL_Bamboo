from collections import deque
import random
import numpy as np

class ReplayBuffer:
    """
    Experience Replay Buffer
    
    Stores and samples transitions for reinforcement learning.
    """
    
    def __init__(self, capacity=10000):
        """
        Initialize replay buffer with given capacity
        
        Args:
            capacity: Maximum buffer capacity
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new transition to the buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Batch of transitions as numpy arrays
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def size(self):
        """
        Get current buffer size
        
        Returns:
            Number of transitions in buffer
        """
        return len(self.buffer)