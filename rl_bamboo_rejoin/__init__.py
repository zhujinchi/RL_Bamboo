"""
Bamboo Matching - A Game-Theoretic Reinforcement Learning Framework for Bamboo Slip Fragment Matching

This package implements a framework that combines Soft Actor-Critic reinforcement learning with 
game theory to solve the challenging problem of ancient bamboo slip fragment matching.
"""

__version__ = '0.1.0'
__author__ = 'Jinchi Zhu'

# Import key components to make them available at the package level
from .fragments import Fragment, compute_weighted_score, load_fragments_from_csv, generate_synthetic_fragments
from .models import VectorNet, CompareNet, inference
from .rl import SACAgent, ReplayBuffer, Actor, Critic
from .game_theory import GameTheoryAdjuster
from .environment import FragmentMatchingEnv
from .visualization import visualize_training, analyze_learned_weights, compare_results

# Define what should be available when using "from rl_bamboo_rejoin import *"
__all__ = [
    # Fragment-related
    'Fragment', 'compute_weighted_score', 'load_fragments_from_csv', 'generate_synthetic_fragments',
    
    # Model-related
    'VectorNet', 'CompareNet', 'inference',
    
    # RL-related
    'SACAgent', 'ReplayBuffer', 'Actor', 'Critic',
    
    # Game theory-related
    'GameTheoryAdjuster',
    
    # Environment-related
    'FragmentMatchingEnv',
    
    # Visualization-related
    'visualize_training', 'analyze_learned_weights', 'compare_results'
]