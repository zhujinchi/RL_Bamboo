import numpy as np
import random
from typing import Tuple, Dict, List, Any, Optional

from ..fragments.fragment import Fragment
from ..game_theory.adjuster import GameTheoryAdjuster
from ..fragments.utils import compute_weighted_score


class FragmentMatchingEnv:
    """
    Bamboo slip fragment matching environment
    
    This environment simulates the bamboo slip fragment matching process for reinforcement learning,
    supporting both basic matching and game theory enhanced matching logic.
    """
    
    def __init__(self, 
                 fragments_top: List[Fragment], 
                 fragments_bottom: List[Fragment], 
                 dis_map: List[List[float]], 
                 alpha: float = 0.5, 
                 use_game_theory: bool = True,
                 fixed_fragment_idx: Optional[int] = None):
        """
        Initialize the matching environment
        
        Args:
            fragments_top: List of top fragments
            fragments_bottom: List of bottom fragments
            dis_map: Distance matrix representing dissimilarity between top and bottom fragments
            alpha: Balance coefficient for feature adjustment scores and edge similarity
            use_game_theory: Whether to use game theory adjustment
            fixed_fragment_idx: If provided, environment will always use this fragment for training
        """
        self.fragments_top = fragments_top
        self.fragments_bottom = fragments_bottom
        self.dis_map = dis_map
        self.alpha = alpha
        self.num_fragments = len(fragments_top)
        self.current_top_idx = 0
        self.recent_improvements = []
        self.fixed_fragment_idx = fixed_fragment_idx
        
        # Initialize game theory adjuster (only when use_game_theory=True)
        self.use_game_theory = use_game_theory
        if use_game_theory:
            self.game_theory_adjuster = GameTheoryAdjuster(
                competition_intensity=0.6,  # Adjustable parameter
                risk_aversion=0.5,          # Adjustable parameter
                strategy_options=["conservative", "neutral", "aggressive", "highly_aggressive"]
            )
        
        # Reset environment
        self.reset()
    
    def _get_state(self, top_idx: int, bottom_idx: int) -> np.ndarray:
        """
        Construct state representation
        
        Args:
            top_idx: Index of top fragment
            bottom_idx: Index of bottom fragment
            
        Returns:
            NumPy array representing the state
        """
        top_fragment = self.fragments_top[top_idx]
        bottom_fragment = self.fragments_bottom[bottom_idx]
        
        # Merge features of both fragments into a state representation
        match_contributions = top_fragment.feature_match_contribution(bottom_fragment)
        
        # Add original features and match contributions
        state = [
            int(top_fragment.text_features),
            int(top_fragment.damaged),
            int(top_fragment.layered),
            int(top_fragment.vertical_pattern),
            int(bottom_fragment.text_features),
            int(bottom_fragment.damaged),
            int(bottom_fragment.layered),
            int(bottom_fragment.vertical_pattern),
            # Add match contribution features - this is the core fusion operation
            match_contributions[0],  # Text feature match contribution
            match_contributions[1],  # Damage feature match contribution
            match_contributions[2],  # Layering feature match contribution
            match_contributions[3]   # Vertical pattern match contribution
        ]
        return np.array(state, dtype=np.float32)
    
    def reset(self) -> np.ndarray:
        """
        Reset environment, randomly select a top fragment or use fixed fragment if specified
        
        Returns:
            Initial state
        """
        if self.fixed_fragment_idx is not None:
            # Use the fixed fragment index
            self.current_top_idx = self.fixed_fragment_idx
        else:
            # Randomly select a fragment
            self.current_top_idx = random.randint(0, self.num_fragments - 1)
            
        state = self._get_state(self.current_top_idx, self.current_top_idx)  # Initial state uses diagonal element
        return state
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step of action, calculate reward and next state
        
        Args:
            action: Action vector containing weights and scaling factor
            
        Returns:
            next_state: Next state
            reward: Reward value
            done: Whether the episode is finished
            info: Additional information
        """
        # Action contains weights and scaling factor
        raw_weights = action[:-1]  # First 4 are weights
        raw_scale = action[-1]     # Last one is scaling factor
        
        # Convert weights and scaling factor to appropriate ranges
        weights = (raw_weights + 1) / 2  # Convert to [0,1]
        sum_weights = np.sum(weights)
        weights = weights / sum_weights if sum_weights > 0 else np.ones(4) / 4
        
        scale_factor = (raw_scale + 1) * 5  # Convert to [0,10]
        
        # Get current top fragment
        top_fragment = self.fragments_top[self.current_top_idx]
        
        # Calculate original ranking
        row_scores = [(j, self.dis_map[self.current_top_idx][j]) for j in range(self.num_fragments)]
        original_ranklist = sorted(row_scores, key=lambda x: x[1])  # Sort by distance ascending (lower is more similar)
        
        # Calculate new ranking
        adjustment_scores = []
        for j in range(self.num_fragments):
            match_contributions = top_fragment.feature_match_contribution(self.fragments_bottom[j])
            feature_score = compute_weighted_score(match_contributions, weights, scale_factor)
            adjustment_scores.append((j, feature_score))
        
        # Apply game theory adjustment (if enabled)
        if self.use_game_theory:
            # Use game theory adjuster to process adjustment_scores
            adjustment_scores = self.game_theory_adjuster.adjust(
                row_scores,               # Original distance scores
                adjustment_scores,        # Feature adjustment scores
                self.current_top_idx      # Current fragment index
            )
        
        # Combine two lists to get final ranking
        final_scores = []
        for j in range(self.num_fragments):
            original_score = next(score for idx, score in row_scores if idx == j)
            # Convert to similarity (higher is more similar)
            original_similarity = 1.0 - original_score / np.max(self.dis_map)  
            
            adjustment_score = next(score for idx, score in adjustment_scores if idx == j)
            
            # Weighted combination score
            final_score = self.alpha * adjustment_score + (1 - self.alpha) * original_similarity
            final_scores.append((j, final_score))
        
        final_ranklist = sorted(final_scores, key=lambda x: x[1], reverse=True)  # Sort descending
        
        # Find ideal match position in rankings
        ideal_match_idx = self.current_top_idx  # Diagonal element
        original_pos = next((i for i, (idx, _) in enumerate(original_ranklist) if idx == ideal_match_idx), -1)
        final_pos = next((i for i, (idx, _) in enumerate(final_ranklist) if idx == ideal_match_idx), -1)
        
        # Calculate reward
        if original_pos != -1 and final_pos != -1:
            rank_improvement = original_pos - final_pos
            reward = np.tanh(rank_improvement * 0.5)
            
            if final_pos < 3:  # If ranked in top 3
                bonus = (3 - final_pos) * 0.2
                reward += bonus
            
            # Record rank improvement
            self.recent_improvements.append(rank_improvement)
        else:
            reward = -0.5  # Negative reward if match not found
            rank_improvement = 0
            
        # Select next fragment based on fixed or random selection
        if self.fixed_fragment_idx is not None:
            # When using fixed fragment, keep using the same fragment
            next_top_idx = self.fixed_fragment_idx
        else:
            # When not fixed, cycle through fragments
            next_top_idx = (self.current_top_idx + 1) % self.num_fragments
            
        self.current_top_idx = next_top_idx
        next_state = self._get_state(next_top_idx, next_top_idx)  # Next state also uses diagonal element
        
        # Each episode runs for a limited number of steps, simplified as random termination
        done = random.random() < 0.02  # 2% chance of ending
        
        # Provide additional information
        info = {
            'rank_improvement': rank_improvement,
            'original_pos': original_pos,
            'final_pos': final_pos,
            'weights': weights,
            'scale_factor': scale_factor
        }
        
        return next_state, reward, done, info
    
    def get_avg_improvement(self) -> float:
        """
        Get recent average rank improvement
        
        Returns:
            Average rank improvement value
        """
        if len(self.recent_improvements) > 0:
            return np.mean(self.recent_improvements)
        return 0.0