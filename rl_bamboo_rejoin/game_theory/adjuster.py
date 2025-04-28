import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import random


class GameTheoryAdjuster:
    """
    Game Theory-based Fragment Matching Score Adjuster
    
    This class implements a pluggable module for optimizing fragment matching results
    by applying game theory strategies after the original matching and feature adjustments.
    """
    
    def __init__(self, 
                 competition_intensity: float = 0.5,
                 risk_aversion: float = 0.3,
                 learning_rate: float = 0.1,
                 history_weight: float = 0.7,
                 strategy_options: List[str] = ["conservative", "neutral", "aggressive"],
                 random_seed: Optional[int] = None):
        """
        Initialize the game theory adjuster
        
        Args:
            competition_intensity: Competition intensity coefficient, affects strategy adjustment magnitude
            risk_aversion: Risk aversion coefficient, affects strategy conservativeness
            learning_rate: Learning rate, affects the update speed of historical strategies
            history_weight: History information weight, affects historical information's influence on current decisions
            strategy_options: List of available strategy options
            random_seed: Random seed for reproducibility
        """
        self.competition_intensity = competition_intensity
        self.risk_aversion = risk_aversion
        self.learning_rate = learning_rate
        self.history_weight = history_weight
        self.strategy_options = strategy_options
        
        # Initialize history records
        self.history = {}  # Store fragments' historical behaviors
        self.strategy_performance = {}  # Store strategies' historical performance
        
        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def adjust(self, 
               original_scores: List[Tuple[int, float]], 
               adjustment_scores: List[Tuple[int, float]],
               current_fragment_idx: int) -> List[Tuple[int, float]]:
        """
        Apply game theory strategies to adjust scores
        
        Args:
            original_scores: Original score list, each item is (index, score)
            adjustment_scores: Adjustment score list, each item is (index, score)
            current_fragment_idx: Index of the current fragment being considered
            
        Returns:
            Adjusted score list
        """
        # Analyze current fragment's position in original ranking
        original_ranked = sorted(original_scores, key=lambda x: x[1])  # Sort by distance ascending (smaller is more similar)
        current_orig_rank = next((i for i, (idx, _) in enumerate(original_ranked) if idx == current_fragment_idx), -1)
        
        # If current fragment not found, return original adjustment scores
        if current_orig_rank == -1:
            return adjustment_scores
        
        # Predict possible strategies of other fragments
        competitor_predictions = self._predict_competitors(
            original_scores, adjustment_scores, current_fragment_idx, original_ranked
        )
        
        # Get current fragment's adjustment score
        current_adjustment = next((score for idx, score in adjustment_scores if idx == current_fragment_idx), 0)
        
        # Generate possible strategy options
        strategy_options = self._generate_strategy_options(current_adjustment)
        
        # Select best strategy
        best_strategy, best_strategy_name = self._select_best_strategy(
            strategy_options,
            original_scores,
            adjustment_scores,
            current_fragment_idx,
            competitor_predictions,
            original_ranked,
            current_orig_rank
        )
        
        # Update history records
        self._update_history(current_fragment_idx, best_strategy_name, current_orig_rank)
        
        # Apply selected strategy
        new_adjustment_scores = [(idx, score) for idx, score in adjustment_scores]
        for i, (idx, _) in enumerate(new_adjustment_scores):
            if idx == current_fragment_idx:
                new_adjustment_scores[i] = (idx, best_strategy)
                break
        
        return new_adjustment_scores
    
    def _predict_competitors(self, 
                            original_scores: List[Tuple[int, float]],
                            adjustment_scores: List[Tuple[int, float]],
                            current_idx: int,
                            original_ranked: List[Tuple[int, float]]) -> Dict[int, Dict[str, float]]:
        """
        Predict competitors' possible strategies
        
        Args:
            original_scores: Original score list
            adjustment_scores: Adjustment score list
            current_idx: Current fragment index
            original_ranked: Original ranking list
            
        Returns:
            Dictionary of competitor predictions
        """
        competitor_predictions = {}
        current_rank = next((i for i, (idx, _) in enumerate(original_ranked) if idx == current_idx), -1)
        
        for idx, orig_score in original_scores:
            if idx == current_idx:
                continue  # Skip current fragment
                
            # Get competitor's position in original ranking
            competitor_rank = next((i for i, (j, _) in enumerate(original_ranked) if j == idx), -1)
            if competitor_rank == -1:
                continue
                
            # Calculate competition intensity (closer rankings mean more intense competition)
            rank_distance = abs(competitor_rank - current_rank)
            competition_factor = 1.0 / (1.0 + rank_distance) * self.competition_intensity
            
            # Consider historical behavior
            if idx in self.history:
                hist = self.history[idx]
                aggression = hist.get('aggression', 0.5)
                consistency = hist.get('consistency', 0.5)
            else:
                # Default to medium aggression and consistency
                aggression = 0.5
                consistency = 0.5
            
            # Predict adjustment magnitude
            adj = next((score for i, score in adjustment_scores if i == idx), 0)
            base_adjustment = abs(adj) if adj != 0 else 0.1
            
            # Adjustment magnitude is affected by competition intensity and aggression
            mean_adjustment = base_adjustment * (1.0 + competition_factor * aggression)
            
            # Uncertainty is affected by consistency
            std_dev = base_adjustment * (1.0 - consistency) * 0.3
            
            competitor_predictions[idx] = {
                'mean': mean_adjustment,
                'std_dev': std_dev
            }
            
        return competitor_predictions
    
    def _generate_strategy_options(self, current_adjustment: float) -> Dict[str, float]:
        """
        Generate strategy options
        
        Args:
            current_adjustment: Current adjustment score
            
        Returns:
            Dictionary of strategy options
        """
        base_value = abs(current_adjustment) if current_adjustment != 0 else 0.1
        sign = 1 if current_adjustment >= 0 else -1
        
        strategy_options = {
            "conservative": current_adjustment * 0.8,  # Conservative strategy
            "neutral": current_adjustment,             # Neutral strategy
            "aggressive": current_adjustment * 1.2,    # Aggressive strategy
            "highly_aggressive": current_adjustment * 1.5  # Highly aggressive strategy
        }
        
        # Filter strategy options based on initialized strategy options
        return {k: v for k, v in strategy_options.items() if k in self.strategy_options}
    
    def _select_best_strategy(self,
                             strategy_options: Dict[str, float],
                             original_scores: List[Tuple[int, float]],
                             adjustment_scores: List[Tuple[int, float]],
                             current_idx: int,
                             competitor_predictions: Dict[int, Dict[str, float]],
                             original_ranked: List[Tuple[int, float]],
                             current_orig_rank: int) -> Tuple[float, str]:
        """
        Select the best strategy
        
        Args:
            strategy_options: Strategy options dictionary
            original_scores: Original score list
            adjustment_scores: Adjustment score list
            current_idx: Current fragment index
            competitor_predictions: Competitor predictions
            original_ranked: Original ranking list
            current_orig_rank: Current fragment's original rank
            
        Returns:
            Tuple of (selected strategy value, strategy name)
        """
        best_utility = float('-inf')
        best_strategy = None
        best_strategy_name = None
        
        # Calculate utility for each strategy
        for strategy_name, strategy_value in strategy_options.items():
            # Simulate outcome after using this strategy
            simulated_ranking = self._simulate_ranking(
                original_scores,
                adjustment_scores,
                current_idx,
                strategy_value,
                competitor_predictions
            )
            
            # Calculate utility
            utility = self._compute_utility(
                simulated_ranking,
                current_idx,
                current_orig_rank,
                strategy_name
            )
            
            # Consider historical strategy performance
            if strategy_name in self.strategy_performance:
                historical_performance = self.strategy_performance[strategy_name]
                utility = (1 - self.history_weight) * utility + self.history_weight * historical_performance
            
            if utility > best_utility:
                best_utility = utility
                best_strategy = strategy_value
                best_strategy_name = strategy_name
        
        # If no best strategy found, use default strategy
        if best_strategy is None:
            best_strategy = next(iter(strategy_options.values()))
            best_strategy_name = next(iter(strategy_options.keys()))
            
        return best_strategy, best_strategy_name
    
    def _simulate_ranking(self,
                         original_scores: List[Tuple[int, float]],
                         adjustment_scores: List[Tuple[int, float]],
                         current_idx: int,
                         strategy_value: float,
                         competitor_predictions: Dict[int, Dict[str, float]],
                         alpha: float = 0.5) -> List[Tuple[int, float]]:
        """
        Simulate final ranking
        
        Args:
            original_scores: Original score list
            adjustment_scores: Adjustment score list
            current_idx: Current fragment index
            strategy_value: Strategy value to apply
            competitor_predictions: Competitor predictions
            alpha: Balance coefficient
            
        Returns:
            Simulated ranking list
        """
        # Calculate maximum original score (for normalization)
        max_orig_score = max(score for _, score in original_scores)
        
        # Build simulated final scores
        final_scores = []
        for idx, orig_score in original_scores:
            # Original similarity (converted to larger is more similar)
            orig_similarity = 1.0 - orig_score / max_orig_score
            
            if idx == current_idx:
                # Use our strategy value
                adj_score = strategy_value
            elif idx in competitor_predictions:
                # Use predicted competitor adjustment value
                pred = competitor_predictions[idx]
                # Sample from distribution
                adj_score = np.random.normal(pred['mean'], pred['std_dev'])
            else:
                # Use original adjustment score
                adj_score = next((score for i, score in adjustment_scores if i == idx), 0)
                
            # Calculate final score (weighted combination)
            final_score = alpha * adj_score + (1 - alpha) * orig_similarity
            final_scores.append((idx, final_score))
        
        # Sort (descending, larger value means better)
        ranked = sorted(final_scores, key=lambda x: x[1], reverse=True)
        return ranked
    
    def _compute_utility(self,
                        ranked_list: List[Tuple[int, float]],
                        current_idx: int,
                        original_rank: int,
                        strategy_name: str) -> float:
        """
        Compute utility function
        
        Args:
            ranked_list: Ranked list
            current_idx: Current fragment index
            original_rank: Original rank
            strategy_name: Strategy name
            
        Returns:
            Utility value
        """
        # Find position in new ranking
        new_rank = next((i for i, (idx, _) in enumerate(ranked_list) if idx == current_idx), -1)
        if new_rank == -1:
            return float('-inf')
            
        # Base utility: rank improvement
        rank_improvement = original_rank - new_rank
        base_utility = rank_improvement * 10  # 10 points for each rank improvement
        
        # Additional bonus: entering top N positions
        top_bonus = 0
        if new_rank < 5:  # Extra bonus for top 5
            top_bonus = (5 - new_rank) * 15
        
        # Risk penalty: based on risk aversion coefficient and strategy
        risk_factor = 0
        if strategy_name == "aggressive":
            risk_factor = 0.5
        elif strategy_name == "highly_aggressive":
            risk_factor = 1.0
            
        risk_penalty = rank_improvement * risk_factor * self.risk_aversion * 5
        
        # Calculate total utility
        total_utility = base_utility + top_bonus - risk_penalty
        
        return total_utility
    
    def _update_history(self, fragment_idx: int, strategy_name: str, original_rank: int) -> None:
        """
        Update history records
        
        Args:
            fragment_idx: Fragment index
            strategy_name: Strategy name
            original_rank: Original rank
        """
        # If new fragment, initialize history record
        if fragment_idx not in self.history:
            self.history[fragment_idx] = {
                'strategies': [],
                'aggression': 0.5,
                'consistency': 0.5,
                'recent_ranks': []
            }
        
        # Update strategy history
        self.history[fragment_idx]['strategies'].append(strategy_name)
        
        # Keep only recent N strategies
        max_history = 10
        if len(self.history[fragment_idx]['strategies']) > max_history:
            self.history[fragment_idx]['strategies'] = self.history[fragment_idx]['strategies'][-max_history:]
        
        # Update aggression (based on frequency of choosing aggressive strategies)
        strategies = self.history[fragment_idx]['strategies']
        aggressive_count = sum(1 for s in strategies if s in ['aggressive', 'highly_aggressive'])
        aggression = aggressive_count / len(strategies) if strategies else 0.5
        
        # Smooth update
        old_aggression = self.history[fragment_idx]['aggression']
        self.history[fragment_idx]['aggression'] = (1 - self.learning_rate) * old_aggression + self.learning_rate * aggression
        
        # Update consistency (based on strategy selection consistency)
        if len(strategies) >= 3:
            # Calculate proportion of most common strategy
            strategy_counts = {}
            for s in strategies:
                strategy_counts[s] = strategy_counts.get(s, 0) + 1
            most_common = max(strategy_counts.values())
            consistency = most_common / len(strategies)
        else:
            consistency = 0.5
            
        # Smooth update
        old_consistency = self.history[fragment_idx]['consistency']
        self.history[fragment_idx]['consistency'] = (1 - self.learning_rate) * old_consistency + self.learning_rate * consistency
        
        # Update rank history
        self.history[fragment_idx]['recent_ranks'].append(original_rank)
        if len(self.history[fragment_idx]['recent_ranks']) > max_history:
            self.history[fragment_idx]['recent_ranks'] = self.history[fragment_idx]['recent_ranks'][-max_history:]
        
        # Update strategy performance
        if len(self.history[fragment_idx]['recent_ranks']) >= 2:
            prev_rank = self.history[fragment_idx]['recent_ranks'][-2]
            current_rank = original_rank
            rank_change = prev_rank - current_rank  # Positive value means rank improvement
            
            # Last used strategy
            if len(strategies) >= 2:
                prev_strategy = strategies[-2]
                
                # Update strategy performance
                if prev_strategy not in self.strategy_performance:
                    self.strategy_performance[prev_strategy] = 0
                
                # Smooth update
                old_performance = self.strategy_performance[prev_strategy]
                # Convert rank change to utility
                utility_change = rank_change * 10 + (5 if current_rank < 5 else 0)
                self.strategy_performance[prev_strategy] = (1 - self.learning_rate) * old_performance + self.learning_rate * utility_change