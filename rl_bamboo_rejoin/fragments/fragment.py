from typing import List, Tuple, Optional

class Fragment:
    """
    Bamboo slip fragment class
    
    Represents a single bamboo slip fragment with its features.
    """
    
    def __init__(self, id: int, text_features: bool = False, damaged: bool = False, 
                 layered: bool = False, vertical_pattern: bool = False):
        """
        Initialize a fragment
        
        Args:
            id: Fragment unique identifier
            text_features: Whether it has text on the edge
            damaged: Whether it shows damage
            layered: Whether the break is layered
            vertical_pattern: Whether it has vertical grain pattern
        """
        self.id = id
        self.text_features = bool(text_features)
        self.damaged = bool(damaged)
        self.layered = bool(layered)
        self.vertical_pattern = bool(vertical_pattern)
    
    def feature_match_contribution(self, other_fragment: 'Fragment') -> List[int]:
        """
        Calculate feature match contributions with another fragment
        
        Returns a list of four feature match contributions
        
        Contribution rules:
        - Both fragments don't have feature: contribution = 0
        - One has feature, one doesn't: contribution = -1
        - Both have feature: contribution = 1
        
        Args:
            other_fragment: The fragment to compare with
            
        Returns:
            List of match contributions for each feature
        """
        features_match = [
            self._single_feature_match(self.text_features, other_fragment.text_features),
            self._single_feature_match(self.damaged, other_fragment.damaged),
            self._single_feature_match(self.layered, other_fragment.layered),
            self._single_feature_match(self.vertical_pattern, other_fragment.vertical_pattern)
        ]
        return features_match
    
    def _single_feature_match(self, feature1: bool, feature2: bool) -> int:
        """
        Calculate a single feature match contribution
        
        Args:
            feature1: First fragment's feature value
            feature2: Second fragment's feature value
            
        Returns:
            Match contribution value (0, -1, 1)
        """
        if not feature1 and not feature2:
            return 0
        elif feature1 and feature2:
            return 1
        else:
            return -1
    
    def __repr__(self) -> str:
        """Improved string representation"""
        features = [
            self.text_features, 
            self.damaged, 
            self.layered, 
            self.vertical_pattern
        ]
        return f"Fragment(id={self.id}, features={features})"