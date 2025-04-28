import csv
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from .fragment import Fragment


def compute_weighted_score(match_contributions: List[int], 
                           weights: List[float], 
                           scale_factor: float = 1.0) -> float:
    """
    Calculate weighted match score
    
    Args:
        match_contributions: Feature match contribution list [text, damaged, layered, vertical]
        weights: Feature weights [w1, w2, w3, w4]
        scale_factor: Scaling factor to adjust feature score magnitude
        
    Returns:
        Weighted match score
    """
    return scale_factor * np.dot(match_contributions, weights)


def load_fragments_from_csv(filename: str, count: int = 118) -> Tuple[List[Fragment], List[Fragment]]:
    """
    Load fragment data from CSV file
    
    Args:
        filename: Path to the CSV file
        count: Number of fragments to load
        
    Returns:
        Tuple of (bottom_fragments, top_fragments)
    """
    fragments_data = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert numeric fields to integers
            for key in row:
                if key != 'fragment_id':
                    row[key] = int(row[key])
                else:
                    row[key] = int(row[key])
            fragments_data.append(row)

    bottom_fragments = []
    top_fragments = []
    for i in range(min(count, len(fragments_data))):
        # Read bottom fragment data
        text_features = fragments_data[i]['bottom_text_features']
        damaged = fragments_data[i]['bottom_damaged']
        layered = fragments_data[i]['bottom_layered']
        vertical_pattern = fragments_data[i]['bottom_vertical_pattern']

        bottom_fragment = Fragment(
            id=i+1,  # ID starts from 1
            text_features=text_features,
            damaged=damaged,
            layered=layered,
            vertical_pattern=vertical_pattern
        )
        bottom_fragments.append(bottom_fragment)

        # Read top fragment data
        text_features = fragments_data[i]['top_text_features']
        damaged = fragments_data[i]['top_damaged']
        layered = fragments_data[i]['top_layered']
        vertical_pattern = fragments_data[i]['top_vertical_pattern']

        top_fragment = Fragment(
            id=i+1,  # ID starts from 1
            text_features=text_features,
            damaged=damaged,
            layered=layered,
            vertical_pattern=vertical_pattern
        )
        top_fragments.append(top_fragment)
    
    return bottom_fragments, top_fragments


def generate_synthetic_fragments(count: int = 118) -> List[Fragment]:
    """
    Generate synthetic fragment data
    
    Args:
        count: Number of fragments to generate
        
    Returns:
        List of generated Fragment objects
    """
    fragments = []
    for i in range(count):
        # Generate features with certain probabilities
        text_features = random.random() > 0.6  # 40% chance of having text
        damaged = random.random() > 0.5       # 50% chance of damage
        layered = random.random() > 0.7       # 30% chance of layering
        vertical_pattern = random.random() > 0.8  # 20% chance of vertical pattern
        
        fragment = Fragment(
            id=i+1,  # ID starts from 1
            text_features=text_features,
            damaged=damaged,
            layered=layered,
            vertical_pattern=vertical_pattern
        )
        fragments.append(fragment)
    
    return fragments