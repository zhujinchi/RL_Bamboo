#!/usr/bin/env python
"""
Example script for analyzing the weights learned by a trained SAC agent.
This demonstrates how to load a trained model and visualize what it has learned.
"""

import os
import argparse
import torch
import numpy as np

from rl_bamboo_rejoin import (
    load_fragments_from_csv, generate_synthetic_fragments,
    FragmentMatchingEnv, SACAgent,
    analyze_learned_weights
)


def load_agent(model_path, state_dim, action_dim, hidden_dim=128):
    """
    Load a trained agent from a saved model file
    
    Args:
        model_path: Path to the saved model file
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Dimension of hidden layers
        
    Returns:
        agent: Loaded SAC agent
    """
    agent = SACAgent(state_dim, action_dim, hidden_dim)
    
    try:
        agent.actor.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
    return agent


def main():
    """Main function to run the weight analysis example"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze learned weights of a SAC agent')
    parser.add_argument('--model_path', type=str, default='resource/models/sac_actor_30000.pth',
                      help='Path to the saved model file')
    parser.add_argument('--data_path', type=str, default='resource/data/bamboo_features.csv',
                      help='Path to the fragment data CSV file')
    parser.add_argument('--num_samples', type=int, default=50,
                      help='Number of sample states to analyze')
    args = parser.parse_args()
    
    # Load fragment data
    print("Loading fragment data...")
    try:
        fragments_bottom, fragments_top = load_fragments_from_csv(args.data_path, 118)
        print(f"Loaded {len(fragments_top)} fragments")
    except:
        print("Unable to load fragment data, generating synthetic fragments")
        fragments_top = generate_synthetic_fragments(118)
        fragments_bottom = generate_synthetic_fragments(118)
    
    # Create an environment to determine state dimensions
    env = FragmentMatchingEnv(fragments_top, fragments_bottom, [], use_game_theory=False)
    state = env.reset()
    state_dim = len(state)
    action_dim = 5  # 4 weights + 1 scaling factor
    
    # Load trained agent
    print(f"Loading trained agent from {args.model_path}...")
    agent = load_agent(args.model_path, state_dim, action_dim)
    
    if agent is None:
        print("Failed to load agent, exiting")
        return
    
    # Analyze learned weights
    print(f"Analyzing weights across {args.num_samples} samples...")
    analysis_results = analyze_learned_weights(
        agent, fragments_top, fragments_bottom, num_samples=args.num_samples
    )
    
    # Print feature importance
    feature_names = ["Text Features", "Damage Features", "Layering Features", "Vertical Pattern"]
    importance_rank = np.argsort(analysis_results['avg_weights'])[::-1]
    
    print("\nFeature Importance Ranking:")
    for i, idx in enumerate(importance_rank):
        print(f"{i+1}. {feature_names[idx]}: {analysis_results['avg_weights'][idx]:.4f} ± {analysis_results['std_weights'][idx]:.4f}")
    
    print(f"\nScale Factor: {analysis_results['avg_scale']:.4f} ± {analysis_results['std_scale']:.4f}")
    
    # Analyze weight correlations
    weight_matrix = np.array(analysis_results['weight_records'])
    corr_matrix = np.corrcoef(weight_matrix.T)
    
    print("\nWeight Correlations:")
    for i in range(4):
        for j in range(i+1, 4):
            print(f"{feature_names[i]} vs {feature_names[j]}: {corr_matrix[i,j]:.4f}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()