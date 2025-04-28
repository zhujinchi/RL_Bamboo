#!/usr/bin/env python
"""
Example script comparing fragment matching performance with and without game theory.
This demonstrates how to set up both environments and compare their results.
"""

import os
import time
import numpy as np
import torch
from tqdm import tqdm

from rl_bamboo_rejoin import (
    VectorNet, CompareNet, inference,
    load_fragments_from_csv, generate_synthetic_fragments,
    FragmentMatchingEnv, SACAgent, ReplayBuffer,
    compare_results
)


def train_sac(env, total_timesteps=20000, batch_size=128, 
              eval_interval=1000, save_interval=5000, model_prefix=""):
    """
    Train a SAC agent for fragment matching
    
    Args:
        env: Fragment matching environment
        total_timesteps: Total training steps
        batch_size: Batch size for updates
        eval_interval: Interval for evaluation
        save_interval: Interval for saving model
        model_prefix: Prefix for saved model files
    
    Returns:
        agent: Trained SAC agent
        rewards: List of episode rewards
    """
    # Environment parameters
    state = env.reset()
    state_dim = len(state)
    action_dim = 5  # 4 weights + 1 scaling factor
    
    # Create SAC Agent
    agent = SACAgent(state_dim, action_dim, hidden_dim=128, 
                     alpha=0.2, gamma=0.99, tau=0.005, lr=3e-4)
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=100000)
    
    # Track training progress
    rewards = []
    total_reward = 0
    episode_count = 0
    
    critic_losses = []
    actor_losses = []
    
    # Main training loop
    progress_bar = tqdm(range(1, total_timesteps + 1))
    for t in progress_bar:
        # Initially choose random actions to fill experience buffer
        if t < 5000:
            action = np.random.uniform(-1, 1, size=action_dim)
        else:
            action = agent.select_action(state)
            
        # Environment interaction
        next_state, reward, done, info = env.step(action)
        
        # Store experience
        replay_buffer.add(state, action, reward, next_state, float(done))
        
        state = next_state
        total_reward += reward
        
        # Update SAC Agent
        if t >= 5000 and t % 1 == 0:  # Update every step
            critic_loss, actor_loss = agent.update(replay_buffer, batch_size)
        
        # If episode ends
        if done:
            rewards.append(total_reward)
            
            # Reset environment
            state = env.reset()
            total_reward = 0
            episode_count += 1
            
            # Update progress bar description
            avg_reward = np.mean(rewards[-100:]) if rewards else 0
            progress_bar.set_description(f"{model_prefix} Episode {episode_count} | Avg Reward: {avg_reward:.2f}")
        
        # Periodic evaluation
        if t % eval_interval == 0:
            avg_improvement = env.get_avg_improvement()
            print(f"\n{model_prefix} Step {t}: Average ranking improvement: {avg_improvement:.2f} positions")
        
        # Periodic model saving
        if t % save_interval == 0:
            save_dir = 'resource/models'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if model_prefix:
                torch.save(agent.actor.state_dict(), f'{save_dir}/{model_prefix}_actor_{t}.pth')
                torch.save(agent.critic.state_dict(), f'{save_dir}/{model_prefix}_critic_{t}.pth')
                print(f"{model_prefix} model saved at step {t}")
    
    return agent, rewards


def prepare_environment(use_game_theory=True):
    """
    Prepare the fragment matching environment
    
    Args:
        use_game_theory: Whether to use game theory adjustment
        
    Returns:
        env: Fragment matching environment
        fragments_top: List of top fragments
        fragments_bottom: List of bottom fragments
        dis_map: Distance matrix
    """
    # Load pre-trained edge matching models (if available)
    f_model = VectorNet()
    c_model = CompareNet()
    
    try:
        f_model.load_state_dict(torch.load('resource/models/f_model.pth'))
        c_model.load_state_dict(torch.load('resource/models/c_model.pth'))
    except:
        print("Unable to load pre-trained models, using randomly initialized models")
    
    # Generate or load distance matrix
    try:
        # Try to load real vector data
        real_vectors = np.load("resource/vector_real_118_patch.npy")
        real_vectors = real_vectors[:, 2:4, :].astype(np.float32)
        
        length = len(real_vectors)
        dis_map = []
        for i in range(length):
            dis_list = []
            for j in range(length):
                v1 = real_vectors[i, 0, :][np.newaxis, np.newaxis, :]
                v2 = real_vectors[j, 1, :][np.newaxis, np.newaxis, :]
                v1[0][0] = [x-v1.min() for x in v1[0][0]]
                v2[0][0] = [x-v2.min() for x in v2[0][0]]
                v1 /= v1.max()+1e-6
                v2 /= v2.max()+1e-6
                v1 = torch.tensor(v1, dtype=torch.float32)
                v2 = torch.tensor(v2, dtype=torch.float32)
                dis = inference(f_model, c_model, v1, v2)
                dis_list.append(dis.item())
            dis_map.append(dis_list)
    except:
        print("Unable to load real vector data, using random distance matrix")
        # Randomly generate a distance matrix
        num_fragments = 118
        dis_map = np.random.rand(num_fragments, num_fragments)
        # Ensure diagonal elements are smaller (more similar)
        for i in range(num_fragments):
            dis_map[i, i] = dis_map[i, i] * 0.1
    
    # Load or generate fragment data
    try:
        fragments_bottom, fragments_top = load_fragments_from_csv('resource/bamboo_features.csv', 118)
    except:
        print("Unable to load fragment data, generating synthetic fragments")
        fragments_top = generate_synthetic_fragments(118)
        fragments_bottom = generate_synthetic_fragments(118)
    
    # Create environment
    env = FragmentMatchingEnv(
        fragments_top, 
        fragments_bottom, 
        dis_map, 
        alpha=0.5, 
        use_game_theory=use_game_theory
    )
    
    return env, fragments_top, fragments_bottom, dis_map


def main():
    """Main function to run the comparison example"""
    start_time = time.time()
    
    print("Setting up environments...")
    # Create environment with game theory adjustment
    env_with_game_theory, fragments_top, fragments_bottom, dis_map = prepare_environment(use_game_theory=True)
    
    # Create environment without game theory adjustment
    env_without_game_theory, _, _, _ = prepare_environment(use_game_theory=False)
    
    # Train SAC model with game theory
    print("\nTraining model with game theory adjustment...")
    agent_with_gt, rewards_with_gt = train_sac(
        env_with_game_theory,
        total_timesteps=9000,
        model_prefix="with_gt"
    )
    
    # Train SAC model without game theory
    print("\nTraining model without game theory adjustment...")
    agent_without_gt, rewards_without_gt = train_sac(
        env_without_game_theory,
        total_timesteps=9000,
        model_prefix="without_gt"
    )
    
    # Compare results
    print("\nComparing results...")
    comparison = compare_results(rewards_with_gt, rewards_without_gt)
    
    # Calculate total time
    total_time = time.time() - start_time
    print(f"\nTotal comparison time: {total_time:.2f} seconds")
    
    return comparison


if __name__ == "__main__":
    main()