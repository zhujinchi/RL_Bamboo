#!/usr/bin/env python
"""
Example script for training a SAC agent for bamboo slip fragment matching.
This demonstrates the basic training loop and visualization of results.
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
    visualize_training
)


def train_sac(env, total_timesteps=50000, batch_size=128, 
              eval_interval=1000, save_interval=5000, save_dir='resource/models'):
    """
    Train a SAC agent for fragment matching
    
    Args:
        env: Fragment matching environment
        total_timesteps: Total training steps
        batch_size: Batch size for updates
        eval_interval: Interval for evaluation
        save_interval: Interval for saving model
        save_dir: Directory to save models
    
    Returns:
        agent: Trained SAC agent
        rewards: List of episode rewards
        critic_losses: List of critic losses
        actor_losses: List of actor losses
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
            if critic_loss > 0:  # Avoid storing initial zero values
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)
        
        # If episode ends
        if done:
            rewards.append(total_reward)
            
            # Reset environment
            state = env.reset()
            total_reward = 0
            episode_count += 1
            
            # Update progress bar description
            avg_reward = np.mean(rewards[-100:]) if rewards else 0
            progress_bar.set_description(f"Episode {episode_count} | Avg Reward: {avg_reward:.2f}")
        
        # Periodic evaluation
        if t % eval_interval == 0:
            avg_improvement = env.get_avg_improvement()
            print(f"\nStep {t}: Average ranking improvement: {avg_improvement:.2f} positions")
        
        # Periodic model saving
        if t % save_interval == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(agent.actor.state_dict(), f'{save_dir}/sac_actor_{t}.pth')
            torch.save(agent.critic.state_dict(), f'{save_dir}/sac_critic_{t}.pth')
            print(f"Model saved at step {t}")
    
    return agent, rewards, critic_losses, actor_losses


def main():
    """Main function to run the training example"""
    start_time = time.time()
    
    # Load pre-trained edge matching models (if available)
    print("Loading pre-trained models...")
    f_model = VectorNet()
    c_model = CompareNet()
    
    try:
        f_model.load_state_dict(torch.load('resource/models/f_model.pth'))
        c_model.load_state_dict(torch.load('resource/models/c_model.pth'))
        print("Models loaded successfully")
    except:
        print("Unable to load pre-trained models, using randomly initialized models")
    
    # Generate or load distance matrix
    print("Generating distance matrix...")
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
        np.save("resource/distance_matrix.npy", dis_map)
    except:
        print("Unable to load real vector data, using random distance matrix")
        # Randomly generate a distance matrix
        num_fragments = 118
        dis_map = np.random.rand(num_fragments, num_fragments)
        # Ensure diagonal elements are smaller (more similar)
        for i in range(num_fragments):
            dis_map[i, i] = dis_map[i, i] * 0.1
    
    # Load or generate fragment data
    print("Loading fragment data...")
    try:
        fragments_bottom, fragments_top = load_fragments_from_csv('resource/bamboo_features.csv', 118)
        print(f"Loaded {len(fragments_top)} fragments")
    except:
        print("Unable to load fragment data, generating synthetic fragments")
        fragments_top = generate_synthetic_fragments(118)
        fragments_bottom = generate_synthetic_fragments(118)
    
    # Create environment with game theory adjustment
    env = FragmentMatchingEnv(
        fragments_top, 
        fragments_bottom, 
        dis_map, 
        alpha=0.5, 
        use_game_theory=True
    )
    
    # Train SAC model
    print("\nStarting SAC training with game theory adjustment...")
    agent, rewards, critic_losses, actor_losses = train_sac(
        env,
        total_timesteps=30000,  # Reduced for example
        batch_size=128,
        eval_interval=5000,
        save_interval=10000
    )
    
    # Visualize training results
    visualize_training(rewards, critic_losses, actor_losses)
    
    # Calculate training time
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds")
    
    return agent, rewards, critic_losses, actor_losses


if __name__ == "__main__":
    main()