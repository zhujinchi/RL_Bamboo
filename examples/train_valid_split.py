#!/usr/bin/env python
"""
100对训练，18对测试
Training script for bamboo slip fragment matching with train-validation split.
Uses a subset of fragments for training and evaluates on the remaining fragments.
Uses a fixed random seed for reproducibility.
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import pandas as pd

from rl_bamboo_rejoin import (
    VectorNet, CompareNet, inference,
    load_fragments_from_csv, generate_synthetic_fragments,
    FragmentMatchingEnv, SACAgent, ReplayBuffer
)


class TrainValidSplitTrainer:
    """
    Trainer that uses a train-validation split approach for fragment matching.
    """
    def __init__(self, 
                 fragments_top, fragments_bottom, dis_map, 
                 train_size=100, 
                 total_timesteps=50000, 
                 batch_size=128, 
                 alpha=0.5, 
                 use_game_theory=True, 
                 save_dir='resource/train_valid_split',
                 random_seed=3813):  # 添加固定随机种子参数
        """
        Initialize the trainer with train-validation split.
        
        Args:
            fragments_top: List of top fragments
            fragments_bottom: List of bottom fragments
            dis_map: Distance matrix
            train_size: Number of fragments to use for training
            total_timesteps: Total training steps
            batch_size: Batch size for SAC updates
            alpha: Weight factor for combining scores
            use_game_theory: Whether to use game theory adjustment
            save_dir: Directory to save models and results
            random_seed: Fixed random seed for reproducibility
        """
        self.fragments_top = fragments_top
        self.fragments_bottom = fragments_bottom
        self.dis_map = dis_map
        self.num_fragments = len(fragments_top)
        
        if train_size >= self.num_fragments:
            train_size = int(self.num_fragments * 0.8)  # Default to 80% if train_size too large
            print(f"Train size adjusted to {train_size} (80% of data)")
            
        self.train_size = train_size
        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.alpha = alpha
        self.use_game_theory = use_game_theory
        self.save_dir = save_dir
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
        
        # Create output directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Split data into train and validation sets
        self._create_train_validation_split()
        
        # Initialize SAC agent
        state_dim = 12  # Based on the state representation in FragmentMatchingEnv
        action_dim = 5  # 4 feature weights + 1 scaling factor
        self.agent = SACAgent(state_dim, action_dim, hidden_dim=128, 
                         alpha=0.2, gamma=0.99, tau=0.005, lr=3e-4)
        
        # Initialize metrics tracking
        self.training_metrics = {
            'rewards': [],
            'avg_rewards': [],
            'critic_losses': [],
            'actor_losses': [],
            'train_improvements': [],
            'validation_improvements': []
        }
    
    def _create_train_validation_split(self):
        """
        Create train and validation index splits using fixed random seed.
        """
        # Create indices for all fragments
        all_indices = list(range(self.num_fragments))
        
        # Randomly select train indices (using fixed seed from class initialization)
        self.train_indices = sorted(random.sample(all_indices, self.train_size))
        
        # Remaining indices become validation set
        self.validation_indices = sorted(list(set(all_indices) - set(self.train_indices)))
        
        print(f"Created train-validation split: "
              f"{len(self.train_indices)} for training, "
              f"{len(self.validation_indices)} for validation")
        
        # Save the split to a file for reference
        split_df = pd.DataFrame({
            'train_indices': self.train_indices + [None] * (len(self.validation_indices) - len(self.train_indices)),
            'validation_indices': self.validation_indices + [None] * (len(self.train_indices) - len(self.validation_indices))
        })
        split_df.to_csv(f'{self.save_dir}/train_validation_split_seed{self.random_seed}.csv', index=False)
    
    def train(self):
        """
        Train the model on the training set fragments.
        
        Returns:
            Dictionary of training metrics
        """
        # Create environment that only samples from training fragments
        env = self._create_training_environment()
        
        # Initialize training variables
        replay_buffer = ReplayBuffer(capacity=100000)
        rewards = []
        episode_reward = 0
        episode_count = 0
        
        critic_losses = []
        actor_losses = []
        
        # Get initial state
        state = env.reset()
        
        # Training loop
        progress_bar = tqdm(range(1, self.total_timesteps + 1))
        for t in progress_bar:
            # Initially choose random actions to fill buffer
            if t < 5000:
                action = np.random.uniform(-1, 1, size=5)
            else:
                action = self.agent.select_action(state)
                
            # Environment interaction
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            replay_buffer.add(state, action, reward, next_state, float(done))
            
            state = next_state
            episode_reward += reward
            
            # Update SAC Agent
            if t >= 5000 and t % 1 == 0:  # Update every step after initial exploration
                critic_loss, actor_loss = self.agent.update(replay_buffer, self.batch_size)
                if critic_loss > 0:
                    critic_losses.append(critic_loss)
                    actor_losses.append(actor_loss)
            
            # If episode ends
            if done:
                rewards.append(episode_reward)
                
                # Reset environment
                state = env.reset()
                episode_reward = 0
                episode_count += 1
                
                # Update progress bar description
                avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                progress_bar.set_description(f"Episode {episode_count} | Avg Reward: {avg_reward:.2f}")
            
            # Periodically evaluate on validation set
            if t % 1000 == 0 or t == self.total_timesteps:
                train_improvement = env.get_avg_improvement()
                validation_improvement = self.evaluate_on_validation_set()
                
                print(f"\nStep {t}")
                print(f"Train set average rank improvement: {train_improvement:.2f}")
                print(f"Validation set average rank improvement: {validation_improvement:.2f}")
                
                self.training_metrics['train_improvements'].append(train_improvement)
                self.training_metrics['validation_improvements'].append(validation_improvement)
                
                # Save checkpoint
                self._save_checkpoint(t)
        
        # Save final training metrics
        self.training_metrics['rewards'] = rewards
        
        if len(rewards) >= 100:
            # Calculate rolling average with window size 100
            rolling_avg = [np.mean(rewards[max(0, i-100):i+1]) for i in range(len(rewards))]
            self.training_metrics['avg_rewards'] = rolling_avg
        else:
            self.training_metrics['avg_rewards'] = rewards.copy()
            
        self.training_metrics['critic_losses'] = critic_losses
        self.training_metrics['actor_losses'] = actor_losses
        
        # Save final model
        self._save_checkpoint('final')
        
        # Save metrics
        self._save_metrics()
        
        # Update distance matrix using the trained model
        print("\nTraining complete. Updating distance matrix...")
        updated_dis_map = self.update_distance_matrix()
        
        # Save updated distance matrix in metrics
        self.training_metrics['updated_dis_map'] = 'saved'  # Just indicate it was saved
        
        return self.training_metrics
        
    
    def _create_training_environment(self):
        """
        Create environment that only samples from training fragments.
        
        Returns:
            Environment instance that samples only from training set
        """
        class TrainingFragmentEnv(FragmentMatchingEnv):
            def __init__(self, base_env, train_indices):
                # Copy all attributes from base_env
                self.__dict__.update(base_env.__dict__)
                self.train_indices = train_indices
            
            def reset(self):
                # Only sample from training indices
                self.current_top_idx = random.choice(self.train_indices)
                state = self._get_state(self.current_top_idx, self.current_top_idx)
                return state
            
            def step(self, action):
                next_state, reward, done, info = super().step(action)
                # Ensure next fragment is from training set
                if self.current_top_idx not in self.train_indices:
                    self.current_top_idx = random.choice(self.train_indices)
                    next_state = self._get_state(self.current_top_idx, self.current_top_idx)
                return next_state, reward, done, info
        
        # Create base environment
        base_env = FragmentMatchingEnv(
            self.fragments_top,
            self.fragments_bottom,
            self.dis_map,
            alpha=self.alpha,
            use_game_theory=self.use_game_theory
        )
        
        # Wrap with training-only environment
        return TrainingFragmentEnv(base_env, self.train_indices)
    
    def evaluate_on_validation_set(self):
        """
        Evaluate current model on validation set fragments.
        
        Returns:
            Average rank improvement on validation set
        """
        if not self.validation_indices:
            return 0.0
            
        improvements = []
        
        # Create environment for evaluation
        env = FragmentMatchingEnv(
            self.fragments_top,
            self.fragments_bottom,
            self.dis_map,
            alpha=self.alpha,
            use_game_theory=self.use_game_theory
        )
        
        for idx in self.validation_indices:
            # Manually set current fragment
            env.current_top_idx = idx
            
            # Get state for this fragment
            state = env._get_state(idx, idx)
            
            # Get action from current policy (no exploration)
            action = self.agent.select_action(state, evaluate=True)
            
            # Execute action and record improvement
            _, _, _, info = env.step(action)
            
            if 'rank_improvement' in info:
                improvements.append(info['rank_improvement'])
        
        avg_improvement = np.mean(improvements) if improvements else 0.0
        return avg_improvement
    
    def _save_checkpoint(self, step):
        """
        Save model checkpoint.
        
        Args:
            step: Current training step or identifier
        """
        torch.save(self.agent.actor.state_dict(), f'{self.save_dir}/sac_actor_{step}.pth')
        torch.save(self.agent.critic.state_dict(), f'{self.save_dir}/sac_critic_{step}.pth')
    
    def _save_metrics(self):
        """
        Save training metrics to files.
        """
        # Save train-validation splits
        split_df = pd.DataFrame({
            'train_indices': self.train_indices + [None] * (len(self.validation_indices) - len(self.train_indices)),
            'validation_indices': self.validation_indices + [None] * (len(self.train_indices) - len(self.validation_indices))
        })
        split_df.to_csv(f'{self.save_dir}/train_validation_split.csv', index=False)
        
        # Save performance metrics
        performance_df = pd.DataFrame({
            'step': list(range(0, self.total_timesteps + 1, 1000))[:len(self.training_metrics['train_improvements'])],
            'train_improvements': self.training_metrics['train_improvements'],
            'validation_improvements': self.training_metrics['validation_improvements']
        })
        performance_df.to_csv(f'{self.save_dir}/performance_metrics.csv', index=False)
        
        # Save reward history (summarized to reduce file size)
        reward_df = pd.DataFrame({
            'episode': list(range(1, len(self.training_metrics['rewards']) + 1)),
            'reward': self.training_metrics['rewards'],
            'avg_reward': self.training_metrics['avg_rewards']
        })
        reward_df.to_csv(f'{self.save_dir}/reward_history.csv', index=False)
    
    def visualize_results(self):
        """
        Visualize training process and results.
        """
        if not self.training_metrics['rewards']:
            print("No training data to visualize")
            return
            
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot rewards
        axs[0, 0].plot(self.training_metrics['rewards'], 'b-', alpha=0.3)
        axs[0, 0].plot(self.training_metrics['avg_rewards'], 'r-')
        axs[0, 0].set_title(f'Training Rewards (Seed: {self.random_seed})')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Reward')
        axs[0, 0].grid(True)
        axs[0, 0].legend(['Raw Rewards', 'Average Rewards'])
        
        # Plot losses
        if self.training_metrics['critic_losses']:
            axs[0, 1].plot(self.training_metrics['critic_losses'][::10], 'g-')  # Sample every 10th for clarity
            axs[0, 1].plot(self.training_metrics['actor_losses'][::10], 'b-')
            axs[0, 1].set_title('Training Losses')
            axs[0, 1].set_xlabel('Update Step (sampled)')
            axs[0, 1].set_ylabel('Loss')
            axs[0, 1].grid(True)
            axs[0, 1].legend(['Critic Loss', 'Actor Loss'])
        
        # Plot train vs validation improvements
        steps = list(range(0, self.total_timesteps + 1, 1000))
        if len(steps) > len(self.training_metrics['train_improvements']):
            steps = steps[:len(self.training_metrics['train_improvements'])]
            
        axs[1, 0].plot(steps, self.training_metrics['train_improvements'], 'b-o')
        axs[1, 0].plot(steps, self.training_metrics['validation_improvements'], 'r-o')
        axs[1, 0].set_title('Rank Improvement: Train vs Validation')
        axs[1, 0].set_xlabel('Training Step')
        axs[1, 0].set_ylabel('Average Rank Improvement')
        axs[1, 0].grid(True)
        axs[1, 0].legend(['Training Set', 'Validation Set'])
        
        # Plot training/validation set sizes
        axs[1, 1].bar(['Training Set', 'Validation Set'], [len(self.train_indices), len(self.validation_indices)])
        axs[1, 1].set_title(f'Data Split (Seed: {self.random_seed})')
        axs[1, 1].set_ylabel('Number of Fragments')
        axs[1, 1].grid(True, axis='y')
        
        for i, v in enumerate([len(self.train_indices), len(self.validation_indices)]):
            axs[1, 1].text(i, v + 0.5, str(v), ha='center')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_visualization.png')
        plt.show()

    def _visualize_distance_matrix_comparison(self, original_dis_map, updated_dis_map):
        """
        Visualize comparison between original and updated distance matrices
        
        Args:
            original_dis_map: Original distance matrix
            updated_dis_map: Updated distance matrix
        """
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot original distance matrix
        im1 = axs[0].imshow(original_dis_map, cmap='viridis')
        axs[0].set_title('Original Distance Matrix')
        axs[0].set_xlabel('Bottom Fragment Index')
        axs[0].set_ylabel('Top Fragment Index')
        fig.colorbar(im1, ax=axs[0])
        
        # Plot updated distance matrix
        im2 = axs[1].imshow(updated_dis_map, cmap='viridis')
        axs[1].set_title('Updated Distance Matrix')
        axs[1].set_xlabel('Bottom Fragment Index')
        axs[1].set_ylabel('Top Fragment Index')
        fig.colorbar(im2, ax=axs[1])
        
        # Plot difference matrix
        diff_map = original_dis_map - updated_dis_map
        im3 = axs[2].imshow(diff_map, cmap='RdBu', vmin=-np.max(np.abs(diff_map)), vmax=np.max(np.abs(diff_map)))
        axs[2].set_title('Difference (Red=Improvement, Blue=Worse)')
        axs[2].set_xlabel('Bottom Fragment Index')
        axs[2].set_ylabel('Top Fragment Index')
        fig.colorbar(im3, ax=axs[2])
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/distance_matrix_comparison.png')
        plt.close()

    def update_distance_matrix(self):
        """
        Update distance matrix using the final trained model
        
        Returns:
            Updated distance matrix
        """
        print("\nUpdating distance matrix...")
        
        # Create environment
        env = FragmentMatchingEnv(
            self.fragments_top,
            self.fragments_bottom,
            self.dis_map.copy(),
            alpha=self.alpha,
            use_game_theory=self.use_game_theory
        )
        
        # Define compute_weighted_score function locally if not accessible
        def compute_weighted_score(match_contributions, weights, scale_factor):
            """
            Compute weighted score based on match contributions
            
            Args:
                match_contributions: List of feature match contributions
                weights: List of weights for each feature
                scale_factor: Scaling factor
                
            Returns:
                Weighted score
            """
            weighted_sum = sum(c * w for c, w in zip(match_contributions, weights))
            return weighted_sum * scale_factor
        
        # Initialize updated distance matrix
        updated_dis_map = np.zeros_like(self.dis_map)
        
        # Process each fragment
        for i in tqdm(range(self.num_fragments), desc="Processing fragments"):
            # Set current fragment
            env.current_top_idx = i
            
            # Get state for this fragment
            state = env._get_state(i, i)
            
            # Get action from current policy (no exploration)
            action = self.agent.select_action(state, evaluate=True)
            
            # Extract weights and scaling factor from action
            raw_weights = action[:-1]  # First 4 are weights
            raw_scale = action[-1]     # Last one is scaling factor
            
            # Convert weights and scaling factor to appropriate ranges (same as in env.step)
            weights = (raw_weights + 1) / 2  # Convert to [0,1]
            sum_weights = np.sum(weights)
            weights = weights / sum_weights if sum_weights > 0 else np.ones(4) / 4
            scale_factor = (raw_scale + 1) * 5  # Convert to [0,10]
            
            # Get current top fragment
            top_fragment = self.fragments_top[i]
            
            # Calculate original ranking
            original_scores = [(j, self.dis_map[i][j]) for j in range(self.num_fragments)]
            
            # Calculate feature adjustment scores
            adjustment_scores = []
            for j in range(self.num_fragments):
                match_contributions = top_fragment.feature_match_contribution(self.fragments_bottom[j])
                feature_score = compute_weighted_score(match_contributions, weights, scale_factor)
                adjustment_scores.append((j, feature_score))
            
            # Apply game theory adjustment if enabled
            if self.use_game_theory:
                adjustment_scores = env.game_theory_adjuster.adjust(
                    original_scores,
                    adjustment_scores,
                    i
                )
            
            # Combine scores to get final ranking (same as in env.step)
            final_scores = []
            for j in range(self.num_fragments):
                original_score = next(score for idx, score in original_scores if idx == j)
                original_similarity = 1.0 - original_score / np.max(self.dis_map)
                
                adjustment_score = next(score for idx, score in adjustment_scores if idx == j)
                
                final_score = self.alpha * adjustment_score + (1 - self.alpha) * original_similarity
                final_scores.append((j, final_score))
            
            # Convert final scores back to distances
            # Since final_score is similarity (higher = more similar)
            # We need to convert back to distance (lower = more similar)
            for j, score in final_scores:
                # Convert to distance: 1 - normalized_score
                # We want to preserve the overall scale of the original distance matrix
                updated_dis_map[i, j] = 1.0 - score
            
        # Normalize the updated distance matrix to match the scale of the original
        max_original = np.max(self.dis_map)
        max_updated = np.max(updated_dis_map)
        if max_updated > 0:
            updated_dis_map = updated_dis_map * (max_original / max_updated)
        
        # Save updated distance matrix
        np.save(f'{self.save_dir}/updated_distance_matrix.npy', updated_dis_map)
        print(f"Updated distance matrix saved to {self.save_dir}/updated_distance_matrix.npy")
        
        # Calculate improvement metrics
        diagonal_improvement = np.sum(updated_dis_map[range(self.num_fragments), range(self.num_fragments)] < 
                                self.dis_map[range(self.num_fragments), range(self.num_fragments)])
        
        print(f"Diagonal improvement (true matches): {diagonal_improvement}/{self.num_fragments} ({diagonal_improvement/self.num_fragments*100:.2f}%)")
        
        # Visualize comparison
        self._visualize_distance_matrix_comparison(self.dis_map, updated_dis_map)
        
        return updated_dis_map



def main():
    """Main function to run train-validation split training"""
    start_time = time.time()
    
    # Set fixed random seed for reproducibility
    #RANDOM_SEED = 3813
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    
    print(f"Using fixed random seed: {RANDOM_SEED}")
    
    # Load or generate fragment data
    print("Loading fragment data...")
    try:
        fragments_bottom, fragments_top = load_fragments_from_csv('resource/bamboo_features.csv', 118)
        print(f"Loaded {len(fragments_top)} fragments")
    except:
        print("Unable to load fragment data, generating synthetic fragments")
        fragments_top = generate_synthetic_fragments(118)
        fragments_bottom = generate_synthetic_fragments(118)
    
    # Load or generate distance matrix
    print("Loading distance matrix...")
    try:
        dis_map = np.load('resource/distance_matrix.npy')
        print(f"Loaded distance matrix with shape {dis_map.shape}")
    except:
        print("Unable to load distance matrix, generating random one")
        num_fragments = len(fragments_top)
        dis_map = np.random.rand(num_fragments, num_fragments)
        # Make diagonal elements closer to 0 (more similar)
        for i in range(num_fragments):
            dis_map[i, i] *= 0.1
    
    # Create and run trainer
    trainer = TrainValidSplitTrainer(
        fragments_top=fragments_top,
        fragments_bottom=fragments_bottom,
        dis_map=dis_map,
        train_size=100,  # Use 100 fragments for training
        total_timesteps=10000,
        batch_size=128,
        alpha=0.5,
        use_game_theory=True,
        save_dir='resource/train_valid_split',
        random_seed=RANDOM_SEED  # Pass the fixed random seed
    )
    
    # Run training
    print("\nStarting training with train-validation split...")
    training_metrics = trainer.train()
    
    # Visualize results
    trainer.visualize_results()
    
    # Calculate training time
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds")

    # 在训练结束后添加以下代码
    print("\n验证更新后的距离矩阵...")
    try:
        updated_dis_map = np.load(f'{trainer.save_dir}/updated_distance_matrix.npy')
        print(f"成功加载更新后的距离矩阵，形状为 {updated_dis_map.shape}")
        
        # 可选：计算一些基本统计信息
        print(f"更新后距离矩阵统计：")
        print(f"  最小值: {np.min(updated_dis_map):.4f}")
        print(f"  最大值: {np.max(updated_dis_map):.4f}")
        print(f"  平均值: {np.mean(updated_dis_map):.4f}")
        print(f"  标准差: {np.std(updated_dis_map):.4f}")
    except Exception as e:
        print(f"加载更新后的距离矩阵时出错: {e}")
    
    return trainer, training_metrics


if __name__ == "__main__":
    main()