import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import random
from scipy import stats
import pandas as pd
import os

def visualize_training(rewards: List[float], critic_losses: List[float], actor_losses: List[float], save_path: Optional[str] = "resource/save/v_train/training_progress"):
    """
    Visualize training progress with improved aesthetics
    
    Args:
        rewards: List of episode rewards
        critic_losses: List of critic loss values
        actor_losses: List of actor loss values
        save_path: Optional path to save the figure
    """
    # Create directory for saving if it doesn't exist
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with higher resolution and better aspect ratio
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=120)
    
    # Define colors
    main_color = '#4878CF'  # Blue
    smooth_color = '#6ACC65'  # Green
    accent_color = '#D65F5F'  # Red
    
    # Plot reward history
    ax = axs[0, 0]
    ax.plot(rewards, color=main_color, alpha=0.8, linewidth=1.2)
    ax.set_title('Episode Rewards', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot smoothed reward history
    ax = axs[0, 1]
    window_size = 20
    smoothed_rewards = []
    if len(rewards) >= window_size:
        smoothed_rewards = [np.mean(rewards[max(0, i-window_size):i+1]) 
                         for i in range(len(rewards))]
        ax.plot(smoothed_rewards, color=smooth_color, linewidth=2)
        ax.set_title(f'Smoothed Episode Rewards (Window={window_size})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        
        # Add horizontal dashed line at the highest smoothed reward
        max_reward = max(smoothed_rewards)
        ax.axhline(y=max_reward, color='gray', linestyle='--', alpha=0.6)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot critic loss
    ax = axs[1, 0]
    smoothed_critic_losses = []
    if critic_losses:
        ax.plot(critic_losses, color=main_color, alpha=0.7, linewidth=1.0)
        
        # Plot smoothed critic loss
        if len(critic_losses) >= 100:
            idx = np.arange(0, len(critic_losses), 100)
            smoothed_critic_losses = [np.mean(critic_losses[max(0, i-100):i+1]) 
                                for i in idx]
            ax.plot(idx, smoothed_critic_losses, color=accent_color, linewidth=2)
            
        ax.set_title('Critic Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Update Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot actor loss
    ax = axs[1, 1]
    smoothed_actor_losses = []
    if actor_losses:
        ax.plot(actor_losses, color=main_color, alpha=0.7, linewidth=1.0)
        
        # Plot smoothed actor loss
        if len(actor_losses) >= 100:
            idx = np.arange(0, len(actor_losses), 100)
            smoothed_actor_losses = [np.mean(actor_losses[max(0, i-100):i+1]) 
                                for i in idx]
            ax.plot(idx, smoothed_actor_losses, color=accent_color, linewidth=2)
            
        ax.set_title('Actor Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Update Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        # Save figure
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        
        # Save data to separate CSV files for different types of data
        # 1. Save raw rewards data
        rewards_df = pd.DataFrame({
            'episode': list(range(len(rewards))),
            'reward': rewards
        })
        rewards_df.to_csv(f"{save_path}_rewards.csv", index=False)
        
        # 2. Save smoothed rewards if available
        if len(smoothed_rewards) > 0:
            smoothed_df = pd.DataFrame({
                'episode': list(range(len(smoothed_rewards))),
                'smoothed_reward': smoothed_rewards
            })
            smoothed_df.to_csv(f"{save_path}_smoothed_rewards.csv", index=False)
        
        # 3. Save critic losses if available
        if len(critic_losses) > 0:
            critic_df = pd.DataFrame({
                'step': list(range(len(critic_losses))),
                'critic_loss': critic_losses
            })
            critic_df.to_csv(f"{save_path}_critic_losses.csv", index=False)
        
        # 4. Save actor losses if available
        if len(actor_losses) > 0:
            actor_df = pd.DataFrame({
                'step': list(range(len(actor_losses))),
                'actor_loss': actor_losses
            })
            actor_df.to_csv(f"{save_path}_actor_losses.csv", index=False)
        
        # 5. Save smoothed losses if available
        if len(smoothed_critic_losses) > 0 or len(smoothed_actor_losses) > 0:
            smoothed_losses = {}
            
            if len(smoothed_critic_losses) > 0:
                idx = np.arange(0, len(critic_losses), 100)
                smoothed_losses['step'] = idx
                smoothed_losses['smoothed_critic_loss'] = smoothed_critic_losses
            
            if len(smoothed_actor_losses) > 0:
                if 'step' not in smoothed_losses:
                    idx = np.arange(0, len(actor_losses), 100)
                    smoothed_losses['step'] = idx
                smoothed_losses['smoothed_actor_loss'] = smoothed_actor_losses
            
            pd.DataFrame(smoothed_losses).to_csv(f"{save_path}_smoothed_losses.csv", index=False)
    
    plt.show()


def analyze_learned_weights(agent, fragments_top, fragments_bottom, num_samples=50, save_path="resource/save/v_weight/weight_analysis"):
    """
    Analyze weights learned by the SAC agent with enhanced visualization
    
    Args:
        agent: Trained SAC agent
        fragments_top: List of top fragments
        fragments_bottom: List of bottom fragments
        num_samples: Number of samples to analyze
        save_path: Optional path to save the figure and data
        
    Returns:
        Dictionary containing analysis results
    """
    # Create directory for saving if it doesn't exist
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    weight_records = []
    scale_records = []
    
    # Sample states and get corresponding weights
    for _ in range(num_samples):
        top_idx = random.randint(0, len(fragments_top) - 1)
        bottom_idx = random.randint(0, len(fragments_bottom) - 1)
        
        try:
            # Import dynamically to avoid circular imports
            from ..environment.fragment_env import FragmentMatchingEnv
            env = FragmentMatchingEnv(fragments_top, fragments_bottom, [])
            state = env._get_state(top_idx, bottom_idx)
        except ImportError:
            # Create a mock state if environment can't be imported
            print("Warning: Could not import FragmentMatchingEnv, using mock state")
            state = np.zeros(12, dtype=np.float32)  # Sample state dimension
        
        # Get action (weights and scaling factor)
        action = agent.select_action(state, evaluate=True)
        
        # Parse action
        raw_weights = action[:-1]
        weights = (raw_weights + 1) / 2  # Convert to [0,1]
        sum_weights = np.sum(weights)
        weights = weights / sum_weights if sum_weights > 0 else np.ones(4) / 4
        
        scale_factor = (action[-1] + 1) * 5  # Convert to [0,10]
        
        weight_records.append(weights)
        scale_records.append(scale_factor)
    
    # Calculate average values and standard deviations
    avg_weights = np.mean(weight_records, axis=0)
    std_weights = np.std(weight_records, axis=0)
    avg_scale = np.mean(scale_records)
    std_scale = np.std(scale_records)
    
    # Visualize weight distribution with enhanced style
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=120)
    
    # Define colors
    main_color = '#4878CF'
    accent_color = '#6ACC65'
    palette = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7']
    
    # Plot average weights and their standard deviations
    feature_names = ["Text Features", "Damage Features", "Layering Features", "Vertical Pattern"]
    x = np.arange(4)
    ax = axs[0, 0]
    bars = ax.bar(x, avg_weights, yerr=std_weights, capsize=5, color=palette, alpha=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=15, ha='right')
    ax.set_ylim(0, max(1.0, max(avg_weights) + max(std_weights) + 0.1))
    ax.set_title('Average Feature Weights', fontsize=14, fontweight='bold')
    ax.set_ylabel('Weight Value', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot weight box plots
    ax = axs[0, 1]
    box = ax.boxplot([np.array(weight_records)[:, i] for i in range(4)], 
                 labels=feature_names, patch_artist=True, 
                 medianprops={'color': 'black'})
    
    # Color the boxes
    for patch, color in zip(box['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title('Weight Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Weight Value', fontsize=12)
    ax.set_xticklabels(feature_names, rotation=15, ha='right')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot scaling factor distribution
    ax = axs[1, 0]
    n, bins, patches = ax.hist(scale_records, bins=20, color=main_color, alpha=0.7)
    ax.axvline(avg_scale, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {avg_scale:.2f}')
    
    # Add normal distribution curve
    x = np.linspace(min(scale_records), max(scale_records), 100)
    y = stats.norm.pdf(x, avg_scale, std_scale) * len(scale_records) * (bins[1] - bins[0])
    ax.plot(x, y, 'r-', linewidth=2, alpha=0.7)
    
    ax.set_title(f'Scale Factor Distribution (Mean={avg_scale:.2f}, Std={std_scale:.2f})', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Scale Factor', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot weight correlation heatmap
    ax = axs[1, 1]
    weight_matrix = np.array(weight_records)
    corr_matrix = np.corrcoef(weight_matrix.T)
    
    # Use a better colormap for correlation matrix
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation', fontsize=10)
    
    # Add correlation values
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                   ha='center', va='center', color=text_color, fontsize=10)
    
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(feature_names)
    ax.set_title('Weight Correlation', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Calculate importance ranking
    importance_rank = np.argsort(avg_weights)[::-1]
    
    # Print statistical information
    print("Feature Weight Analysis:")
    for i, name in enumerate(feature_names):
        print(f"{name}: {avg_weights[i]:.4f} ± {std_weights[i]:.4f}")
    print(f"Scale Factor: {avg_scale:.4f} ± {std_scale:.4f}")
    
    print("\nFeature Importance Ranking:")
    for i, idx in enumerate(importance_rank):
        print(f"{i+1}. {feature_names[idx]}: {avg_weights[idx]:.4f}")
    
    # Save figure and data if path is provided
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Save figure
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        
        # Save weight records - this part is fine as weight_records and scale_records have the same length
        weight_data = pd.DataFrame(weight_records, columns=feature_names)
        weight_data['scale_factor'] = scale_records
        weight_data.to_csv(f"{save_path}_weights.csv", index=False)
        
        # Save summary statistics - fix potential length mismatch issue
        summary_rows = []
        for i, feature in enumerate(feature_names):
            summary_rows.append({
                'feature': feature,
                'mean': avg_weights[i],
                'std': std_weights[i]
            })
        summary_rows.append({
            'feature': 'Scale Factor',
            'mean': avg_scale,
            'std': std_scale
        })
        pd.DataFrame(summary_rows).to_csv(f"{save_path}_summary.csv", index=False)
        
        # Save correlation matrix - this part is fine as corr_matrix is a square matrix
        corr_df = pd.DataFrame(corr_matrix, columns=feature_names, index=feature_names)
        corr_df.to_csv(f"{save_path}_correlation.csv")
    
    plt.show()
    
    return {
        'avg_weights': avg_weights,
        'std_weights': std_weights,
        'avg_scale': avg_scale,
        'std_scale': std_scale,
        'weight_records': weight_records,
        'scale_records': scale_records,
        'corr_matrix': corr_matrix,
        'importance_rank': [feature_names[idx] for idx in importance_rank]
    }


def compare_results(rewards1: List[float], rewards2: List[float], window_size: int = 100, save_path: Optional[str] = "resource/save/v_compare/compare_results") -> Dict[str, Any]:
    """
    Compare results between two methods with enhanced visualization
    
    Args:
        rewards1: Rewards history with game theory adjustment
        rewards2: Rewards history without game theory adjustment
        window_size: Window size for smoothing
        save_path: Optional path to save the figure and data
        
    Returns:
        Dictionary of comparison metrics
    """
    # Create directory for saving if it doesn't exist
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Ensure both reward lists have data
    if not rewards1 or not rewards2:
        print("Warning: Reward lists are empty, cannot compare")
        return {}
    
    # Calculate smoothed reward curves
    def smooth_rewards(rewards, window=window_size):
        smoothed = [np.mean(rewards[max(0, i-window):i+1]) 
                   for i in range(len(rewards))]
        return smoothed
    
    smoothed_rewards1 = smooth_rewards(rewards1)
    smoothed_rewards2 = smooth_rewards(rewards2)
    
    # Calculate statistics
    avg_reward1 = np.mean(rewards1[-1000:]) if len(rewards1) > 1000 else np.mean(rewards1)
    avg_reward2 = np.mean(rewards2[-1000:]) if len(rewards2) > 1000 else np.mean(rewards2)
    
    std_reward1 = np.std(rewards1[-1000:]) if len(rewards1) > 1000 else np.std(rewards1)
    std_reward2 = np.std(rewards2[-1000:]) if len(rewards2) > 1000 else np.std(rewards2)
    
    max_reward1 = np.max(rewards1)
    max_reward2 = np.max(rewards2)
    
    # Convergence speed comparison
    threshold1 = 0.9 * avg_reward1
    threshold2 = 0.9 * avg_reward2
    
    for i, r in enumerate(smoothed_rewards1):
        if r >= threshold1:
            convergence1 = i
            break
    else:
        convergence1 = len(smoothed_rewards1)
        
    for i, r in enumerate(smoothed_rewards2):
        if r >= threshold2:
            convergence2 = i
            break
    else:
        convergence2 = len(smoothed_rewards2)
    
    # Plot comparison figures with enhanced style
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=120)
    
    # Define colors
    blue_color = '#4878CF'
    green_color = '#6ACC65'
    
    # Plot average reward comparison
    ax = axs[0, 0]
    episodes = range(min(len(smoothed_rewards1), len(smoothed_rewards2)))
    ax.plot(episodes, smoothed_rewards1[:len(episodes)], color=blue_color, linewidth=2, label='With Game Theory')
    ax.plot(episodes, smoothed_rewards2[:len(episodes)], color=green_color, linewidth=2, label='Without Game Theory')
    ax.set_title('Smoothed Episode Rewards', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot reward distribution comparison
    ax = axs[0, 1]
    violin_parts = ax.violinplot([rewards1[-1000:] if len(rewards1) > 1000 else rewards1, 
                                 rewards2[-1000:] if len(rewards2) > 1000 else rewards2],
                               showmeans=True)
    
    # Customize violin colors
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor([blue_color, green_color][i])
        pc.set_alpha(0.7)
    
    violin_parts['cmeans'].set_color('black')
    
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['With Game Theory', 'Without Game Theory'])
    ax.set_title('Reward Distribution (Last 1000 Episodes)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot convergence speed comparison
    ax = axs[1, 0]
    bars = ax.bar(['With Game Theory', 'Without Game Theory'], 
           [convergence1, convergence2],
           color=[blue_color, green_color], alpha=0.8, width=0.6)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    ax.set_title('Episodes to Convergence', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Episodes', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot statistical metrics comparison
    ax = axs[1, 1]
    metrics = ['Average\nReward', 'Std.\nDeviation', 'Max\nReward']
    x = np.arange(len(metrics))
    width = 0.35
    
    # Normalize values for better visualization
    max_vals = [max(avg_reward1, avg_reward2), 
                max(std_reward1, std_reward2), 
                max(max_reward1, max_reward2)]
    norm_values_with_gt = [avg_reward1/max_vals[0], std_reward1/max_vals[1], max_reward1/max_vals[2]]
    norm_values_without_gt = [avg_reward2/max_vals[0], std_reward2/max_vals[1], max_reward2/max_vals[2]]
    
    bars1 = ax.bar(x - width/2, norm_values_with_gt, width, label='With Game Theory', color=blue_color, alpha=0.8)
    bars2 = ax.bar(x + width/2, norm_values_without_gt, width, label='Without Game Theory', color=green_color, alpha=0.8)
    
    # Add value labels
    def add_labels(bars, original_values):
        for i, (bar, val) in enumerate(zip(bars, original_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, rotation=0)
    
    add_labels(bars1, [avg_reward1, std_reward1, max_reward1])
    add_labels(bars2, [avg_reward2, std_reward2, max_reward2])
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title('Performance Metrics Comparison (Normalized)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Comparison: With vs. Without Game Theory', fontsize=16, y=1.02)
    plt.subplots_adjust(top=0.92)
    
    # Perform statistical significance test
    sample1 = rewards1[-1000:] if len(rewards1) > 1000 else rewards1
    sample2 = rewards2[-1000:] if len(rewards2) > 1000 else rewards2
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(sample1, sample2)
    
    # Add confidence interval
    ci_low, ci_high = stats.t.interval(0.95, len(sample1)+len(sample2)-2, 
                                      loc=np.mean(sample1)-np.mean(sample2),
                                      scale=stats.sem(np.concatenate([sample1, sample2])))
    
    # Print statistical summary
    print("\n===== Performance Comparison Summary =====")
    print(f"Average Reward: {avg_reward1:.4f} (Game Theory) vs {avg_reward2:.4f} (No Game Theory), Improvement: {((avg_reward1/avg_reward2)-1)*100:.2f}%")
    #print(f"Convergence Speed: {convergence1} (Game Theory) vs {convergence2} (No Game Theory), Improvement: {((convergence2/convergence1)-1)*100:.2f}% (if positive)")
    print(f"Maximum Reward: {max_reward1:.4f} (Game Theory) vs {max_reward2:.4f} (No Game Theory), Improvement: {((max_reward1/max_reward2)-1)*100:.2f}%")
    print(f"Reward Stability (Std. Dev.): {std_reward1:.4f} (Game Theory) vs {std_reward2:.4f} (No Game Theory), Improvement: {((std_reward2/std_reward1)-1)*100:.2f}%")
    print(f"\nStatistical Significance: t-test p-value = {p_value:.8f}")
    print(f"95% Confidence Interval for mean difference: [{ci_low:.4f}, {ci_high:.4f}]")
    
    if p_value < 0.05:
        print("Conclusion: The difference between methods is statistically significant (p < 0.05)")
    else:
        print("Conclusion: The difference between methods is not statistically significant (p >= 0.05)")
    
    # Save figure and data if path is provided
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Save figure
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        
        # Save raw rewards to separate CSV files
        pd.DataFrame({
            'episode': list(range(len(rewards1))),
            'rewards_with_game_theory': rewards1
        }).to_csv(f"{save_path}_rewards_with_gt.csv", index=False)
        
        pd.DataFrame({
            'episode': list(range(len(rewards2))),
            'rewards_without_game_theory': rewards2
        }).to_csv(f"{save_path}_rewards_without_gt.csv", index=False)
        
        # Save smoothed rewards to separate CSV files
        pd.DataFrame({
            'episode': list(range(len(smoothed_rewards1))),
            'smoothed_rewards_with_game_theory': smoothed_rewards1
        }).to_csv(f"{save_path}_smoothed_rewards_with_gt.csv", index=False)
        
        pd.DataFrame({
            'episode': list(range(len(smoothed_rewards2))),
            'smoothed_rewards_without_game_theory': smoothed_rewards2
        }).to_csv(f"{save_path}_smoothed_rewards_without_gt.csv", index=False)
        
        # Save summary statistics - use list of dictionaries to avoid combining arrays of different lengths
        summary_rows = [
            {'metric': 'Average Reward', 'with_game_theory': avg_reward1, 'without_game_theory': avg_reward2, 
             'improvement_percentage': ((avg_reward1/avg_reward2)-1)*100 if avg_reward2 != 0 else None},
            
            {'metric': 'Standard Deviation', 'with_game_theory': std_reward1, 'without_game_theory': std_reward2, 
             'improvement_percentage': ((std_reward2/std_reward1)-1)*100 if std_reward1 != 0 else None},
            
            {'metric': 'Maximum Reward', 'with_game_theory': max_reward1, 'without_game_theory': max_reward2, 
             'improvement_percentage': ((max_reward1/max_reward2)-1)*100 if max_reward2 != 0 else None},
            
            {'metric': 'Episodes to Convergence', 'with_game_theory': convergence1, 'without_game_theory': convergence2, 
             'improvement_percentage': ((convergence2/convergence1)-1)*100 if convergence1 != 0 else None},
            
            {'metric': 'p-value', 'with_game_theory': p_value, 'without_game_theory': p_value, 
             'improvement_percentage': None},
            
            {'metric': 'CI Low', 'with_game_theory': ci_low, 'without_game_theory': ci_low, 
             'improvement_percentage': None},
            
            {'metric': 'CI High', 'with_game_theory': ci_high, 'without_game_theory': ci_high, 
             'improvement_percentage': None}
        ]
        
        pd.DataFrame(summary_rows).to_csv(f"{save_path}_summary_stats.csv", index=False)
    
    plt.show()
    
    return {
        'avg_reward': (avg_reward1, avg_reward2),
        'std_reward': (std_reward1, std_reward2),
        'max_reward': (max_reward1, max_reward2),
        'convergence': (convergence1, convergence2),
        'p_value': p_value,
        'confidence_interval': (ci_low, ci_high)
    }