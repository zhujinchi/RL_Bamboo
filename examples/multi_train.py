#!/usr/bin/env python
"""
Multiple runs experiment script for bamboo slip fragment matching.
Runs training multiple times with different seeds and reports average performance.
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import pandas as pd
from tqdm import tqdm
import argparse
from datetime import datetime

from rl_bamboo_rejoin import (
    VectorNet, CompareNet, inference,
    load_fragments_from_csv, generate_synthetic_fragments,
    FragmentMatchingEnv, SACAgent, ReplayBuffer
)

from train_valid_split import TrainValidSplitTrainer


class MultiRunExperiment:
    """
    Runs multiple training experiments with different random seeds and reports statistics.
    """
    def __init__(self, 
                 fragments_top, 
                 fragments_bottom, 
                 dis_map,
                 num_runs=5,
                 train_size=100,
                 total_timesteps=50000,
                 batch_size=128,
                 alpha=0.5,
                 use_game_theory=True,
                 base_save_dir='resource/multi_run_experiment'):
        """
        Initialize multi-run experiment.
        
        Args:
            fragments_top: List of top fragments
            fragments_bottom: List of bottom fragments
            dis_map: Distance matrix
            num_runs: Number of runs to perform
            train_size: Number of fragments in training set
            total_timesteps: Steps per training run
            batch_size: Batch size for SAC updates
            alpha: Balance coefficient
            use_game_theory: Whether to use game theory
            base_save_dir: Base directory for saving results
        """
        self.fragments_top = fragments_top
        self.fragments_bottom = fragments_bottom
        self.dis_map = dis_map
        self.num_runs = num_runs
        self.train_size = train_size
        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.alpha = alpha
        self.use_game_theory = use_game_theory
        
        # Create timestamped directory for this experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_save_dir = f"{base_save_dir}_{timestamp}"
        if not os.path.exists(self.base_save_dir):
            os.makedirs(self.base_save_dir)
        
        # Initialize results storage
        self.all_runs_results = {
            'run_id': [],
            'seed': [],
            'final_train_improvement': [],
            'final_validation_improvement': [],
            'best_validation_improvement': [],
            'training_time': []
        }
        
        # Store checkpoint metrics for each run
        self.checkpoint_metrics = {
            'run_id': [],
            'step': [],
            'train_improvement': [],
            'validation_improvement': []
        }
    
    def run_experiments(self):
        """
        Run multiple training experiments with different seeds.
        
        Returns:
            Dictionary with experiment results
        """
        print(f"Starting {self.num_runs} training runs...")
        
        # Set starting seed
        base_seed = int(time.time()) % 10000
        
        for run in range(1, self.num_runs + 1):
            # Set seed for this run
            seed = base_seed + run
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            print(f"\n\n{'='*50}")
            print(f"Starting Run {run}/{self.num_runs} (Seed: {seed})")
            print(f"{'='*50}\n")
            
            # Create save directory for this run
            run_save_dir = f"{self.base_save_dir}/run_{run}"
            if not os.path.exists(run_save_dir):
                os.makedirs(run_save_dir)
            
            # Create and train model for this run
            start_time = time.time()
            
            trainer = TrainValidSplitTrainer(
                fragments_top=self.fragments_top,
                fragments_bottom=self.fragments_bottom,
                dis_map=self.dis_map,
                train_size=self.train_size,
                total_timesteps=self.total_timesteps,
                batch_size=self.batch_size,
                alpha=self.alpha,
                use_game_theory=self.use_game_theory,
                save_dir=run_save_dir
            )
            
            # Train model
            training_metrics = trainer.train()
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Record results
            self._record_run_results(run, seed, training_metrics, training_time)
            
            # Save training and validation split
            self._save_data_split(run, trainer.train_indices, trainer.validation_indices)
            
            # Record checkpoint metrics
            self._record_checkpoint_metrics(run, training_metrics)
            
            # Generate visualizations for this run
            trainer.visualize_results()
            
            # Save combined metrics so far (after each run)
            self._save_combined_results()
        
        # Generate final analysis and visualizations
        self._analyze_results()
        
        return self.all_runs_results
    
    def _record_run_results(self, run, seed, metrics, training_time):
        """Record results from a single run"""
        # Extract final and best metrics
        if metrics['train_improvements'] and metrics['validation_improvements']:
            final_train = metrics['train_improvements'][-1]
            final_validation = metrics['validation_improvements'][-1]
            best_validation = max(metrics['validation_improvements'])
        else:
            final_train = 0.0
            final_validation = 0.0
            best_validation = 0.0
        
        # Add to results
        self.all_runs_results['run_id'].append(run)
        self.all_runs_results['seed'].append(seed)
        self.all_runs_results['final_train_improvement'].append(final_train)
        self.all_runs_results['final_validation_improvement'].append(final_validation)
        self.all_runs_results['best_validation_improvement'].append(best_validation)
        self.all_runs_results['training_time'].append(training_time)
        
        # Print summary for this run
        print(f"\nRun {run} Results:")
        print(f"Final train improvement: {final_train:.2f}")
        print(f"Final validation improvement: {final_validation:.2f}")
        print(f"Best validation improvement: {best_validation:.2f}")
        print(f"Training time: {training_time:.2f} seconds")
    
    def _record_checkpoint_metrics(self, run, metrics):
        """Record checkpoint metrics from a run"""
        if not metrics['train_improvements'] or not metrics['validation_improvements']:
            return
            
        # Assume checkpoints are every 5000 steps
        steps = list(range(0, self.total_timesteps + 1, 5000))
        if len(steps) > len(metrics['train_improvements']):
            steps = steps[:len(metrics['train_improvements'])]
        
        for i, step in enumerate(steps):
            self.checkpoint_metrics['run_id'].append(run)
            self.checkpoint_metrics['step'].append(step)
            self.checkpoint_metrics['train_improvement'].append(metrics['train_improvements'][i])
            self.checkpoint_metrics['validation_improvement'].append(metrics['validation_improvements'][i])
    
    def _save_data_split(self, run, train_indices, validation_indices):
        """Save train-validation split for a run"""
        split_df = pd.DataFrame({
            'train_indices': train_indices + [None] * max(0, len(validation_indices) - len(train_indices)),
            'validation_indices': validation_indices + [None] * max(0, len(train_indices) - len(validation_indices))
        })
        split_df.to_csv(f"{self.base_save_dir}/run_{run}/data_split.csv", index=False)
    
    def _save_combined_results(self):
        """Save combined results from all runs completed so far"""
        # Save all runs summary
        runs_df = pd.DataFrame(self.all_runs_results)
        runs_df.to_csv(f"{self.base_save_dir}/all_runs_summary.csv", index=False)
        
        # Save checkpoint metrics
        checkpoints_df = pd.DataFrame(self.checkpoint_metrics)
        checkpoints_df.to_csv(f"{self.base_save_dir}/checkpoint_metrics.csv", index=False)
    
    def _analyze_results(self):
        """Analyze results from all runs and create visualizations"""
        # Create summary dataframe
        summary_df = pd.DataFrame(self.all_runs_results)
        
        # Calculate statistics
        final_val_mean = np.mean(summary_df['final_validation_improvement'])
        final_val_std = np.std(summary_df['final_validation_improvement'])
        best_val_mean = np.mean(summary_df['best_validation_improvement'])
        best_val_std = np.std(summary_df['best_validation_improvement'])
        
        # Create summary file
        with open(f"{self.base_save_dir}/experiment_summary.txt", 'w') as f:
            f.write(f"Multi-Run Experiment Summary\n")
            f.write(f"==========================\n\n")
            f.write(f"Number of runs: {self.num_runs}\n")
            f.write(f"Train set size: {self.train_size}\n")
            f.write(f"Validation set size: {len(self.fragments_top) - self.train_size}\n")
            f.write(f"Total timesteps per run: {self.total_timesteps}\n")
            f.write(f"Game theory enabled: {self.use_game_theory}\n\n")
            
            f.write(f"Final Results:\n")
            f.write(f"-------------\n")
            f.write(f"Final validation improvement: {final_val_mean:.2f} ± {final_val_std:.2f}\n")
            f.write(f"Best validation improvement: {best_val_mean:.2f} ± {best_val_std:.2f}\n\n")
            
            f.write(f"Individual Run Results:\n")
            f.write(f"----------------------\n")
            for i in range(len(summary_df)):
                run = summary_df.iloc[i]
                f.write(f"Run {run['run_id']} (Seed {run['seed']}): ")
                f.write(f"Final Val: {run['final_validation_improvement']:.2f}, ")
                f.write(f"Best Val: {run['best_validation_improvement']:.2f}, ")
                f.write(f"Time: {run['training_time']:.2f}s\n")
        
        # Print summary
        print("\n" + "="*50)
        print("Experiment Summary:")
        print(f"Final validation improvement: {final_val_mean:.2f} ± {final_val_std:.2f}")
        print(f"Best validation improvement: {best_val_mean:.2f} ± {best_val_std:.2f}")
        print("="*50)
        
        # Create visualizations
        self._create_visualizations()
    
    def _create_visualizations(self):
        """Create visualizations of multi-run results"""
        # Load data
        summary_df = pd.DataFrame(self.all_runs_results)
        checkpoints_df = pd.DataFrame(self.checkpoint_metrics)
        
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Final validation improvements across runs
        axs[0, 0].bar(summary_df['run_id'], summary_df['final_validation_improvement'])
        axs[0, 0].axhline(y=np.mean(summary_df['final_validation_improvement']), 
                      color='r', linestyle='-', label=f"Mean: {np.mean(summary_df['final_validation_improvement']):.2f}")
        axs[0, 0].set_title('Final Validation Rank Improvement by Run')
        axs[0, 0].set_xlabel('Run ID')
        axs[0, 0].set_ylabel('Rank Improvement')
        axs[0, 0].grid(True, axis='y')
        axs[0, 0].legend()
        
        # Plot 2: Training curves (validation improvement) for all runs
        for run_id in sorted(checkpoints_df['run_id'].unique()):
            run_data = checkpoints_df[checkpoints_df['run_id'] == run_id]
            axs[0, 1].plot(run_data['step'], run_data['validation_improvement'], 
                       marker='o', label=f"Run {run_id}")
        
        # Add average line
        avg_by_step = checkpoints_df.groupby('step')['validation_improvement'].mean().reset_index()
        axs[0, 1].plot(avg_by_step['step'], avg_by_step['validation_improvement'], 
                   'k--', linewidth=2, label='Average')
        
        axs[0, 1].set_title('Validation Improvement Throughout Training')
        axs[0, 1].set_xlabel('Training Step')
        axs[0, 1].set_ylabel('Validation Improvement')
        axs[0, 1].grid(True)
        axs[0, 1].legend()
        
        # Plot 3: Distribution of final validation improvements
        axs[1, 0].hist(summary_df['final_validation_improvement'], bins=min(10, self.num_runs))
        axs[1, 0].axvline(x=np.mean(summary_df['final_validation_improvement']), 
                      color='r', linestyle='-', 
                      label=f"Mean: {np.mean(summary_df['final_validation_improvement']):.2f}")
        axs[1, 0].set_title('Distribution of Final Validation Improvements')
        axs[1, 0].set_xlabel('Validation Improvement')
        axs[1, 0].set_ylabel('Count')
        axs[1, 0].grid(True)
        axs[1, 0].legend()
        
        # Plot 4: Train vs Validation improvement (final)
        axs[1, 1].scatter(summary_df['final_train_improvement'], 
                      summary_df['final_validation_improvement'])
        for i, row in summary_df.iterrows():
            axs[1, 1].annotate(f"Run {int(row['run_id'])}", 
                          (row['final_train_improvement'], row['final_validation_improvement']))
            
        # Add diagonal line
        min_val = min(summary_df['final_train_improvement'].min(), 
                    summary_df['final_validation_improvement'].min())
        max_val = max(summary_df['final_train_improvement'].max(), 
                    summary_df['final_validation_improvement'].max())
        axs[1, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        axs[1, 1].set_title('Train vs Validation Improvement')
        axs[1, 1].set_xlabel('Train Improvement')
        axs[1, 1].set_ylabel('Validation Improvement')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.base_save_dir}/multi_run_analysis.png")
        plt.close()
        
        # Create training progress visualization
        self._create_training_progress_visualization()
    
    def _create_training_progress_visualization(self):
        """Create visualization of training progress across runs"""
        checkpoints_df = pd.DataFrame(self.checkpoint_metrics)
        
        # Calculate statistics by step
        stats_by_step = checkpoints_df.groupby('step').agg({
            'validation_improvement': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        # Rename columns
        stats_by_step.columns = ['step', 'mean', 'std', 'min', 'max']
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot mean with error band
        plt.plot(stats_by_step['step'], stats_by_step['mean'], 'b-', linewidth=2, label='Mean')
        plt.fill_between(stats_by_step['step'], 
                        stats_by_step['mean'] - stats_by_step['std'], 
                        stats_by_step['mean'] + stats_by_step['std'], 
                        color='b', alpha=0.2, label='±1 Std Dev')
        
        # Plot min and max
        plt.plot(stats_by_step['step'], stats_by_step['min'], 'g--', linewidth=1, label='Min')
        plt.plot(stats_by_step['step'], stats_by_step['max'], 'r--', linewidth=1, label='Max')
        
        plt.title('Validation Improvement Progress Across All Runs')
        plt.xlabel('Training Step')
        plt.ylabel('Validation Rank Improvement')
        plt.grid(True)
        plt.legend()
        
        plt.savefig(f"{self.base_save_dir}/training_progress.png")
        plt.close()


def main():
    """Main function to run multiple training experiments"""
    parser = argparse.ArgumentParser(description='Run multiple bamboo slip training experiments')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to perform')
    parser.add_argument('--train_size', type=int, default=100, help='Number of fragments in training set')
    parser.add_argument('--timesteps', type=int, default=50000, help='Training steps per run')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for updates')
    parser.add_argument('--alpha', type=float, default=0.5, help='Balance coefficient')
    parser.add_argument('--no_game_theory', action='store_true', help='Disable game theory adjustment')
    parser.add_argument('--save_dir', type=str, default='resource/multi_run_experiment', 
                        help='Base directory for saving results')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Load fragment data
    print("Loading fragment data...")
    try:
        fragments_bottom, fragments_top = load_fragments_from_csv('resource/bamboo_features.csv', 118)
        print(f"Loaded {len(fragments_top)} fragments")
    except:
        print("Unable to load fragment data, generating synthetic fragments")
        fragments_top = generate_synthetic_fragments(118)
        fragments_bottom = generate_synthetic_fragments(118)
    
    # Load distance matrix
    print("Loading distance matrix...")
    try:
        dis_map = np.load('resource/distance_matrix.npy')
        print(f"Loaded distance matrix with shape {dis_map.shape}")
    except:
        print("Unable to load distance matrix, generating random one")
        num_fragments = len(fragments_top)
        dis_map = np.random.rand(num_fragments, num_fragments)
        # Make diagonal elements closer to 0
        for i in range(num_fragments):
            dis_map[i, i] *= 0.1
    
    # Create and run multi-run experiment
    experiment = MultiRunExperiment(
        fragments_top=fragments_top,
        fragments_bottom=fragments_bottom,
        dis_map=dis_map,
        num_runs=args.num_runs,
        train_size=args.train_size,
        total_timesteps=args.timesteps,
        batch_size=args.batch_size,
        alpha=args.alpha,
        use_game_theory=not args.no_game_theory,
        base_save_dir=args.save_dir
    )
    
    # Run experiments
    results = experiment.run_experiments()
    
    # Calculate total time
    total_time = time.time() - start_time
    print(f"\nTotal experiment time: {total_time:.2f} seconds")
    
    return experiment, results


if __name__ == "__main__":
    main()