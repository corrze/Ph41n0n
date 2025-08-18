# tetris_training_plotter.py
# Plot training metrics from Tetris DQN checkpoints and logs


# "Run python tetris_training_plotter.py --save my_tetris_analysis.png " to get the plots

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Optional
import argparse

def load_metrics_from_checkpoint(checkpoint_path: str) -> Optional[Dict]:
    """Load training metrics from a checkpoint file"""
    try:
        # Try safe loading first
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except:
        try:
            # Fallback to unsafe loading
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Failed to load {checkpoint_path}: {e}")
            return None
    
    if 'metrics' in checkpoint:
        return checkpoint['metrics']
    return None

def parse_training_logs(log_file: str) -> Dict[str, List]:
    """Parse training logs from console output"""
    metrics = {
        'episodes': [],
        'avg_rewards': [],
        'avg_lines': [],
        'avg_pieces': [],
        'epsilon': [],
        'buffer_size': [],
        'eval_episodes': [],
        'eval_rewards': [],
        'eval_lines': [],
        'eval_pieces': []
    }
    
    # Regular expressions for parsing
    episode_pattern = r"Episode\s+(\d+)\s*\|\s*Avg Reward:\s*([\d.-]+)\s*\|\s*Avg Lines:\s*([\d.-]+)\s*\|\s*Avg Pieces:\s*([\d.-]+)\s*\|\s*Epsilon:\s*([\d.-]+)\s*\|\s*Buffer:\s*(\d+)"
    eval_pattern = r"EVAL - Reward:\s*([\d.-]+),\s*Lines:\s*([\d.-]+),\s*Pieces:\s*([\d.-]+)"
    
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found")
        return metrics
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    current_episode = None
    
    for line in lines:
        # Parse episode data
        episode_match = re.search(episode_pattern, line)
        if episode_match:
            episode, reward, lines_cleared, pieces, eps, buffer = episode_match.groups()
            metrics['episodes'].append(int(episode))
            metrics['avg_rewards'].append(float(reward))
            metrics['avg_lines'].append(float(lines_cleared))
            metrics['avg_pieces'].append(float(pieces))
            metrics['epsilon'].append(float(eps))
            metrics['buffer_size'].append(int(buffer))
            current_episode = int(episode)
        
        # Parse evaluation data
        eval_match = re.search(eval_pattern, line)
        if eval_match and current_episode:
            eval_reward, eval_lines, eval_pieces = eval_match.groups()
            metrics['eval_episodes'].append(current_episode)
            metrics['eval_rewards'].append(float(eval_reward))
            metrics['eval_lines'].append(float(eval_lines))
            metrics['eval_pieces'].append(float(eval_pieces))
    
    return metrics

def find_all_checkpoints(checkpoint_dir: str = "checkpoints") -> List[Tuple[int, str]]:
    """Find all checkpoint files and return sorted by episode number"""
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("placement_tetris_ep") and filename.endswith(".pth"):
            try:
                episode_str = filename.replace("placement_tetris_ep", "").replace(".pth", "")
                episode_num = int(episode_str)
                checkpoints.append((episode_num, os.path.join(checkpoint_dir, filename)))
            except ValueError:
                continue
    
    return sorted(checkpoints)

def load_metrics_from_checkpoints(checkpoint_dir: str = "checkpoints") -> Dict[str, List]:
    """Load metrics from all available checkpoints"""
    all_metrics = {
        'episodes': [],
        'rewards': [],
        'lines': [],
        'pieces': []
    }
    
    checkpoints = find_all_checkpoints(checkpoint_dir)
    
    for episode_num, checkpoint_path in checkpoints:
        metrics = load_metrics_from_checkpoint(checkpoint_path)
        if metrics and 'episode_rewards' in metrics:
            # Get the data from this checkpoint
            episode_rewards = metrics['episode_rewards']
            episode_lines = metrics.get('episode_lines', [])
            episode_pieces = metrics.get('episode_pieces', [])
            
            # Add to our collection (avoid duplicates)
            start_idx = len(all_metrics['episodes'])
            episodes_to_add = list(range(start_idx + 1, start_idx + len(episode_rewards) + 1))
            
            all_metrics['episodes'].extend(episodes_to_add)
            all_metrics['rewards'].extend(episode_rewards)
            all_metrics['lines'].extend(episode_lines)
            all_metrics['pieces'].extend(episode_pieces)
    
    return all_metrics

def smooth_data(data: List[float], window: int = 100) -> List[float]:
    """Apply moving average smoothing to data"""
    if len(data) < window:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window // 2)
        end_idx = min(len(data), i + window // 2 + 1)
        smoothed.append(np.mean(data[start_idx:end_idx]))
    
    return smoothed

def plot_training_metrics(metrics: Dict[str, List], log_metrics: Dict[str, List] = None, 
                         save_path: str = None, smooth_window: int = 100):
    """Create comprehensive training plots"""
    
    # Set up the plot style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 12))
    
    # Create subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Colors
    colors = {
        'reward': '#2E86AB',
        'lines': '#A23B72', 
        'pieces': '#F18F01',
        'epsilon': '#C73E1D',
        'eval': '#6A994E'
    }
    
    # 1. Reward over time
    ax1 = fig.add_subplot(gs[0, 0])
    if metrics['episodes'] and metrics['rewards']:
        episodes = metrics['episodes']
        rewards = metrics['rewards']
        smoothed_rewards = smooth_data(rewards, smooth_window)
        
        ax1.plot(episodes, rewards, alpha=0.3, color=colors['reward'], linewidth=0.5, label='Raw')
        ax1.plot(episodes, smoothed_rewards, color=colors['reward'], linewidth=2, label=f'Smoothed ({smooth_window})')
        ax1.set_title('Episode Rewards', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # 2. Lines cleared over time
    ax2 = fig.add_subplot(gs[0, 1])
    if metrics['episodes'] and metrics['lines']:
        lines = metrics['lines']
        smoothed_lines = smooth_data(lines, smooth_window)
        
        ax2.plot(episodes, lines, alpha=0.3, color=colors['lines'], linewidth=0.5, label='Raw')
        ax2.plot(episodes, smoothed_lines, color=colors['lines'], linewidth=2, label=f'Smoothed ({smooth_window})')
        ax2.set_title('Lines Cleared per Episode', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Lines Cleared')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # 3. Pieces placed over time
    ax3 = fig.add_subplot(gs[0, 2])
    if metrics['episodes'] and metrics['pieces']:
        pieces = metrics['pieces']
        smoothed_pieces = smooth_data(pieces, smooth_window)
        
        ax3.plot(episodes, pieces, alpha=0.3, color=colors['pieces'], linewidth=0.5, label='Raw')
        ax3.plot(episodes, smoothed_pieces, color=colors['pieces'], linewidth=2, label=f'Smoothed ({smooth_window})')
        ax3.set_title('Pieces Placed per Episode', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Pieces Placed')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # 4. Training vs Evaluation comparison (if log metrics available)
    ax4 = fig.add_subplot(gs[1, 0])
    if log_metrics and log_metrics['episodes']:
        # Training performance
        train_episodes = log_metrics['episodes']
        train_rewards = log_metrics['avg_rewards']
        ax4.plot(train_episodes, train_rewards, color=colors['reward'], linewidth=2, label='Training Avg')
        
        # Evaluation performance
        if log_metrics['eval_episodes']:
            eval_episodes = log_metrics['eval_episodes']
            eval_rewards = log_metrics['eval_rewards']
            ax4.scatter(eval_episodes, eval_rewards, color=colors['eval'], s=50, label='Evaluation', zorder=5)
        
        ax4.set_title('Training vs Evaluation Rewards', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Reward')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    # 5. Epsilon decay (if available)
    ax5 = fig.add_subplot(gs[1, 1])
    if log_metrics and log_metrics['epsilon']:
        ax5.plot(log_metrics['episodes'], log_metrics['epsilon'], color=colors['epsilon'], linewidth=2)
        ax5.set_title('Epsilon Decay', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Epsilon')
        ax5.grid(True, alpha=0.3)
    
    # 6. Performance distribution
    ax6 = fig.add_subplot(gs[1, 2])
    if metrics['rewards']:
        ax6.hist(metrics['rewards'], bins=50, alpha=0.7, color=colors['reward'], edgecolor='black')
        ax6.axvline(np.mean(metrics['rewards']), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(metrics["rewards"]):.1f}')
        ax6.set_title('Reward Distribution', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Reward')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. Lines cleared distribution
    ax7 = fig.add_subplot(gs[2, 0])
    if metrics['lines']:
        line_counts = {}
        for line_count in metrics['lines']:
            line_counts[line_count] = line_counts.get(line_count, 0) + 1
        
        lines_sorted = sorted(line_counts.keys())
        counts = [line_counts[l] for l in lines_sorted]
        
        ax7.bar(lines_sorted, counts, color=colors['lines'], alpha=0.7, edgecolor='black')
        ax7.set_title('Lines Cleared Distribution', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Lines Cleared')
        ax7.set_ylabel('Frequency')
        ax7.grid(True, alpha=0.3)
    
    # 8. Rolling statistics
    ax8 = fig.add_subplot(gs[2, 1])
    if metrics['episodes'] and len(metrics['rewards']) > 1000:
        # Calculate rolling statistics
        window = 1000
        rolling_mean = []
        rolling_std = []
        episodes_rolling = []
        
        for i in range(window, len(metrics['rewards'])):
            window_data = metrics['rewards'][i-window:i]
            rolling_mean.append(np.mean(window_data))
            rolling_std.append(np.std(window_data))
            episodes_rolling.append(metrics['episodes'][i])
        
        ax8.plot(episodes_rolling, rolling_mean, color=colors['reward'], linewidth=2, label='Rolling Mean')
        ax8.fill_between(episodes_rolling, 
                        np.array(rolling_mean) - np.array(rolling_std),
                        np.array(rolling_mean) + np.array(rolling_std),
                        alpha=0.3, color=colors['reward'], label='±1 Std')
        ax8.set_title(f'Rolling Statistics (window={window})', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Episode')
        ax8.set_ylabel('Reward')
        ax8.grid(True, alpha=0.3)
        ax8.legend()
    
    # 9. Training summary stats
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Calculate summary statistics
    if metrics['rewards']:
        stats_text = f"""
        TRAINING SUMMARY
        
        Total Episodes: {len(metrics['episodes']):,}
        
        REWARDS:
        Mean: {np.mean(metrics['rewards']):.2f}
        Std: {np.std(metrics['rewards']):.2f}
        Max: {np.max(metrics['rewards']):.2f}
        Min: {np.min(metrics['rewards']):.2f}
        
        LINES CLEARED:
        Mean: {np.mean(metrics['lines']):.2f}
        Max: {np.max(metrics['lines'])}
        Total: {np.sum(metrics['lines']):,}
        
        PIECES PLACED:
        Mean: {np.mean(metrics['pieces']):.1f}
        Max: {np.max(metrics['pieces'])}
        
        BEST 100 EPISODES:
        Avg Reward: {np.mean(sorted(metrics['rewards'])[-100:]):.2f}
        Avg Lines: {np.mean(sorted(metrics['lines'])[-100:]):.2f}
        """
        
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Tetris DQN Training Analysis', fontsize=18, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot Tetris training metrics")
    parser.add_argument("--checkpoints", type=str, default="checkpoints", help="Checkpoints directory")
    parser.add_argument("--logs", type=str, help="Training log file (optional)")
    parser.add_argument("--save", type=str, help="Save plot to file")
    parser.add_argument("--smooth", type=int, default=100, help="Smoothing window size")
    
    args = parser.parse_args()
    
    print("Loading metrics from checkpoints...")
    checkpoint_metrics = load_metrics_from_checkpoints(args.checkpoints)
    
    log_metrics = None
    if args.logs:
        print("Loading metrics from log file...")
        log_metrics = parse_training_logs(args.logs)
    
    if not checkpoint_metrics['episodes']:
        print("No training data found in checkpoints!")
        if not log_metrics or not log_metrics['episodes']:
            print("No log data either. Make sure you have checkpoints or log files.")
            return
        checkpoint_metrics = {
            'episodes': list(range(1, len(log_metrics['avg_rewards']) + 1)),
            'rewards': log_metrics['avg_rewards'],
            'lines': log_metrics['avg_lines'],
            'pieces': log_metrics['avg_pieces']
        }
    
    print(f"Found {len(checkpoint_metrics['episodes'])} episodes of training data")
    
    plot_training_metrics(checkpoint_metrics, log_metrics, args.save, args.smooth)

if __name__ == "__main__":
    main()