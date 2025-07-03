import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def plot_training_scores(scores, title="Training Scores", algorithm_name="algorithm", save_path=None):
    """Plot training scores over episodes"""
    plt.figure(figsize=(12, 6))
    
    # Plot raw scores
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.6, color='blue', label='Episode Scores')
    plt.title(f'{title} - Raw Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot moving average
    plt.subplot(1, 2, 2)
    if len(scores) >= 10:
        moving_avg = moving_average(scores, window_size=min(50, len(scores)//10))
        episodes_avg = range(len(moving_avg))
        plt.plot(episodes_avg, moving_avg, color='red', linewidth=2, label='Moving Average')
    plt.plot(scores, alpha=0.3, color='blue', label='Episode Scores')
    plt.title(f'{title} - Smoothed Trend')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{algorithm_name}_performance_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return save_path

def moving_average(data, window_size=10):
    """Calculate moving average of data"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def compare_algorithms(q_scores=None, dqn_scores=None, ac_scores=None, save_path=None):
    """Compare performance of different algorithms"""
    plt.figure(figsize=(15, 10))
    
    # Raw scores comparison
    plt.subplot(2, 2, 1)
    if q_scores:
        plt.plot(q_scores, label='Q-Learning', alpha=0.7, color='blue')
    if dqn_scores:
        plt.plot(dqn_scores, label='DQN', alpha=0.7, color='red')
    if ac_scores:
        plt.plot(ac_scores, label='Actor-Critic', alpha=0.7, color='green')
    
    plt.title('Algorithm Performance Comparison - Raw Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Moving averages comparison
    plt.subplot(2, 2, 2)
    if q_scores and len(q_scores) >= 10:
        q_avg = moving_average(q_scores, window_size=min(50, len(q_scores)//10))
        plt.plot(range(len(q_avg)), q_avg, label='Q-Learning (MA)', linewidth=2, color='blue')
    if dqn_scores and len(dqn_scores) >= 10:
        dqn_avg = moving_average(dqn_scores, window_size=min(50, len(dqn_scores)//10))
        plt.plot(range(len(dqn_avg)), dqn_avg, label='DQN (MA)', linewidth=2, color='red')
    if ac_scores and len(ac_scores) >= 10:
        ac_avg = moving_average(ac_scores, window_size=min(50, len(ac_scores)//10))
        plt.plot(range(len(ac_avg)), ac_avg, label='Actor-Critic (MA)', linewidth=2, color='green')
    
    plt.title('Algorithm Performance - Moving Averages')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final performance comparison (last 100 episodes)
    plt.subplot(2, 2, 3)
    algorithms = []
    final_scores = []
    colors = []
    
    if q_scores:
        algorithms.append('Q-Learning')
        final_scores.append(np.mean(q_scores[-100:]))
        colors.append('blue')
    if dqn_scores:
        algorithms.append('DQN')
        final_scores.append(np.mean(dqn_scores[-100:]))
        colors.append('red')
    if ac_scores:
        algorithms.append('Actor-Critic')
        final_scores.append(np.mean(ac_scores[-100:]))
        colors.append('green')
    
    bars = plt.bar(algorithms, final_scores, color=colors, alpha=0.7)
    plt.title('Final Performance (Last 100 Episodes Average)')
    plt.ylabel('Average Score')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars, final_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Learning speed comparison
    plt.subplot(2, 2, 4)
    learning_speeds = []
    
    for scores, name in [(q_scores, 'Q-Learning'), (dqn_scores, 'DQN'), (ac_scores, 'Actor-Critic')]:
        if scores:
            final_avg = np.mean(scores[-100:])
            target = 0.9 * final_avg
            
            if len(scores) >= 50:
                moving_avg = moving_average(scores, window_size=50)
                convergence_episode = None
                for i, avg_score in enumerate(moving_avg):
                    if avg_score >= target:
                        convergence_episode = i + 50
                        break
                
                if convergence_episode:
                    learning_speeds.append(convergence_episode)
                else:
                    learning_speeds.append(len(scores))
            else:
                learning_speeds.append(len(scores))
    
    if learning_speeds:
        bars = plt.bar(algorithms, learning_speeds, color=colors, alpha=0.7)
        plt.title('Learning Speed (Episodes to 90% Performance)')
        plt.ylabel('Episodes')
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, speed in zip(bars, learning_speeds):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{speed}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"algorithm_comparison_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return save_path

def generate_summary_report(q_scores=None, dqn_scores=None, ac_scores=None, save_path=None):
    """Generate a text summary report of algorithm performance"""
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"performance_summary_{timestamp}.txt"
    
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("REINFORCEMENT LEARNING ALGORITHMS PERFORMANCE REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        algorithms_data = [
            ("Q-Learning", q_scores),
            ("DQN", dqn_scores),
            ("Actor-Critic", ac_scores)
        ]
        
        for name, scores in algorithms_data:
            if scores:
                f.write(f"{name.upper()} PERFORMANCE:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Episodes: {len(scores)}\n")
                f.write(f"Final Score: {scores[-1]:.2f}\n")
                f.write(f"Average Score (All): {np.mean(scores):.2f}\n")
                f.write(f"Average Score (Last 100): {np.mean(scores[-100:]):.2f}\n")
                f.write(f"Best Score: {np.max(scores):.2f}\n")
                f.write(f"Worst Score: {np.min(scores):.2f}\n")
                f.write(f"Standard Deviation: {np.std(scores):.2f}\n")
                
                consistency = np.std(scores[-100:]) if len(scores) >= 100 else np.std(scores)
                f.write(f"Consistency (Last 100 StdDev): {consistency:.2f}\n")
                f.write("\n")
        
        # Ranking
        f.write("ALGORITHM RANKING:\n")
        f.write("-" * 20 + "\n")
        
        final_performances = []
        for name, scores in algorithms_data:
            if scores:
                final_performances.append((name, np.mean(scores[-100:])))
        
        final_performances.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, score) in enumerate(final_performances, 1):
            f.write(f"{i}. {name}: {score:.2f}\n")
    
    return save_path

def save_scores_to_file(scores, algorithm_name, save_path=None):
    """Save scores to a CSV file for later analysis"""
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"{algorithm_name}_scores_{timestamp}.csv"
    
    with open(save_path, 'w') as f:
        f.write("Episode,Score\n")
        for i, score in enumerate(scores):
            f.write(f"{i+1},{score}\n")
    
    return save_path
