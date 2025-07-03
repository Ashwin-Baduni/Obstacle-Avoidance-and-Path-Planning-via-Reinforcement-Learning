import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/ash/py311/lib'

import sys
from datetime import datetime
from algorithms.q_learning import train_q_learning, display_q_table
from algorithms.dqn_agent import train_dqn
from algorithms.actor_critic import train_actor_critic
from core.utils import plot_training_scores, compare_algorithms, generate_summary_report, save_scores_to_file

def create_output_directories():
    """Create organized output directory structure"""
    base_output_dir = 'outputs'
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # Create subdirectories for each algorithm and file type
    subdirs = [
        'q_learning/logs',
        'q_learning/visualizations',
        'q_learning/data',
        'dqn/logs',
        'dqn/visualizations', 
        'dqn/data',
        'actor_critic/logs',
        'actor_critic/visualizations',
        'actor_critic/data',
        'comparisons/visualizations',
        'comparisons/reports'
    ]
    
    for subdir in subdirs:
        full_path = os.path.join(base_output_dir, subdir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
    
    return base_output_dir

def create_output_filename(algorithm_name, file_type='logs'):
    """Create timestamped output filename in organized directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = create_output_directories()
    
    if file_type == 'logs':
        return os.path.join(base_dir, algorithm_name, 'logs', f"{algorithm_name}_output_{timestamp}.txt")
    elif file_type == 'visualization':
        return os.path.join(base_dir, algorithm_name, 'visualizations', f"{algorithm_name}_performance_{timestamp}.png")
    elif file_type == 'data':
        return os.path.join(base_dir, algorithm_name, 'data', f"{algorithm_name}_scores_{timestamp}.csv")
    elif file_type == 'comparison':
        return os.path.join(base_dir, 'comparisons', 'visualizations', f"algorithm_comparison_{timestamp}.png")
    elif file_type == 'report':
        return os.path.join(base_dir, 'comparisons', 'reports', f"performance_summary_{timestamp}.txt")

def redirect_output(filename):
    """Redirect stdout to file"""
    return open(filename, 'w')

def ask_for_visualization():
    """Ask user if they want to generate visualizations"""
    while True:
        choice = input("Generate visualizations? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")

def main():
    print("=" * 50)
    print("    REINFORCEMENT LEARNING ALGORITHMS")
    print("=" * 50)
    print("1. Q-Learning")
    print("2. Deep Q-Network (DQN)")
    print("3. Actor-Critic")
    print("4. Run All Algorithms")
    print("5. Exit")
    print("=" * 50)
    
    # Create output directories
    base_dir = create_output_directories()
    print(f"📁 Output directory created: {base_dir}")
    
    # Storage for all algorithm scores
    all_scores = {
        'q_learning': None,
        'dqn': None,
        'actor_critic': None
    }
    
    while True:
        choice = input("\nSelect algorithm to run (1-5): ").strip()
        
        if choice == "1":
            print("\nRunning Q-Learning...")
            output_file = create_output_filename("q_learning", "logs")
            print(f"📄 Output will be saved to: {output_file}")
            
            with redirect_output(output_file) as f:
                original_stdout = sys.stdout
                sys.stdout = f
                agent, env, scores = train_q_learning()
                display_q_table(agent, env)
                sys.stdout = original_stdout
            
            all_scores['q_learning'] = scores
            print("✅ Q-Learning completed!")
            
            if ask_for_visualization():
                print("Generating visualizations...")
                plot_file = create_output_filename("q_learning", "visualization")
                csv_file = create_output_filename("q_learning", "data")
                
                plot_training_scores(scores, "Q-Learning Performance", "q_learning", plot_file)
                save_scores_to_file(scores, "q_learning", csv_file)
                
                print(f"📊 Visualization saved: {plot_file}")
                print(f"📄 Scores saved: {csv_file}")
            
        elif choice == "2":
            print("\nRunning DQN...")
            output_file = create_output_filename("dqn", "logs")
            print(f"📄 Output will be saved to: {output_file}")
            
            with redirect_output(output_file) as f:
                original_stdout = sys.stdout
                sys.stdout = f
                agent, scores = train_dqn()
                sys.stdout = original_stdout
            
            all_scores['dqn'] = scores
            print("✅ DQN completed!")
            
            if ask_for_visualization():
                print("Generating visualizations...")
                plot_file = create_output_filename("dqn", "visualization")
                csv_file = create_output_filename("dqn", "data")
                
                plot_training_scores(scores, "DQN Performance", "dqn", plot_file)
                save_scores_to_file(scores, "dqn", csv_file)
                
                print(f"📊 Visualization saved: {plot_file}")
                print(f"📄 Scores saved: {csv_file}")
            
        elif choice == "3":
            print("\nRunning Actor-Critic...")
            output_file = create_output_filename("actor_critic", "logs")
            print(f"📄 Output will be saved to: {output_file}")
            
            with redirect_output(output_file) as f:
                original_stdout = sys.stdout
                sys.stdout = f
                agent, scores = train_actor_critic()
                sys.stdout = original_stdout
            
            all_scores['actor_critic'] = scores
            print("✅ Actor-Critic completed!")
            
            if ask_for_visualization():
                print("Generating visualizations...")
                plot_file = create_output_filename("actor_critic", "visualization")
                csv_file = create_output_filename("actor_critic", "data")
                
                plot_training_scores(scores, "Actor-Critic Performance", "actor_critic", plot_file)
                save_scores_to_file(scores, "actor_critic", csv_file)
                
                print(f"📊 Visualization saved: {plot_file}")
                print(f"📄 Scores saved: {csv_file}")
            
        elif choice == "4":
            print("\nRunning all algorithms sequentially...")
            
            # Q-Learning
            print("\n1/3: Q-Learning")
            output_file = create_output_filename("q_learning", "logs")
            print(f"📄 Output will be saved to: {output_file}")
            
            with redirect_output(output_file) as f:
                original_stdout = sys.stdout
                sys.stdout = f
                agent, env, scores = train_q_learning()
                display_q_table(agent, env)
                sys.stdout = original_stdout
            
            all_scores['q_learning'] = scores
            
            # DQN
            print("\n2/3: Deep Q-Network")
            output_file = create_output_filename("dqn", "logs")
            print(f"📄 Output will be saved to: {output_file}")
            
            with redirect_output(output_file) as f:
                original_stdout = sys.stdout
                sys.stdout = f
                agent, scores = train_dqn()
                sys.stdout = original_stdout
            
            all_scores['dqn'] = scores
            
            # Actor-Critic
            print("\n3/3: Actor-Critic")
            output_file = create_output_filename("actor_critic", "logs")
            print(f"📄 Output will be saved to: {output_file}")
            
            with redirect_output(output_file) as f:
                original_stdout = sys.stdout
                sys.stdout = f
                agent, scores = train_actor_critic()
                sys.stdout = original_stdout
            
            all_scores['actor_critic'] = scores
            
            print("\n✅ All algorithms completed!")
            
            if ask_for_visualization():
                print("\nGenerating comprehensive analysis...")
                
                # Individual plots and data
                for name, scores in all_scores.items():
                    if scores:
                        plot_file = create_output_filename(name, "visualization")
                        csv_file = create_output_filename(name, "data")
                        
                        plot_training_scores(scores, f"{name.replace('_', '-').title()} Performance", name, plot_file)
                        save_scores_to_file(scores, name, csv_file)
                        
                        print(f"📊 {name.title()} visualization: {plot_file}")
                        print(f"📄 {name.title()} scores: {csv_file}")
                
                # Comparison plot
                comparison_file = create_output_filename("", "comparison")
                compare_algorithms(
                    q_scores=all_scores['q_learning'],
                    dqn_scores=all_scores['dqn'],
                    ac_scores=all_scores['actor_critic'],
                    save_path=comparison_file
                )
                print(f"📊 Algorithm comparison: {comparison_file}")
                
                # Summary report
                report_file = create_output_filename("", "report")
                generate_summary_report(
                    q_scores=all_scores['q_learning'],
                    dqn_scores=all_scores['dqn'],
                    ac_scores=all_scores['actor_critic'],
                    save_path=report_file
                )
                print(f"📄 Summary report: {report_file}")
            
        elif choice == "5":
            print("Exiting...")
            break
            
        else:
            print("Invalid choice! Please select 1-5.")
        
        # Check if user has run multiple algorithms and offer comparison
        completed_algorithms = [name for name, scores in all_scores.items() if scores is not None]
        if len(completed_algorithms) > 1:
            compare_choice = input(f"\nYou have results from {len(completed_algorithms)} algorithms. Generate comparison? (y/n): ").strip().lower()
            if compare_choice in ['y', 'yes']:
                print("Generating algorithm comparison...")
                comparison_file = create_output_filename("", "comparison")
                report_file = create_output_filename("", "report")
                
                compare_algorithms(
                    q_scores=all_scores['q_learning'],
                    dqn_scores=all_scores['dqn'],
                    ac_scores=all_scores['actor_critic'],
                    save_path=comparison_file
                )
                generate_summary_report(
                    q_scores=all_scores['q_learning'],
                    dqn_scores=all_scores['dqn'],
                    ac_scores=all_scores['actor_critic'],
                    save_path=report_file
                )
                print(f"📊 Comparison visualization: {comparison_file}")
                print(f"📄 Performance report: {report_file}")
        
        # Ask if user wants to continue
        continue_choice = input("\nDo you want to run another algorithm? (y/n): ").strip().lower()
        if continue_choice != 'y':
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()
