import numpy as np
from tqdm import tqdm
from core.environment import GridWorld
from datetime import datetime

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state, action] = new_q

def train_q_learning():
    print("=" * 50)
    print("Q-LEARNING TRAINING")
    print("=" * 50)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    env = GridWorld()
    agent = QLearningAgent(n_states=9, n_actions=4)
    
    episodes = 1000
    episode_rewards = []
    
    # Progress bar for episodes
    pbar = tqdm(range(episodes), desc="Training Q-Learning", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for episode in pbar:
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100
        
        while steps < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        
        # Update progress bar with current reward
        pbar.set_postfix({'Reward': f'{total_reward:.1f}', 
                         'Avg': f'{np.mean(episode_rewards[-100:]):.1f}'})
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Avg Reward: {np.mean(episode_rewards[-100:]):.2f}")
    
    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    print("Q-Learning training completed!")
    return agent, env, episode_rewards

def display_q_table(agent, env):
    print("\n" + "=" * 80)
    print("Q-TABLE ANALYSIS")
    print("=" * 80)
    
    print("\nDetailed Q-table Grid:")
    actions = ['^', '>', 'v', '<']
    
    # Header
    header = " |"
    for i in range(env.size):
        for j in range(env.size):
            if (i, j) == env.obstacle_pos:
                header += " OBSTACLE |"
            elif (i, j) == env.goal_pos:
                header += "   GOAL   |"
            else:
                header += f" ({i},{j}) |"
    print(header)
    print("-" * len(header))
    
    # Q-values for each action
    for action in range(4):
        row = f" {actions[action]} |"
        for i in range(env.size):
            for j in range(env.size):
                state = env.get_state_from_pos((i, j))
                if (i, j) == env.obstacle_pos:
                    row += " OBSTACLE |"
                elif (i, j) == env.goal_pos:
                    row += "   GOAL   |"
                else:
                    q_val = agent.q_table[state, action]
                    row += f" {q_val:8.2f} |"
        print(row)
        print("-" * len(header))
    
    print("\nBest Actions Grid:")
    print("-" * 43)
    for i in range(env.size):
        row = "|"
        for j in range(env.size):
            if (i, j) == env.obstacle_pos:
                row += "  OBSTACLE  |"
            elif (i, j) == env.goal_pos:
                row += "   GOAL   |"
            else:
                state = env.get_state_from_pos((i, j))
                best_action = np.argmax(agent.q_table[state])
                best_q = agent.q_table[state, best_action]
                row += f" {actions[best_action]}: {best_q:6.2f}  |"
        print(row)
        print("-" * 43)

if __name__ == "__main__":
    output_file = f"q_learning_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    print(f"Output will be saved to: {output_file}")
    
    import sys
    with open(output_file, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        agent, env, scores = train_q_learning()
        display_q_table(agent, env)
        sys.stdout = original_stdout
    
    print("✅ Q-Learning completed!")
