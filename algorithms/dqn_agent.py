import numpy as np
import random
from collections import deque
from tqdm import tqdm
from datetime import datetime
import sys
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from core.environment import GridWorld

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.model(state_tensor)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            current_q_values = self.model(states)
            next_q_values = self.model(next_states)
            
            target_q_values = current_q_values.numpy()
            
            for i in range(batch_size):
                if dones[i]:
                    target_q_values[i][actions[i]] = rewards[i]
                else:
                    target_q_values[i][actions[i]] = rewards[i] + 0.95 * np.max(next_q_values[i])
            
            target_q_values = tf.convert_to_tensor(target_q_values, dtype=tf.float32)
            loss = tf.keras.losses.mse(target_q_values, current_q_values)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def state_to_input(state, grid_size=3):
    """Convert state to one-hot encoded input"""
    input_vector = np.zeros(grid_size * grid_size)
    input_vector[state] = 1
    return input_vector

def train_dqn():
    print("=" * 50)
    print("DEEP Q-NETWORK (DQN) TRAINING")
    print("=" * 50)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    env = GridWorld()
    agent = DQNAgent(state_size=9, action_size=4)
    
    episodes = 500
    batch_size = 32
    scores = []
    
    # Progress bar for episodes
    pbar = tqdm(range(episodes), desc="Training DQN", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for episode in pbar:
        state = env.reset()
        state_input = state_to_input(state)
        total_reward = 0
        steps = 0
        max_steps = 100
        
        while steps < max_steps:
            action = agent.act(state_input)
            next_state, reward, done, _ = env.step(action)
            next_state_input = state_to_input(next_state)
            
            agent.remember(state_input, action, reward, next_state_input, done)
            
            state_input = next_state_input
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        scores.append(total_reward)
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        
        # Update progress bar
        avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
        pbar.set_postfix({'Score': f'{total_reward:.1f}', 
                         'Avg': f'{avg_score:.1f}',
                         'Epsilon': f'{agent.epsilon:.3f}'})
        
        if episode % 50 == 0:
            print(f"Episode: {episode}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}, Avg Score: {avg_score:.2f}")
    
    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Final average score (last 50 episodes): {np.mean(scores[-50:]):.2f}")
    print("DQN training completed!")
    return agent, scores

if __name__ == "__main__":
    output_file = f"dqn_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    print(f"Output will be saved to: {output_file}")
    
    with open(output_file, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        agent, scores = train_dqn()
        sys.stdout = original_stdout
    
    print("✅ DQN completed!")
