import numpy as np
from tqdm import tqdm
from datetime import datetime
import sys
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from core.environment import GridWorld

class ActorCriticAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Build actor and critic networks
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
    def build_actor(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='softmax')
        ])
        return model
    
    def build_critic(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        return model
    
    def act(self, state):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs = self.actor(state_tensor)
        action = np.random.choice(self.action_size, p=action_probs.numpy()[0])
        return action
    
    def train(self, state, action, reward, next_state, done):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        next_state_tensor = tf.convert_to_tensor([next_state], dtype=tf.float32)
        
        # Calculate TD error
        with tf.GradientTape() as critic_tape:
            current_value = self.critic(state_tensor)
            next_value = self.critic(next_state_tensor)
            
            if done:
                target_value = reward
            else:
                target_value = reward + 0.95 * next_value[0]
            
            td_error = target_value - current_value[0]
            critic_loss = tf.square(td_error)
        
        # Update critic
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        # Update actor
        with tf.GradientTape() as actor_tape:
            action_probs = self.actor(state_tensor)
            action_prob = action_probs[0][action]
            actor_loss = -tf.math.log(action_prob) * td_error
        
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        return td_error.numpy()

def state_to_input(state, grid_size=3):
    """Convert state to one-hot encoded input"""
    input_vector = np.zeros(grid_size * grid_size)
    input_vector[state] = 1
    return input_vector

def train_actor_critic():
    print("=" * 50)
    print("ACTOR-CRITIC TRAINING")
    print("=" * 50)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    env = GridWorld()
    agent = ActorCriticAgent(state_size=9, action_size=4)
    
    episodes = 500
    scores = []
    
    # Progress bar for episodes
    pbar = tqdm(range(episodes), desc="Training Actor-Critic", 
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
            
            # Train the agent
            td_error = agent.train(state_input, action, reward, next_state_input, done)
            
            state_input = next_state_input
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        scores.append(total_reward)
        
        # Update progress bar
        avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
        pbar.set_postfix({'Score': f'{total_reward:.1f}', 
                         'Avg': f'{avg_score:.1f}'})
        
        if episode % 50 == 0:
            print(f"Episode: {episode}/{episodes}, Score: {total_reward}, Avg Score: {avg_score:.2f}")
    
    print(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Final average score (last 50 episodes): {np.mean(scores[-50:]):.2f}")
    print("Actor-Critic training completed!")
    return agent, scores

if __name__ == "__main__":
    output_file = f"actor_critic_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    print(f"Output will be saved to: {output_file}")
    
    with open(output_file, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        agent, scores = train_actor_critic()
        sys.stdout = original_stdout
    
    print("✅ Actor-Critic completed!")
