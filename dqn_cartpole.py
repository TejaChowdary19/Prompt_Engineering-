import gym
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from collections import deque
import matplotlib.pyplot as plt
import os
import time

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Print TensorFlow version and GPU availability
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Create Deep Q-Network model
def create_dqn_model(input_shape, num_actions):
    """Create a Deep Q-Network model."""
    model = keras.Sequential([
        keras.layers.Dense(24, activation='relu', input_shape=input_shape),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# DQN Agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay memory
        self.gamma = 0.95    # Discount factor
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = create_dqn_model((state_size,), action_size)
        self.target_model = create_dqn_model((state_size,), action_size)
        self.update_target_model()
        
    def update_target_model(self):
        """Copy weights from model to target_model."""
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """Act using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train the model using experience replay."""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state
            
        # Predict Q-values for current states
        target_f = self.model.predict(states, verbose=0)
        # Predict Q-values for next states
        next_target_f = self.target_model.predict(next_states, verbose=0)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * np.max(next_target_f[i])
            target_f[i][action] = target
            
        # Train the model
        self.model.fit(states, target_f, epochs=1, verbose=0)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        """Load model weights."""
        self.model.load_weights(name)
        
    def save(self, name):
        """Save model weights."""
        self.model.save_weights(name)

# Training function
def train_dqn(env_name, episodes=1000, batch_size=32, render=False):
    """Train the DQN agent."""
    # Create environment with render mode
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    
def train_dqn(env_name, episodes=1000, batch_size=32, render=False):
    """Train the DQN agent."""
    # Create environment
    env = gym.make(env_name)
    
    # Get environment dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    
    # Initialize variables
    scores = []  # List to store scores
    avg_scores = []  # List to store average scores
    steps_per_episode = []  # List to store steps taken in each episode
    
    # Create directory for models
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Training loop
    print(f"Starting training episode {e+1}/{episodes}")
    for e in range(episodes):
        # Reset environment
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        # Initialize variables for this episode
        score = 0
        done = False
        step = 0
        max_steps = 500  # Limit episode length
        
        # Episode loop
        while not done and step < max_steps:
            step += 1
            
            # Render the environment
            if render:
                env.render()
                
            # Get action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            # Modify reward to encourage stability
            reward = reward if not done else -10
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += reward
            
            # Train the agent
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
            if done or step >= max_steps:
                # Update target model periodically
                if e % 10 == 0:
                    agent.update_target_model()
                
                # Print episode stats
                print(f"Episode: {e+1}/{episodes}, Score: {score}, Steps: {step}, Epsilon: {agent.epsilon:.4f}")
                
        # Store episode data
        scores.append(score)
        steps_per_episode.append(step)
        
        # Calculate average score
        avg_score = np.mean(scores[-100:])  # Average of last 100 episodes
        avg_scores.append(avg_score)
        
        # Save model periodically
        if (e + 1) % 100 == 0:
            agent.save(f"models/dqn_agent_{env_name}_{e+1}.h5")
            
            # Plot progress
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(scores)
            plt.plot(avg_scores)
            plt.title('Training Progress')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.legend(['Score', 'Average Score'])
            
            plt.subplot(1, 2, 2)
            plt.plot(steps_per_episode)
            plt.title('Steps per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            
            plt.savefig(f'training_progress_{e+1}.png')
            plt.close()
    
    # Save final model
    agent.save(f"models/dqn_agent_{env_name}_final.h5")
    
    # Plot final progress
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.plot(avg_scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend(['Score', 'Average Score'])
    
    plt.subplot(1, 2, 2)
    plt.plot(steps_per_episode)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.savefig('training_progress_final.png')
    plt.close()
    
    return scores, steps_per_episode, agent

# Testing function
def test_dqn(env_name, agent, episodes=100, render=True):
    """Test the trained DQN agent."""
    # Create environment with render mode
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    
def test_dqn(env_name, agent, episodes=100, render=True):
    """Test the trained DQN agent."""
    # Create environment
    env = gym.make(env_name)
    
    # Get environment dimensions
    state_size = env.observation_space.shape[0]
    
    # Initialize variables
    scores = []
    steps_list = []
    
    # Testing loop
    for e in range(episodes):
        # Reset environment
        state = env.reset()
        sta