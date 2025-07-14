import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import matplotlib.pyplot as plt
import time
import os
import random

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# DQN Agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount factor
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        # Neural Network for Deep-Q learning Model
        model = keras.Sequential([
            keras.layers.Dense(24, activation='relu', input_shape=(self.state_size,)),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
        
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state[0]
            next_states[i] = next_state[0]
        
        # Predict Q-values for current states and next states
        target_f = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * np.max(target_next[i])
            target_f[i][action] = target
            
        # Train the model
        self.model.fit(states, target_f, epochs=1, verbose=0, batch_size=batch_size)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        self.model.load_weights(name)
        
    def save(self, name):
        self.model.save_weights(name)

def main():
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    
    # Create directory for models
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Training parameters
    episodes = 100  # Set a small number for quick testing
    batch_size = 32
    
    # Lists for tracking progress
    scores = []
    
    # Training loop
    for e in range(episodes):
        # Reset environment
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        score = 0
        done = False
        
        # Play one episode
        while not done:
            # Get action
            action = agent.act(state)
            
            # Take action and get results
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            # Adjust reward if episode ends early (fail)
            reward = reward if not done or score >= 499 else -10
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += 1
            
            # Train with experience replay
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            if done:
                # Update target model every episode
                if e % 5 == 0:
                    agent.update_target_model()
                    
                print(f"Episode: {e+1}/{episodes}, Score: {score}, Epsilon: {agent.epsilon:.2f}")
                break
                
        # Add score to list
        scores.append(score)
        
        # Save model every 10 episodes
        if (e + 1) % 10 == 0:
            agent.save(f"models/dqn_cartpole_{e+1}.weights.h5")
    
    # Save final model
    agent.save("models/dqn_cartpole_final.weights.h5")
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(scores)
    plt.title('DQN Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.savefig('training_progress.png')
    plt.show()
    
    # Test the trained agent
    test_episodes = 5
    print("\nTesting agent...")
    
    for e in range(test_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        score = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            score += 1
            
            if done:
                print(f"Test Episode: {e+1}/{test_episodes}, Score: {score}")
                break

if __name__ == "__main__":
    main()