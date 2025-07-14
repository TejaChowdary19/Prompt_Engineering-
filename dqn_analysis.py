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

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, learning_rate=0.001, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                 memory_size=2000, use_target_network=True):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma    # discount factor
        self.epsilon = epsilon   # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.use_target_network = use_target_network
        self.model = self._build_model()
        
        if use_target_network:
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
        if self.use_target_network:
            self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, explore=True):
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
            
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def boltzmann_exploration(self, state, temperature=1.0):
        """Alternative exploration policy using Boltzmann distribution."""
        act_values = self.model.predict(state, verbose=0)[0]
        # Apply temperature scaling
        scaled_values = act_values / temperature
        # Apply softmax to get probabilities
        exp_values = np.exp(scaled_values - np.max(scaled_values))
        probabilities = exp_values / np.sum(exp_values)
        # Choose action based on probabilities
        return np.random.choice(self.action_size, p=probabilities)
        
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.zeros((batch_size, self.state_size))
        next_states = np.zeros((batch_size, self.state_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state[0]
            next_states[i] = next_state[0]
        
        # Predict Q-values for current states
        target_f = self.model.predict(states, verbose=0)
        
        if self.use_target_network:
            # Use target network for next state predictions
            target_next = self.target_model.predict(next_states, verbose=0)
        else:
            # Use main network for next state predictions
            target_next = self.model.predict(next_states, verbose=0)
        
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

def run_experiment(gamma=0.95, learning_rate=0.001, epsilon=1.0, 
                  epsilon_min=0.01, epsilon_decay=0.995, 
                  episodes=100, batch_size=32, use_target_network=True,
                  exploration_policy='epsilon_greedy', temperature=1.0,
                  experiment_name="baseline"):
    """Run a DQN experiment with specified parameters."""
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(
        state_size=state_size, 
        action_size=action_size,
        gamma=gamma,
        learning_rate=learning_rate,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        use_target_network=use_target_network
    )
    
    # Create directory for experiments
    if not os.path.exists('experiments'):
        os.makedirs('experiments')
    
    # Lists for tracking progress
    scores = []
    epsilons = []
    steps_per_episode = []
    
    # Training loop
    for e in range(episodes):
        # Reset environment
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        score = 0
        done = False
        steps = 0
        
        # Play one episode
        while not done:
            steps += 1
            
            # Get action based on selected exploration policy
            if exploration_policy == 'epsilon_greedy':
                action = agent.act(state)
            elif exploration_policy == 'boltzmann':
                action = agent.boltzmann_exploration(state, temperature)
            else:
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
                # Update target model periodically
                if e % 5 == 0 and agent.use_target_network:
                    agent.update_target_model()
                    
                print(f"Episode: {e+1}/{episodes}, Score: {score}, Steps: {steps}, Epsilon: {agent.epsilon:.2f}")
                break
                
        # Store episode data
        scores.append(score)
        epsilons.append(agent.epsilon)
        steps_per_episode.append(steps)
        
        # Save model periodically
        if (e + 1) % 25 == 0:
            agent.save(f"experiments/{experiment_name}_model_{e+1}.weights.h5")
    
    # Save final model
    agent.save(f"experiments/{experiment_name}_model_final.weights.h5")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 3, 2)
    plt.plot(epsilons)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.subplot(1, 3, 3)
    plt.plot(steps_per_episode)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.savefig(f'experiments/{experiment_name}_training_progress.png')
    
    # Save results to file
    results = {
        'scores': scores,
        'average_score': np.mean(scores),
        'max_score': np.max(scores),
        'final_epsilon': agent.epsilon,
        'steps_per_episode': steps_per_episode,
        'average_steps': np.mean(steps_per_episode),
        'parameters': {
            'gamma': gamma,
            'learning_rate': learning_rate,
            'initial_epsilon': epsilon,
            'epsilon_min': epsilon_min,
            'epsilon_decay': epsilon_decay,
            'batch_size': batch_size,
            'use_target_network': use_target_network,
            'exploration_policy': exploration_policy,
            'temperature': temperature
        }
    }
    
    # Print summary
    print("\nExperiment Summary:")
    print(f"Experiment: {experiment_name}")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Max Score: {np.max(scores)}")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    print(f"Average Steps per Episode: {np.mean(steps_per_episode):.2f}")
    
    return results

if __name__ == "__main__":
    # Run baseline experiment
    baseline_results = run_experiment(
        gamma=0.95, 
        learning_rate=0.001, 
        epsilon=1.0, 
        epsilon_min=0.01, 
        epsilon_decay=0.995,
        episodes=100,
        batch_size=32,
        use_target_network=True,
        exploration_policy='epsilon_greedy',
        experiment_name="baseline"
    )
    
    # Experiment with different gamma value
    gamma_experiment = run_experiment(
        gamma=0.99,  # Changed from 0.95
        learning_rate=0.001,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        episodes=100,
        batch_size=32,
        use_target_network=True,
        exploration_policy='epsilon_greedy',
        experiment_name="gamma_099"
    )
    
    # Experiment with different learning rate
    lr_experiment = run_experiment(
        gamma=0.95,
        learning_rate=0.01,  # Changed from 0.001
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        episodes=100,
        batch_size=32,
        use_target_network=True,
        exploration_policy='epsilon_greedy',
        experiment_name="lr_001"
    )
    
    # Experiment with different exploration policy
    boltzmann_experiment = run_experiment(
        gamma=0.95,
        learning_rate=0.001,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        episodes=100,
        batch_size=32,
        use_target_network=True,
        exploration_policy='boltzmann',  # Changed from 'epsilon_greedy'
        temperature=0.5,
        experiment_name="boltzmann"
    )
    
    # Experiment with different epsilon decay
    decay_experiment = run_experiment(
        gamma=0.95,
        learning_rate=0.001,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99,  # Changed from 0.995 (slower decay)
        episodes=100,
        batch_size=32,
        use_target_network=True,
        exploration_policy='epsilon_greedy',
        experiment_name="slower_decay"
    )
    
    # Print comparison
    print("\nExperiment Comparison:")
    print(f"Baseline Average Score: {baseline_results['average_score']:.2f}")
    print(f"Gamma 0.99 Average Score: {gamma_experiment['average_score']:.2f}")
    print(f"Learning Rate 0.01 Average Score: {lr_experiment['average_score']:.2f}")
    print(f"Boltzmann Exploration Average Score: {boltzmann_experiment['average_score']:.2f}")
    print(f"Slower Decay Average Score: {decay_experiment['average_score']:.2f}")