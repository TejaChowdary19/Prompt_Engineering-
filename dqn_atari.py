import gymnasium as gym
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from collections import deque
import matplotlib.pyplot as plt
import cv2
import time
import os

# Make sure TensorFlow is using GPU if available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Preprocessing functions for Atari frames
def preprocess_frame(frame):
    """Preprocess a single frame: grayscale, resize, normalize."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values
    normalized = resized / 255.0
    
    return normalized

def stack_frames(stacked_frames, frame, is_new_episode):
    """Stack 4 frames together as input to the DQN."""
    # Preprocess frame
    frame = preprocess_frame(frame)
    
    if is_new_episode:
        # Clear the stacked frames
        stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for i in range(4)], maxlen=4)
        
        # Stack the frame 4 times
        for _ in range(4):
            stacked_frames.append(frame)
    else:
        # Append frame to deque, automatically removes the oldest
        stacked_frames.append(frame)
    
    # Stack frames into a numpy array for the neural network input (batch_size, height, width, channels)
    stacked_state = np.stack(stacked_frames, axis=2)
    
    return stacked_state, stacked_frames

# Create Deep Q-Network model
def create_dqn_model(input_shape, num_actions):
    """Create a Deep Q-Network model."""
    model = keras.Sequential([
        keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=input_shape),
        keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
    return model

# DQN Agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Experience replay memory
        self.gamma = 0.95    # Discount factor
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.model = create_dqn_model(state_size, action_size)
        self.target_model = create_dqn_model(state_size, action_size)
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
        
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train the model using experience replay."""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.zeros((batch_size, *self.state_size))
        next_states = np.zeros((batch_size, *self.state_size))
        
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
def train_dqn(env_name, episodes=5000, batch_size=32, render=False):
    """Train the DQN agent."""
    # Create and wrap the environment
    env = gym.make(env_name, render_mode='rgb_array')
    
    # Get environment dimensions
    frame = env.reset()[0]  # Get initial frame
    state_size = (84, 84, 4)  # Processed frame size (84x84 with 4 frames stacked)
    action_size = env.action_space.n  # Number of possible actions
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    
    # Initialize variables
    scores = []  # List to store scores
    avg_scores = []  # List to store average scores
    steps_per_episode = []  # List to store steps taken in each episode
    stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for i in range(4)], maxlen=4)
    
    # Create directory for models
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Training loop
    for e in range(episodes):
        # Reset environment
        frame = env.reset()[0]
        state, stacked_frames = stack_frames(stacked_frames, frame, True)
        
        # Initialize variables for this episode
        score = 0
        done = False
        step = 0
        max_steps = 10000  # Limit episode length
        
        # Episode loop
        while not done and step < max_steps:
            step += 1
            
            # Render the environment
            if render:
                env.render()
                
            # Get action
            action = agent.act(state)
            
            # Take action
            next_frame, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Process new frame
            next_state, stacked_frames = stack_frames(stacked_frames, next_frame, False)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += reward
            
            # Train the agent
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
            if done:
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
    # Create and wrap the environment
    env = gym.make(env_name, render_mode='human' if render else 'rgb_array')
    
    # Initialize variables
    scores = []
    steps_list = []
    stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for i in range(4)], maxlen=4)
    
    # Testing loop
    for e in range(episodes):
        # Reset environment
        frame = env.reset()[0]
        state, stacked_frames = stack_frames(stacked_frames, frame, True)
        
        # Initialize variables for this episode
        score = 0
        done = False
        step = 0
        max_steps = 10000  # Limit episode length
        
        # Episode loop
        while not done and step < max_steps:
            step += 1
            
            # Get action
            action = agent.act(state)
            
            # Take action
            next_frame, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Process new frame
            next_state, stacked_frames = stack_frames(stacked_frames, next_frame, False)
            
            # Update state and score
            state = next_state
            score += reward
            
            if done:
                # Print episode stats
                print(f"Test Episode: {e+1}/{episodes}, Score: {score}, Steps: {step}")
                
        # Store episode data
        scores.append(score)
        steps_list.append(step)
    
    # Print test results
    print(f"Average Test Score: {np.mean(scores)}")
    print(f"Average Test Steps: {np.mean(steps_list)}")
    
    return scores, steps_list

# Main execution code
if __name__ == "__main__":
    # Configuration parameters
    ENV_NAME = "Breakout-v4"  # Atari environment name
    TOTAL_EPISODES = 100  # For quick testing; increase for better results (e.g., 5000)
    TEST_EPISODES = 10    # Number of test episodes
    BATCH_SIZE = 32       # Mini-batch size for training
    RENDER_TRAINING = False  # Set to True to visualize training (slows down training significantly)
    
    # Print parameters
    print(f"Environment: {ENV_NAME}")
    print(f"Training Episodes: {TOTAL_EPISODES}")
    print(f"Test Episodes: {TEST_EPISODES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Render Training: {RENDER_TRAINING}")
    
    # Create a timestamp for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print(f"Run timestamp: {timestamp}")
    
    # Train the agent
    print("\nStarting training...\n")
    scores, steps, trained_agent = train_dqn(
        env_name=ENV_NAME,
        episodes=TOTAL_EPISODES,
        batch_size=BATCH_SIZE,
        render=RENDER_TRAINING
    )
    
    # Save the trained agent
    final_model_path = f"models/dqn_agent_{ENV_NAME.replace('/', '_')}_{timestamp}.h5"
    trained_agent.save(final_model_path)
    print(f"\nTraining completed. Model saved to {final_model_path}")
    
    # Print training statistics
    print(f"Final epsilon value: {trained_agent.epsilon:.4f}")
    print(f"Average score (last 100 episodes): {np.mean(scores[-100:]):.2f}")
    print(f"Average steps per episode (last 100 episodes): {np.mean(steps[-100:]):.2f}")
    
    # Ask user if they want to test the trained agent
    test_agent = input("\nDo you want to test the trained agent? (yes/no): ").lower().strip()
    
    if test_agent == "yes" or test_agent == "y":
        print("\nStarting testing...\n")
        test_scores, test_steps = test_dqn(
            env_name=ENV_NAME,
            agent=trained_agent,
            episodes=TEST_EPISODES,
            render=True  # Render during testing
        )
        
        # Print test statistics
        print("\nTesting completed.")
        print(f"Average test score: {np.mean(test_scores):.2f}")
        print(f"Average test steps: {np.mean(test_steps):.2f}")
    else:
        print("\nSkipping testing phase.")
    
    print("\nProgram completed successfully.")