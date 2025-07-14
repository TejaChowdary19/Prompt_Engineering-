import gym
import numpy as np

print("Testing gym environment...")

try:
    # Try creating the environment
    print("Creating environment...")
    env = gym.make('CartPole-v1')
    print("Environment created successfully.")
    
    # Print environment details
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Try resetting the environment
    print("Resetting environment...")
    state = env.reset()
    print(f"Initial state type: {type(state)}")
    print(f"Initial state value: {state}")
    
    # Try taking a single step
    print("Taking a step...")
    action = 0  # Choose action 0
    next_state, reward, done, info = env.step(action)
    print(f"Next state: {next_state}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    print("Test completed successfully.")
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()