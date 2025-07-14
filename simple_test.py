import gym
import numpy as np

print("Testing gym environment...")

try:
    # Create a simple environment
    env = gym.make('CartPole-v1')
    print(f"Environment: {env}")
    
    # Get environment information
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset the environment
    state = env.reset()
    print(f"State type: {type(state)}")
    print(f"State: {state}")
    
    # Take a random action
    action = env.action_space.sample()
    print(f"Taking action: {action}")
    
    # Handle different gym API versions
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 4:
        # Old gym API
        next_state, reward, done, info = result
        print("Using old gym API (4 return values)")
    elif isinstance(result, tuple) and len(result) == 5:
        # New gym API
        next_state, reward, terminated, truncated, info = result
        done = terminated or truncated
        print("Using new gym API (5 return values)")
    else:
        print(f"Unknown result type: {type(result)}, length: {len(result) if isinstance(result, tuple) else 'N/A'}")
        next_state, reward, done, info = None, None, None, None
    
    print(f"Next state: {next_state}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    # Try running a full episode
    print("\nRunning a full episode...")
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 200:
        action = env.action_space.sample()
        result = env.step(action)
        
        if isinstance(result, tuple) and len(result) == 4:
            next_state, reward, done, info = result
        elif isinstance(result, tuple) and len(result) == 5:
            next_state, reward, terminated, truncated, info = result
            done = terminated or truncated
        
        state = next_state
        total_reward += reward
        steps += 1
    
    print(f"Episode finished after {steps} steps with total reward {total_reward}")
    print("Test completed successfully.")
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()