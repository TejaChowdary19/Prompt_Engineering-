import gymnasium as gym
from gymnasium.envs.registration import registry

# Print all registered environments
for env_id in sorted(registry.keys()):
    if 'ALE' in env_id or 'Breakout' in env_id:
        print(env_id)