Deep Q-Learning with CartPole Environment
Project Overview
The project uses a Deep Q-Network (DQN) agent in the OpenAI Gym's CartPole-v1 environment. It demonstrates the deep neural network-based reinforcement learning principle for approximating the Q-function. The code includes experience replay, target networks, and various exploration methods.
Project Structure
dqn_analysis.py: Main analysis script that runs multiple experiments with different hyperparameters and compares their performance.
dqn_cartpole_fixed.py: Robust version of the DQN CartPole agent with fixed model saving.
experiments/: Directory with saved model and training output.
models/: Directory with saved model weights.

Installation

Create a virtual environment:

bashpython3 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate

Install packages:

bashpip install numpy tensorflow matplotlib gym pygame
Running the Code

Running main experiments and analysis:
bashpython dqn_analysis.py
Running stable CartPole implementation only:
bashpython dqn_cartpole_fixed.py
Experiments
The analysis script attempts various hyperparameters and configurations:

Baseline: Default parameters

Gamma (discount factor): 0.95
Learning rate: 0.001
Epsilon decay: 0.995
Exploration policy: Epsilon-greedy

Alternative Gamma: Higher discount factor

Gamma: 0.99

Alternative Learning Rate: Higher learning rate

Learning rate: 0.01

Alternative Exploration Policy: Boltzmann exploration

Temperature: 0.5

Alternative Epsilon Decay: Gradual decay

Epsilon decay: 0.99

Results
The experiments illustrate that:

Slower epsilon decay (0.99) provides best performance with average score of 147.52
Higher learning rate (0.01) improves performance from baseline to 137.52
Increased gamma (0.99) enhances performance over baseline to 124.34
Boltzmann exploration performs worse than epsilon-greedy with mean score of 90.03
Baseline performs with a mean score of 102.06

Major Components

Neural Network Structure: Two hidden layers, 24 units in each
Experience Replay: Buffer size 2000
Target Network: Renewed every 5 episodes
Reward Adjustment: -10 penalty if unable to provide stronger learning signal
Batch Size: 32 for effective learning

Assignment Information
This project fulfills the requirements for LLM Agents & Deep Q-Learning with Atari Games. It demonstrates:

Deep Q-Learning agent implementation
Hyperparameter analysis and influence
Comparison of exploration methods
Discussion of reinforcement learning concepts

License
This project is covered by the MIT License - see the LICENSE file for details.
Acknowledgements

OpenAI Gym for the CartPole environment
TensorFlow/Keras for neural network implementation
Original DQN paper by Mnih et al. (2015)