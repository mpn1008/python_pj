import torch
from torch import nn
import gymnasium as gym
import minigrid
import numpy as np
import torch.optim as optim
from minigrid.wrappers import ImgObsWrapper

from torch.distributions import Categorical

# Define the policy network
class Policy(nn.Module):
    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, action_size)  # Output layer for action probabilities
        print(self.fc1.weight)
        print(self.fc2.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation
        x = self.fc2(x)  # Linear output
        return torch.softmax(x, dim=-1)  # Return probability distribution over actions


# x = torch.rand(3,3)
# model = Policy(4,4)


a = gym.make('MiniGrid-Empty-6x6-v0')  # Select the MiniGrid environment
env=ImgObsWrapper(a)
state_shape, n_actions = env.observation_space.shape, env.action_space.n
state_dim = state_shape[0]
# Create the policy network and optimizer
policy = Policy(state_dim, n_actions)
