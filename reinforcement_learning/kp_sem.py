import gymnasium as gym
import minigrid
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from minigrid.wrappers import FullyObsWrapper

from torch.distributions import Categorical

# # Define the policy network
# class Policy(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(Policy, self).__init__()
#         self.fc1 = nn.Linear(state_size, 128)  # First fully connected layer
#         self.fc2 = nn.Linear(128, action_size)  # Output layer for action probabilities

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))  # ReLU activation
#         x = self.fc2(x)  # Linear output
#         return torch.softmax(x, dim=-1)  # Return probability distribution over actions

# # Function to update the policy based on the collected rewards
# def update_policy(optimizer, log_probs, rewards, gamma=0.99):
#     discounted_rewards = []
    
#     # Calculate the discounted rewards
#     for t in range(len(rewards)):
#         G = sum(rewards[t:] * (gamma ** (t2 - t)) for t2 in range(t, len(rewards)))
#         discounted_rewards.append(G)

#     # Convert to tensor
#     discounted_rewards = torch.tensor(discounted_rewards)

#     # Normalize the rewards
#     discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-6)

#     # Compute the loss
#     losses = []
#     for log_prob, reward in zip(log_probs, discounted_rewards):
#         losses.append(-log_prob * reward)  # Negative log probability times reward
#     loss = torch.stack(losses).sum()

#     # Perform backpropagation and step
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# # Set up the environment and policy
# env = gym.make('MiniGrid-Empty-6x6-v0')  # Select the MiniGrid environment
# env = FullyObsWrapper(env)  # Get fully observable states
# state_size = env.observation_space['image'].shape[0] * env.observation_space['image'].shape[1] * env.observation_space['image'].shape[2] + env.observation_space['direction'].n  # Image size + direction size
# action_size = env.action_space.n  # Number of discrete actions

# # Create the policy network and optimizer
# policy = Policy(state_size, action_size)
# optimizer = optim.Adam(policy.parameters(), lr=0.01)

# # Training loop
# num_episodes = 5000
# for episode in range(1):
#     state, _ = env.reset()
#     print(state['image'])
#     log_probs = []
#     rewards = []
#     done = False

#     while not done:
#         # Select action
#         state = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
#         probs = policy(state)
#         action = torch.multinomial(probs, 1).item()
#         log_prob = torch.log(probs[:, action])

#         # Take step
#         next_state, reward, done, _, _ = env.step(action)
#         score += reward
        
#         rewards.append(reward)
#         log_probs.append(log_prob)
#         state = next_state

# env.close()

#_____________________

class PolicyNetwork(nn.Module):
    def __init__(self, image_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        print(image_dim)
        print(action_dim)
        self.fc1 = nn.Linear(image_dim, 32)
        self.fc2 = nn.Linear(32, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

    def choose_action(self, state):
        # print(type(state))
        # print(state)

        if(isinstance(state,tuple)):
            direction = state[0]['direction']
            image = state[0]['image']
        else:
            direction = state['direction']
            image = state['image']

        # Flattening the image
        # image = image.flatten() / 255.0  # Normalize the image

        # Concatenating direction and image
        input_tensor = torch.FloatTensor(np.concatenate(([direction], image)))
        probs = self.forward(input_tensor)
        distribution = Categorical(probs)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)
    
def train(env, policy, episodes, gamma=0.99):
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    for episode in range(episodes):
        state = env.reset()
        done = False
        log_probs = []
        rewards = []

        while not done:
            action, log_prob = policy.choose_action(state)
            next_state, reward, done, truncated , info = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)

            # print(type(next_state))
            state = next_state

        # Compute returns (discounted rewards)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # Convert to tensor and normalize
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Policy update
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)  # Reinforce objective

        # Perform backpropagation
        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        # if episode  % 100 == 0:
        print(f'Episode {episode}, Total Reward: {sum(rewards)}')


env = gym.make('MiniGrid-Empty-6x6-v0')
state = env.reset()
state_shape, n_actions = env.observation_space, env.action_space.n
state_dim = state_shape['image'].shape[0]
print(state_dim)
env = FullyObsWrapper(env)
state = env.reset()
print(state[0]['image'])
policy = PolicyNetwork(state_dim, n_actions)
train(env, policy, episodes=1000)

# Close the environment
env.close()