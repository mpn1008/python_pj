import gymnasium as gym
import minigrid
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
import torch.optim as optim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO, A2C, DQN
from minigrid.wrappers import ImgObsWrapper
import os


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128)
)
env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="human")
env = ImgObsWrapper(env)

logdir = 'minigrid/logs'
modeldir = 'minigrid/models/dqn'
modelpath = f'{modeldir}/90000.zip'

if not os.path.exists(logdir):
    os.makedirs(logdir)

if not os.path.exists(modeldir):
    os.makedirs(modeldir)


# steps = 10000

# model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=logdir)

# for i in range (1,100):
#     model.learn(total_timesteps=steps, reset_num_timesteps=False, tb_log_name='DQN')
#     model.save(f"{modeldir}/{steps*i}")

model = DQN.load(modelpath, env)

eps = 20

for ep in range (eps):
    obs, _ = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        print(action)
        obs, reward, done, info, _ = env.step(action)
env.close()
