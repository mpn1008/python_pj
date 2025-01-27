from datetime import datetime
from time import time
import re
import numpy as np

import gymnasium as gym
import minigrid
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium.core import ObservationWrapper
from gymnasium.spaces import Box, Dict
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DoorObsWrapper(ObservationWrapper):
#original feature extractor from: https://github.com/BolunDai0216/MinigridMiniworldTransfer/tree/main
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = Dict(
            {
                "image": env.observation_space.spaces["image"],
                "door_color": Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32),
            }
        )

        self.color_one_hot_dict = {
            "red": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "green": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            "blue": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "purple": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "yellow": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            "grey": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        }
        
    def extract_color(self, input_string):
        color_pattern = r'\b(red|blue|green|yellow|purple|grey)\b'
        
        match = re.search(color_pattern, input_string, re.IGNORECASE)
        
        if match:
            return match.group(0)  
        else:
            return None  # No color found
    
        # print(env.observation_space.)
    def observation(self, obs):
        
        # str = 'go to the yellow door'
        a = self.extract_color(obs["mission"])

        wrapped_obs = {
            "image": obs["image"],
            "door_color": self.color_one_hot_dict[a],
        }

        return wrapped_obs


class DoorEnvExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict):
        print(f"observation_space={observation_space.spaces}")
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0

        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                n_input_channels = subspace.shape[0]
                # print(f"n_input_channels={subspace.shape}")
                # print(f"observation_space={observation_space.spaces}")
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 16, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, (2, 2)),
                    nn.ReLU(),
                    nn.Flatten(),
                )

                # Compute shape by doing one forward pass
                with th.no_grad():
                    # print(th.as_tensor(subspace.sample()[None]))
                    n_flatten = cnn(
                        th.as_tensor(subspace.sample()[None]).float()
                    ).shape[1]

                linear = nn.Sequential(nn.Linear(n_flatten, 64), nn.ReLU())
                extractors["image"] = nn.Sequential(*(list(cnn) + list(linear)))
                total_concat_size += 64

            elif key == "door_color":
                extractors["door_color"] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                )
                total_concat_size += 32

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


policy_kwargs = dict(features_extractor_class=DoorEnvExtractor)
stamp = datetime.fromtimestamp(time()).strftime("%Y%m%d-%H%M%S")

def train():
    env = gym.make("MiniGrid-GoToDoor-8x8-v0")
    env = DoorObsWrapper(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=8e4,
        save_path=f"./models/ppo/minigrid_door_{stamp}/",
        name_prefix="iter",
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./logs/ppo/minigrid_door_tensorboard/",
    )
    model.learn(
        4e6,
        tb_log_name=f"DOOR_PPO_{stamp}",
        callback=checkpoint_callback,
    )


def test():
    env = gym.make("MiniGrid-GoToDoor-8x8-v0", render_mode="human")

    env = DoorObsWrapper(env)

    ppo = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)


    # add the experiment time stamp
    ppo = ppo.load(f"models/ppo/final/iter_1200000_steps.zip", env=env)

    obs, info = env.reset()
    rewards = 0

    while True:
        obs, _ = env.reset()
        
        terminated = False
        while not terminated:
            env.render()
            action, _ = ppo.predict(obs)
            print(action)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards += reward
            if terminated or truncated:
                print(f"Test reward: {rewards}")
                obs, info = env.reset()
                rewards = 0
                continue

