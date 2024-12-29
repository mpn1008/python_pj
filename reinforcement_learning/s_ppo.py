from datetime import datetime
# from pdb import set_trace
from time import time
import re
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
    def __init__(self, env):
        """A wrapper that makes image the only observation.
        Args:
            env: The environment to apply the wrapper
        """
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
        
        # Search for the pattern in the input string
        match = re.search(color_pattern, input_string, re.IGNORECASE)
        
        if match:
            return match.group(0)  # Return the matched color
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
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                cnn = nn.Sequential(
                    nn.Conv2d(3, 16, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, (2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, (2, 2)),
                    nn.ReLU(),
                    nn.Flatten(),
                )

                # Compute shape by doing one forward pass
                with th.no_grad():
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

    # Create time stamp of experiment
# stamp = datetime.fromtimestamp(time()).strftime("%Y%m%d-%H%M%S")

# print(stamp)
# env = gym.make("MiniGrid-GoToDoor-8x8-v0")
# env = DoorObsWrapper(env)

# checkpoint_callback = CheckpointCallback(
#     save_freq=5e4,
#     save_path=f"./models/ppo/minigrid_door_{stamp}/",
#     name_prefix="iter",
# )

# model = PPO(
#     "MultiInputPolicy",
#     env,
#     policy_kwargs=policy_kwargs,
#     verbose=1,
#     tensorboard_log="./logs/ppo/minigrid_door_tensorboard/",
# )
# model.learn(
#     3e6,
#     tb_log_name=f"DOOR_PPO_{stamp}",
#     callback=checkpoint_callback,
# )


env = gym.make("MiniGrid-GoToDoor-8x8-v0", render_mode="human")


env = DoorObsWrapper(env)

ppo = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

# add the experiment time stamp
ppo = ppo.load(f"models/ppo/minigrid_door_20241229-171714/iter_2500000_steps.zip", env=env)

obs, info = env.reset()
rewards = 0

# for i in range(2000):
#     action, _state = ppo.predict(obs)
#     print(action)
#     obs, reward, terminated, truncated, info = env.step(action)
#     rewards += reward
#     # time.sleep(0.2) 

#     if terminated or truncated:
#         print(f"Test reward: {rewards}")
#         obs, info = env.reset()
#         rewards = 0
#         continue

# print(f"Test reward: {rewards}")

# env.close()

for ep in range (100):
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

env.close()
