# import gymnasium as gym
# import minigrid
# from stable_baselines3 import A2C
# from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# from minigrid.wrappers import ImgObsWrapper
# import os

# # env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
# env = DummyVecEnv([lambda: gym.make("MiniGrid-Empty-8x8-v0")])
# env = VecFrameStack(env, n_stack=4)
# logdir = 'minigrid/logs'
# modeldir = 'minigrid/models/a2c'
# modelpath = f'{modeldir}/700000.zip'

# if not os.path.exists(logdir):
#     os.makedirs(logdir)

# if not os.path.exists(modeldir):
#     os.makedirs(modeldir)


# # model = A2C("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=logdir)
# model = A2C('CnnPolicy', env, tensorboard_log=logdir)
# steps = 10000

# for i in range (1,100):
#     model.learn(total_timesteps=steps, reset_num_timesteps=False, tb_log_name='A2C')
#     model.save(f"{modeldir}/{steps*i}")

# # model = A2C.load(modelpath, env)

# # eps = 20

# # for ep in range (eps):
# #     obs, _ = env.reset()
# #     done = False
# #     while not done:
# #         env.render()
# #         action, _ = model.predict(obs)
# #         print(action)
# #         obs, reward, done, info, _ = env.step(action)
# # env.close()

# # Create the Minigrid environment
# # env_id = "MiniGrid-Empty-5x5-v0"  # You can change the environment ID as needed
# # env = gym.make(env_id)

# # Wrap the environment with VecFrameStack for CNN training
# # You can stack frames if you're using CNN to leverage temporal information
# # env = VecFrameStack(env, n_stack=4)

# # # Create the A2C model
# # model = A2C('CnnPolicy', env, verbose=1)

# # # Train the model
# # model.learn(total_timesteps=10000)  # You can adjust the number of timesteps

# # # Save the model
# # model.save("a2c_minigrid")

# # To load the model later, use:
# # model = A2C.load("a2c_minigrid")

# # Test the trained model
# # obs = env.reset()
# # for _ in range(1000):
# #     action, _states = model.predict(obs)
# #     obs, rewards, dones, info = env.step(action)
# #     env.render()  # Render the environment

# # env.close()


from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

# i=0
# while True:
#     i+=1
#     model = A2C("MlpPolicy", vec_env, verbose=1, tensorboard_log="minigrid/logs")
#     model.learn(total_timesteps=25000,reset_num_timesteps=False, tb_log_name='a2c_cp')
#     model.save(f"a2c_cartpole-{i}")

# del model # remove to demonstrate saving and loading

model = A2C.load("a2c_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")