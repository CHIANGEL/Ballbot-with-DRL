from training_env import BallbotContinuousBulletEnv
import pandas as pd
import math
import pickle
import gym
import numpy as np
import time 
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.ddpg.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

# initialize the environment
env = BallbotContinuousBulletEnv()

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.3) * np.ones(n_actions))

model = DDPG(LnMlpPolicy, env, param_noise=param_noise, action_noise=action_noise,
             actor_lr=1e-4, critic_lr=1e-4, critic_l2_reg=1e-5,
             tensorboard_log=".\\outputs_version1\\logs\\")

for i in range(0, 200):
    # Count start from 1, not 0
    if i == 15:
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))
        model = DDPG.load(".\\outputs_version1\\models\\DDPG_{}".format(i),
                          env, param_noise=None, action_noise=action_noise, 
                          actor_lr=1e-4, critic_lr=1e-4, critic_l2_reg=1e-5,
                          tensorboard_log=".\\outputs_version1\\logs\\")
    elif i == 30:
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.05) * np.ones(n_actions))
        model = DDPG.load(".\\outputs_version1\\models\\DDPG_{}".format(i),
                          env, param_noise=None, action_noise=action_noise, 
                          actor_lr=1e-4, critic_lr=1e-4, critic_l2_reg=1e-5,
                          tensorboard_log=".\\outputs_version1\\logs\\")
    elif i == 45:
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.03) * np.ones(n_actions))
        model = DDPG.load(".\\outputs_version1\\models\\DDPG_{}".format(i),
                          env, param_noise=None, action_noise=action_noise, 
                          actor_lr=1e-4, critic_lr=1e-4, critic_l2_reg=1e-5,
                          tensorboard_log=".\\outputs_version1\\logs\\")
    elif i == 60:
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.01) * np.ones(n_actions))
        model = DDPG.load(".\\outputs_version1\\models\\DDPG_{}".format(i),
                          env, param_noise=None, action_noise=action_noise, 
                          actor_lr=1e-4, critic_lr=1e-4, critic_l2_reg=1e-5,
                          tensorboard_log=".\\outputs_version1\\logs\\")
    elif i == 75:
        action_noise = None
        model = DDPG.load(".\\outputs_version1\\models\\DDPG_{}".format(i),
                          env, param_noise=None, action_noise=action_noise, 
                          actor_lr=1e-4, critic_lr=1e-4, critic_l2_reg=1e-5,
                          tensorboard_log=".\\outputs_version1\\logs\\")
    model.learn(total_timesteps=10000, tb_log_name="DDPG_{}".format(i + 1), reset_num_timesteps=False)
    model.save(".\\outputs_version1\\models\\DDPG_{}".format(i + 1))
    time.sleep(1)
