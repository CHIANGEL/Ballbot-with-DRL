import gym
import pickle
import pandas
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from training_env import BallbotContinuousBulletEnv
import numpy as np
import math

env = BallbotContinuousBulletEnv()
model = DDPG.load(".\\saved_models\\DDPG")
obs = env.reset()
for i in range(150):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)