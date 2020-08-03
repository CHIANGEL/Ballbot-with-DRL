import gym
import pickle
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from autotest_env import BallbotContinuousBulletEnv
import numpy as nppython 
import math
import OpenSSL.tsafe
import os, time
import shlex, subprocess

model = DDPG.load("./saved_models/DDPG")
env = BallbotContinuousBulletEnv()

target_dir = '.\\auto_test\\DDPG'
angles_dimension_2 = [i / 10.0 for i in range(111, 151)]
angles_dimension_3 = [i for i in range(0, 360, 2)]
min_angle_2 = dict()
max_angle_2 = dict()

def check_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def get_range_for_angle_2_for_base(angle_3):
    minV = min_angle_2[angle_3]
    maxV = max_angle_2[angle_3]
    return [i / 10.0 for i in range(int(minV * 10), int(maxV * 10 + 1))]

def get_range_for_angle_2_for_expand(angle_3):
    minV = max_angle_2[angle_3]
    maxV = 12.0
    return [i / 10.0 for i in range(int(minV * 10 + 1), int(maxV * 10 + 1))]

def get_range_for_angle_2_for_normal(angle_3):
    return angles_dimension_2

def single_test(angle_2, angle_3):
    print('Initial State: {}, {}'.format(angle_2, angle_3))
    init_state_file = open("D:/CoppeliaSim_Edu_V4.1.0_AutoTest/initial_state_DDPG.txt", "w")
    init_state_file.write('{}\n{}\n{}\n'.format(0, angle_2, angle_3))
    init_state_file.close()
    vrep_cmd = 'D:/CoppeliaSim_Edu_V4.1.0_AutoTest/coppeliaSim.exe -vdebug -h scenes/ballbot_for_autotest.ttt'
    args = shlex.split(vrep_cmd)
    env_proccess = subprocess.Popen(args=args, stdout=open('./auto_test_DDPG.log', 'w'), 
                                    stderr=open('./auto_test_DDPG.err', 'w'))
    obs = env.reset(9874)
    tan_theta_z = math.sqrt((math.tan(obs[0]) ** 2) + (math.tan(obs[1]) ** 2))
    theta_z = math.atan(tan_theta_z)
    for i in range(500):
        action, _states = model.predict(obs)
        obs, rewards, dones, info, theta_z = env.step(action)
        if theta_z > 87:
            print('Failed at step {}'.format(i))
            env_proccess.kill()
            env_proccess.wait()
            return False
    print('Success')
    env_proccess.kill()
    env_proccess.wait()
    return True

######################################################
# Get the range of angle_2 of traditional controller #
######################################################

file = open('./auto_test/traditional/traditional_controller.txt', "r")
continue_flag = 0
for line in file.readlines():
    line = line.strip().split()
    angle_3 = int(line[0])
    angle_2 = eval(line[1])
    pass_test = int(line[2])
    if angle_3 not in min_angle_2:
        min_angle_2[angle_3] = angle_2
        max_angle_2[angle_3] = angle_2
        continue_flag = 0
    if continue_flag:
        continue
    if pass_test:
        max_angle_2[angle_3] = angle_2
    else:
        continue_flag = 1

##########################################
# Find the limitation of DDPG controller #
##########################################

check_path(target_dir)
file_path = os.path.join(target_dir, 'DDPG_controller.txt')
file = open(file_path, "a")
start = time.clock()
test_count = 0
for angle_3 in angles_dimension_3:
    angles_dimension_2 = get_range_for_angle_2_for_normal(angle_3)
    print(angles_dimension_2)
    for angle_2 in angles_dimension_2:
        test_count += 1
        pass_test = 1 if single_test(angle_2, angle_3) else 0
        file.write('{}\t{}\t{}\n'.format(angle_3, angle_2, pass_test))
        file.flush()
file.close()
elapsed = (time.clock() - start)
print('Total time: {}s'.format(elapsed))
print('Average time on each single test: {}s'.format(elapsed / test_count))