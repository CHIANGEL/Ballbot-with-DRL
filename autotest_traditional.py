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

env = BallbotContinuousBulletEnv()

target_dir = '.\\auto_test\\traditional'
angles_dimension_2 = [i / 10.0 for i in range(40, 111)]
angles_dimension_3 = [i for i in range(0, 360, 2)]

def check_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def single_test(angle_2, angle_3):
    print('Initial State: {}, {}'.format(angle_2, angle_3))
    init_state_file = open("D:/CoppeliaSim_Edu_V4.1.0_AutoTest/initial_state_traditional.txt", "w")
    init_state_file.write('{}\n{}\n{}\n'.format(0, angle_2, angle_3))
    init_state_file.close()
    vrep_cmd = 'D:/CoppeliaSim_Edu_V4.1.0_AutoTest/coppeliaSim.exe -vdebug -h scenes/ballbot_for_autotest.ttt'
    args = shlex.split(vrep_cmd)
    env_proccess = subprocess.Popen(args=args, stdout=open('./auto_test_traditional.log', 'w'), 
                                    stderr=open('./auto_test_traditional.err', 'w'))
    obs = env.reset(7531)
    for i in range(500):
        obs, rewards, dones, info, theta_z = env.step([0, 0])
        if theta_z > 80:
            print('Failed at step {}'.format(i))
            env_proccess.kill()
            env_proccess.wait()
            return False
    print('Success')
    env_proccess.kill()
    env_proccess.wait()
    return True

#################################################
# Find the limitation of traditional controller #
#################################################
check_path(target_dir)
file_path = os.path.join(target_dir, 'traditional_controller.txt')
file = open(file_path, "a")
start = time.clock()
test_count = 0
for angle_3 in angles_dimension_3:
    for angle_2 in angles_dimension_2:
        test_count += 1
        pass_test = 1 if single_test(angle_2, angle_3) else 0
        file.write('{}\t{}\t{}\n'.format(angle_3, angle_2, pass_test))
        file.flush()
file.close()
elapsed = (time.clock() - start)
print('Total time: {}s'.format(elapsed))
print('Average time on each single test: {}s'.format(elapsed / test_count))