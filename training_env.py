import os, inspect
currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import subprocess
import pybullet as p2
import pybullet_data
import pybullet_utils.bullet_client as bc
from pkg_resources import parse_version
from datetime import datetime

try:
    import sim
except:
    print('--------------------------------------------------------------')
    print('"sim.py" could not be imported. This means very probably that')
    print('either "sim.py" or the remoteApi library could not be found.')
    print('Make sure both are in the same folder as this file,')
    print('or appropriately adjust the file "sim.py"')
    print('--------------------------------------------------------------')
    print('')

logger = logging.getLogger(__name__)


class BallbotContinuousBulletEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, renders=False, discrete_actions=True):
        self._renders = renders
        self._discrete_actions = discrete_actions
        self._render_height = 200
        self._render_width = 320
        self._physics_client_id = -1
        self.target_velo_x = 0
        self.target_velo_y = 0
        self.velo_error_x_threshold = self.target_velo_x * 2
        self.velo_error_y_threshold = self.target_velo_y * 2
        self.velo_error_x_integration_threshold = float('inf')
        self.velo_error_y_integration_threshold = float('inf')
        self.theta_x_threshold = 12 / 180 * math.pi * 2
        self.theta_y_threshold = 12 / 180 * math.pi * 2
        self.theta_x_dot_threshold = 12 / 180 * math.pi * 2
        self.tehta_y_dot_threshold = 12 / 180 * math.pi * 2
        self.z_theta_threshold = math.pi
        self.z_threshold = 0.85
        self.state_threshold = np.array([
            self.theta_x_threshold, self.theta_y_threshold,
            self.theta_x_dot_threshold, self.theta_x_dot_threshold,
            self.z_threshold
        ])
        self.velocity_mag = 0.1
        action_dim = 2
        action_high = np.array([self.velocity_mag] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(low=-self.state_threshold, high=self.state_threshold, dtype=np.float32)
        self.seed()
        self.viewer = None
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.step_number = self.step_number + 1
        print('step {}: {}'.format(self.step_number, action))

        res, retInts, reFloats, retStrings, retBuffer = sim.simxCallScriptFunction(
            self.clientID, 'BallRobot', sim.sim_scripttype_childscript,
            'remote_step', [], [action[0], action[1]], [], bytearray(),
            sim.simx_opmode_oneshot_wait)
        while len(reFloats) == 0:
            res, retInts, reFloats, retStrings, retBuffer = sim.simxCallScriptFunction(
                self.clientID, 'BallRobot', sim.sim_scripttype_childscript,
                'remote_step', [], [action[0], action[1]], [], bytearray(),
                sim.simx_opmode_oneshot_wait)

        tan_theta_z = math.sqrt(
            math.tan(reFloats[0]) * math.tan(reFloats[0]) +
            math.tan(reFloats[1]) * math.tan(reFloats[1]))
        theta_z = math.atan(tan_theta_z)
        done = self.step_number > 100 or reFloats[0] > 30 / 180 * math.pi or reFloats[0] < -30 / 180 * math.pi \
              or reFloats[1] > 30 / 180 * math.pi or reFloats[1] < -30 / 180 * math.pi \
              or reFloats[5] > 1.5 or reFloats[5] < -1.5 \
              or reFloats[4] > 1.5 or reFloats[4] < -1.5
        done = bool(done)

        reward = -theta_z + 0.35
        self.state = reFloats[:4] + reFloats[-1:]
        return np.array(self.state), reward, done, {}

    def reset(self):
        print("-----------reset simulation---------------")
        if self._physics_client_id < 0:
            if self._renders:
                self._p = bc.BulletClient(connection_mode=p2.GUI)
            else:
                self._p = bc.BulletClient()
            self._physics_client_id = self._p._client
        self.step_number = 0

        sim.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = sim.simxStart('127.0.0.1', 9513, True, True, 10000, 1)  # Connect to CoppeliaSim
        if self.clientID != -1:
            print('Connected to remote API server')
            returnCode = sim.simxStopSimulation(self.clientID, sim.simx_opmode_blocking)
            msg = sim.simxGetInMessageInfo(self.clientID, sim.simx_headeroffset_server_state)
            while msg[1] != 0:
                time.sleep(0.05)
                msg = sim.simxGetInMessageInfo(self.clientID, sim.simx_headeroffset_server_state)
                sim.simxStopSimulation(self.clientID, sim.simx_opmode_blocking)
            returnCode = sim.simxStartSimulation(self.clientID, sim.simx_opmode_blocking)

            print("Start Simulation!")
            res, retInts, reFloats, retStrings, retBuffer = sim.simxCallScriptFunction(
                self.clientID, 'BallRobot',
                sim.sim_scripttype_childscript, 'remote_reset', [], [], [],
                bytearray(), sim.simx_opmode_blocking)
            while len(reFloats) == 0:
                res, retInts, reFloats, retStrings, retBuffer = sim.simxCallScriptFunction(
                    self.clientID, 'BallRobot',
                    sim.sim_scripttype_childscript, 'remote_reset', [], [], [],
                    bytearray(), sim.simx_opmode_blocking)
            print(reFloats)
            self.state = reFloats[:4] + reFloats[-1:]

        return np.array(self.state)

    def render(self, mode='human', close=False):
        px = np.array([[[255, 255, 255, 255]] * 2] * 2, dtype=np.uint8)
        rgb_array = np.array(px, dtype=np.uint8)
        return rgb_array

    def close(self):
        if self._physics_client_id >= 0:
            self._p.disconnect()
        self._physics_client_id = -1
