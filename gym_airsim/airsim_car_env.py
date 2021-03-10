from configparser import ConfigParser
import gym
from gym import spaces
import numpy as np
from numpy.linalg import norm
from .car_agent import CarAgent
from os.path import dirname, abspath, join
import sys
import cv2
import math

sys.path.append('..')

class AirSimCarEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,vae):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # read configuration file
        config = ConfigParser()
        config.read(join(dirname(dirname(abspath(__file__))), 'config.ini'))
        self.z_size = int(config['airsim_settings']['z_size'])
        self.vae = vae
        self.track_width = float(config['airsim_settings']['track_width'])

        #number of past actions to concat
        self.n_command_history = 20
        self.n_commands = 1
        self.command_history = np.zeros((1, self.n_commands * self.n_command_history))

        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            shape=(1, self.z_size + self.n_commands * self.n_command_history),
                                            dtype=np.float32)

        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        self.seed()
        self.car_agent = CarAgent()

    def step(self, action):
        # move the car according to the action
        self.car_agent.move(action)
        # compute reward
        car_state = self.car_agent.getCarState()
        reward = self._compute_reward(car_state)

        # check if the episode is done
        done = self._isDone(car_state, reward)

        info = {}
        # get observation
        observation = self.car_agent.observe()

        # solves chicken-egg problem if VAE is not set yet
        if not hasattr(self, "vae"):
            return np.zeros(self.z_size), reward, done, info

        # add to vae buffer for vae optimization if enabled
        self.vae.buffer_append(observation)
        # get encoded observation
        obs = self.vae.encode(observation)
        # concat last actions and update list of last actions
        if self.n_command_history > 0:
            self.command_history = np.roll(self.command_history, shift=-self.n_commands, axis=-1)
            self.command_history[..., -self.n_commands:] = action
            observation = np.concatenate((obs, self.command_history), axis=-1)
            return observation, reward, done, info

    def reset(self):
        self.car_agent.restart()
        self.command_history = np.zeros((1, self.n_commands * self.n_command_history))
        observation = self.car_agent.observe()
        self.vae.buffer_append(observation)
        obs = self.vae.encode(observation)
        if self.n_command_history > 0:
            observation = np.concatenate((obs, self.command_history), axis=-1)
            return observation  # reward, done, info can't be included

    def render(self, mode='human'):
        return

    def close(self):
        self.car_agent.reset()
        return

    def _compute_reward(self, car_state):
        # get closest waypoints to car and car position
        way_point1, way_point2 = self.car_agent.simGet2ClosestWayPoints()
        collision_info = self.car_agent.simGetCollisionInfo()
        car_pos = car_state.kinematics_estimated.position
        car_point = np.array([car_pos.x_val, car_pos.y_val])

        # perpendicular distance to the line connecting 2 closest way points,
        # this distance is approximate to distance to center
        distance_p1_to_p2p3 = lambda p1, p2, p3: abs(np.cross(p2 - p3, p3 - p1)) / norm(p2 - p3)
        distance_to_center = distance_p1_to_p2p3(car_point, way_point1, way_point2)

        reward = self._compute_reward_distance_to_center(distance_to_center, self.track_width)
        # probably car is out of bounds of track
        if reward < 0:
            return reward
        # if car hit cones only
        if collision_info.has_collided:
            if collision_info.object_name.startswith('spline_cones_best') or collision_info.object_name.startswith('trafficone_big_orange'):
                return -10
        return reward


    def _isDone(self, car_state, reward):
        # if reward is negative means either out of bounds or hit
        # in both cases, episode is done
        if reward < 0:
            return True

        car_pos = car_state.kinematics_estimated.position
        car_point = ([car_pos.x_val, car_pos.y_val])
        destination = self.car_agent.simGetWayPoints()[-1]
        distance = norm(car_point - destination)
        # if car got to the finish line, stop
        # we can delete this to make the car go for another lap
        if distance < 0.25:
            print('lap finished')
            return True
        return False

    def _compute_reward_distance_to_center(self,distance_to_center, track_width):
        # Calculate 3 markers that are increasingly further away from the center line
        marker_1 = 0.1 * track_width
        marker_2 = 0.25 * track_width
        marker_3 = 0.5 * track_width

        # Give higher reward if the car is closer to center line and vice versa
        if distance_to_center <= marker_1:
            reward = 1
        elif distance_to_center <= marker_2:
            reward = 0.5
        elif distance_to_center <= marker_3:
            reward = 0.1
        else:
            reward = -10  # likely close to off track

        return reward
