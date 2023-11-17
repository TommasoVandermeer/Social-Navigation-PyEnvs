import gymnasium as gym
import numpy as np
import logging
import configparser
from social_nav_sim import SocialNavSim

class SocialNavGym(gym.Env, SocialNavSim):
    def __init__(self):
        """
        Social Navigation Gym:
        Simulates many agents following different policies
        and a controllable robot. This Gym environment
        can be used to train autonomous robot through
        reinforcement learning.

        """
        ## Start the Social-Nav-Simulator
        self.start_simulator()

        ## Initialize attributes
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None

    def start_simulator(self):
        """
        Initializes the simulator with the specified parameters.

        """
        super().__init__([7,5,False,"hsfm_new_moussaid",False,False,False],scenario="circular_crossing")
        self.run_live()

    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        if self.config.get('humans', 'policy') == 'orca':
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.square_width = config.getfloat('sim', 'square_width')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')
        else:
            raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    def reset(self, phase='test', test_case = None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        pass

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        pass       

    # TO BE IMPLEMENTED

    def generate_random_human_position(self):
        pass

    def generate_circle_crossing_human(self):
        pass

    def generate_square_crossing_human(self):
        pass

    def get_human_times(self):
        pass

    def onestep_lookahead(self, action):
        pass

    def render(self):
        pass

SocialNavGym()