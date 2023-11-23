import gymnasium as gym
import numpy as np
import logging
from social_gym.social_nav_sim import SocialNavSim
from social_gym.src.utils import point_to_segment_dist
from social_gym.src.info import *

HEADLESS = True

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
        # self.humans = None
        self.global_time = None
        self.human_times = None
        # Reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # Simulation configuration
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
        # For visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        # Action and observation spaces - these are dummy spaces just for registration
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Discrete(1)

    def start_simulator(self):
        """
        Initializes the simulator with the specified parameters.

        """
        # Parameters: [radius, n_actors, random, motion_model, headless, runge_kutta, insert_robot, randomize_human_attributes, robot_visible]
        super().__init__([7,5,False,"hsfm_new_moussaid",HEADLESS,False,True,False,True],scenario="circular_crossing")
        # self.run_live()

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
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test': self.human_times = [0] * self.human_num
        else: self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
        if not self.robot.policy.multiagent_training:
            self.train_val_sim = 'circle_crossing'
        if self.config.get('humans', 'policy') == 'trajnet':
            raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                              'val': 0, 'test': self.case_capacity['val']}
            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    if self.train_val_sim == 'circle_crossing':
                        # Parameters: [radius, n_actors, random, motion_model, headless, runge_kutta, insert_robot, randomize_human_attributes, robot_visible]
                        self.generate_circular_crossing_setting([self.circle_radius,human_num,True,"sfm_guo",HEADLESS,False,True,self.randomize_attributes,self.robot.visible])
                    elif self.train_val_sim == 'square_crossing':
                        raise NotImplementedError
                else:
                    if self.test_sim == 'circle_crossing':
                        # Parameters: [radius, n_actors, random, motion_model, headless, runge_kutta, insert_robot, randomize_human_attributes, robot_visible]
                        self.generate_circular_crossing_setting([self.circle_radius,self.human_num,True,"sfm_guo",HEADLESS,False,True,self.randomize_attributes,self.robot.visible])
                    elif self.test_sim == 'square_crossing':
                        raise NotImplementedError
                self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
            else:
                assert phase == 'test'
                if self.case_counter[phase] == -1:
                    # For debugging purposes
                    self.human_num = 3
                    humans = {0: {"pos": [0,-6], "yaw": np.pi / 2, "goals": [[0,5],[0,-6]]},
                              1: {"pos": [-5,-5], "yaw": np.pi / 2, "goals": [[-5,5],[-5,-5]]},
                              2: {"pos": [5,-5], "yaw": np.pi / 2, "goals": [[5,5],[5,-5]]}}
                    robot = {"pos": [7.5,7.5], "yaw": 0.0, "radius": 0.25, "goals": [[7.5,7.5]]}
                    data = {"headless": False, "motion_model": "sfm_helbing", "runge_kutta": False, "insert_robot": True,
                            "grid": True, "humans": humans, "walls": [[]], "robot": robot}
                    self.config_data = data
                else:
                    raise NotImplementedError      
        # Set sampling time for the simulation and reset the simulator
        self.robot.set(self.config_data["robot"]["pos"][0], self.config_data["robot"]["pos"][1], self.config_data["robot"]["goals"][0][0], self.config_data["robot"]["goals"][0][1], 0, 0, self.config_data["robot"]["yaw"])
        self.reset_sim(reset_robot=False)
        self.set_time_step(self.time_step)
        # Initialize some variables used later
        self.states = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()
        # Get current observation and info
        if self.robot.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError
        info = Nothing()
        
        return ob, {0: info}

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        # Predict next action for each human
        human_actions = self.motion_model_manager.predict_actions(self.time_step) # The observation is not passed as it is can be accessed without passing it

        # Collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(self.humans):
            difference = human.position - self.robot.position
            if self.robot.kinematics == 'holonomic':
                robot_velocity = np.array([action.vx, action.vy])
                velocity_difference = human.linear_velocity - robot_velocity
            else:
                robot_velocity = np.array([action.v * np.cos(action.r + self.robot.theta), action.v * np.sin(action.r + self.robot.theta)])
                velocity_difference = human.linear_velocity - robot_velocity
            e = difference + velocity_difference * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(difference[0], difference[1], e[0], e[1], 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # Check if reaching the goal
        end_position = self.robot.compute_position(action, self.time_step)
        reaching_goal = np.linalg.norm(end_position - self.robot.get_goal_position()) < self.robot.radius

        # Compute Reward, truncated, terminated, and info
        if self.global_time >= self.time_limit - 1:
            reward = 0
            truncated = True
            terminated = False
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            truncated = False
            terminated = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            truncated = False
            terminated = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step # adjust the reward based on FPS
            truncated = False
            terminated = False
            info = Danger(dmin)
        else:
            reward = 0
            truncated = False
            terminated = False
            info = Nothing()

        if update:
            # Store state, action value and attention weights
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())

            # Update all agents position
            self.robot.step(action, self.time_step)
            for i, human_action in enumerate(human_actions): self.humans[i].step(human_action, self.time_step)
            self.global_time += self.time_step
            # removed saving of human_times if agent reaches goal for the first time

            # Compute observation
            if self.robot.sensor == 'coordinates':
                ob = [human.get_observable_state() for human in self.humans]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            # Compute observation
            if self.robot.sensor == 'coordinates':
                ob = [human.get_observable_state() for human in self.humans]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        return ob, reward, terminated, truncated, {0: info}

    def render(self):
        if not HEADLESS: self.render_sim()

    def onestep_lookahead(self, action):
        return self.step(action, update=False)
