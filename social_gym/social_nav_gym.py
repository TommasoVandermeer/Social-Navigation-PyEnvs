import gymnasium as gym
import numpy as np
import logging
from social_gym.social_nav_sim import SocialNavSim
from social_gym.src.info import *
from social_gym.src.utils import is_multiple

HEADLESS = True
PARALLELIZE_ROBOT = True
PARALLELIZE_HUMANS = False # WARNING: Parallelizing humans is not convenient if episodes have less than 10 humans
HUMAN_MODELS = ["sfm_helbing","sfm_guo","sfm_moussaid","hsfm_farina","hsfm_guo",
                 "hsfm_moussaid","hsfm_new","hsfm_new_guo","hsfm_new_moussaid","orca"]

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
        super().__init__(config_data = {"insert_robot":True, "robot_visibile":False, "headless": HEADLESS}, scenario="circular_crossing", parallelize_robot = PARALLELIZE_ROBOT, parallelize_humans = PARALLELIZE_HUMANS)
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
        # For ORCA
        self.safety_space = 0
        # For visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        # Action and observation spaces - these are dummy spaces just for registration
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Discrete(1)

    def configure(self, config):
        self.config = config
        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.robot_time_step = config.getfloat('env', 'robot_time_step')
        if not is_multiple(self.robot_time_step, self.time_step): raise ValueError("Robot time step must be a multiple of time step")
        self.time_step_factor = int(self.robot_time_step / self.time_step)
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.human_policy = config.get('humans', 'policy')
        self.robot_radius = config.getfloat('robot', 'radius')
        if self.human_policy in HUMAN_MODELS:
            self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
            self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                              'test': config.getint('env', 'test_size')}
            self.train_val_sim = config.get('sim', 'train_val_sim')
            self.test_sim = config.get('sim', 'test_sim')
            self.traffic_height = config.getfloat('sim', 'traffic_height')
            self.traffic_length = config.getfloat('sim', 'traffic_length')
            self.circle_radius = config.getfloat('sim', 'circle_radius')
            self.human_num = config.getint('sim', 'human_num')
        else: raise NotImplementedError
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        logging.info('human number: {}'.format(self.human_num))
        logging.info('Human policy: {}'.format(self.human_policy))
        if self.randomize_attributes: logging.info("Randomize human's radius and preferred speed")
        else: logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_sim, self.test_sim))
        logging.info('Traffic length: {}, Traffic height: {}, circle width: {}'.format(self.traffic_length, self.traffic_height, self.circle_radius))

    def set_safety_space(self, safety_space:float):
        self.safety_space = safety_space

    def set_robot(self, robot):
        self.robot = robot

    def compute_humans_observable_state(self):
        if self.robot.sensor == 'coordinates': 
            if self.robot.policy.with_theta_and_omega_visible: ob = [human.get_observable_state(visible_theta_and_omega=True) for human in self.humans]
            else: ob = [human.get_observable_state() for human in self.humans]
        elif self.robot.sensor == 'RGB': raise NotImplementedError
        return ob

    def check_actual_collisions_and_goal(self):
        """
        Based on the current state, this function checks if the robot collides with any human and if it has reacheed the goal.
        """
        dmin = 10000.0
        collision = False
        for human in self.humans:
            distance = np.linalg.norm(human.position - self.robot.position) - human.radius - self.robot.radius
            dmin = distance if distance < dmin else dmin
            if dmin <= 0: collision = True
        reaching_goal = np.linalg.norm(self.robot.position - self.robot.get_goal_position()) < self.robot.radius
        return collision, dmin, reaching_goal

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        if self.robot is None: raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None: self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test': self.human_times = [0] * self.human_num
        else: self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
        # if not self.robot.policy.multiagent_training: self.train_val_sim = 'circle_crossing'
        if self.config.get('humans', 'policy') == 'trajnet': raise NotImplementedError
        else:
            counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],'val': 0, 'test': self.case_capacity['val']}
            if self.case_counter[phase] >= 0:
                np.random.seed(counter_offset[phase] + self.case_counter[phase])
                if phase in ['train', 'val']:
                    human_num = self.human_num if self.robot.policy.multiagent_training else 1
                    if self.train_val_sim == 'circle_crossing':
                        self.generate_circular_crossing_setting(insert_robot = True, human_policy = self.human_policy, headless = HEADLESS,
                                                                runge_kutta = False, robot_visible = self.robot.visible, robot_radius = self.robot_radius,
                                                                circle_radius = self.circle_radius, n_actors = human_num, randomize_human_positions = True, 
                                                                randomize_human_attributes = self.randomize_attributes)
                    elif self.train_val_sim == 'parallel_traffic':
                        self.generate_parallel_traffic_scenario(insert_robot = True, human_policy = self.human_policy, headless = HEADLESS,
                                                                runge_kutta = False, robot_visible = self.robot.visible, robot_radius = self.robot_radius,
                                                                traffic_length = self.traffic_length, traffic_height = self.traffic_height, n_actors = human_num, 
                                                                randomize_human_attributes = self.randomize_attributes)
                    elif self.train_val_sim == 'hybrid_scenario':
                        scenario = np.random.choice(['circle_crossing', 'parallel_traffic'])
                        np.random.seed(counter_offset[phase] + self.case_counter[phase])
                        if scenario == 'circle_crossing':
                            self.generate_circular_crossing_setting(insert_robot = True, human_policy = self.human_policy, headless = HEADLESS,
                                                                runge_kutta = False, robot_visible = self.robot.visible, robot_radius = self.robot_radius,
                                                                circle_radius = self.circle_radius, n_actors = human_num, randomize_human_positions = True, 
                                                                randomize_human_attributes = self.randomize_attributes)
                        elif scenario == 'parallel_traffic':
                            self.generate_parallel_traffic_scenario(insert_robot = True, human_policy = self.human_policy, headless = HEADLESS,
                                                                runge_kutta = False, robot_visible = self.robot.visible, robot_radius = self.robot_radius,
                                                                traffic_length = self.traffic_length, traffic_height = self.traffic_height, n_actors = human_num, 
                                                                randomize_human_attributes = self.randomize_attributes)
                else:
                    if self.test_sim == 'circle_crossing':
                        self.generate_circular_crossing_setting(insert_robot = True, human_policy = self.human_policy, headless = HEADLESS,
                                                                runge_kutta = False, robot_visible = self.robot.visible, robot_radius = self.robot_radius,
                                                                circle_radius = self.circle_radius, n_actors = self.human_num, randomize_human_positions = True, 
                                                                randomize_human_attributes = self.randomize_attributes)
                    elif self.test_sim == 'parallel_traffic':
                        self.generate_parallel_traffic_scenario(insert_robot = True, human_policy = self.human_policy, headless = HEADLESS,
                                                                runge_kutta = False, robot_visible = self.robot.visible, robot_radius = self.robot_radius,
                                                                traffic_length = self.traffic_length, traffic_height = self.traffic_height, n_actors = self.human_num, 
                                                                randomize_human_attributes = self.randomize_attributes)
                    elif self.test_sim == 'hybrid_scenario':
                        scenario = np.random.choice(['circle_crossing', 'parallel_traffic'])
                        np.random.seed(counter_offset[phase] + self.case_counter[phase])
                        if scenario == 'circle_crossing':
                            self.generate_circular_crossing_setting(insert_robot = True, human_policy = self.human_policy, headless = HEADLESS,
                                                                runge_kutta = False, robot_visible = self.robot.visible, robot_radius = self.robot_radius,
                                                                circle_radius = self.circle_radius, n_actors = self.human_num, randomize_human_positions = True, 
                                                                randomize_human_attributes = self.randomize_attributes)
                        elif scenario == 'parallel_traffic':
                            self.generate_parallel_traffic_scenario(insert_robot = True, human_policy = self.human_policy, headless = HEADLESS,
                                                                runge_kutta = False, robot_visible = self.robot.visible, robot_radius = self.robot_radius,
                                                                traffic_length = self.traffic_length, traffic_height = self.traffic_height, n_actors = self.human_num, 
                                                                randomize_human_attributes = self.randomize_attributes)
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
                else: raise NotImplementedError      
        # Set robot radius, sampling time for the simulation and reset the simulator and safety space for ORCA
        self.robot.set_radius_and_update_graphics(self.robot_radius)
        self.robot.set(self.config_data["robot"]["pos"][0], self.config_data["robot"]["pos"][1], self.config_data["robot"]["goals"][0][0], self.config_data["robot"]["goals"][0][1], 0, 0, self.config_data["robot"]["yaw"], w=0)
        self.reset_sim(reset_robot=False)
        if self.safety_space > 0: self.motion_model_manager.set_safety_space(self.safety_space)
        self.set_time_step(self.time_step)
        self.set_robot_time_step(self.robot_time_step)
        # Initialize some variables used later
        self.states = list()
        if hasattr(self.robot.policy, 'action_values'): self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'): self.attention_weights = list()
        # Get current observation and info
        ob = self.compute_humans_observable_state()
        info = Nothing()
        return ob, {0: info}

    def step(self, action):
        """
        Detect collision, compute reward and info, update environment and return (ob, reward, terminated, truncated, info)
        """
        # Collision detection and check if reaching the goal
        collision, dmin, reaching_goal = self.collision_detection_and_reaching_goal(action, self.robot_time_step)
        # Compute Reward, truncated, terminated, and info
        reward, terminated, truncated, info = self.compute_reward_and_infos(collision, dmin, reaching_goal, self.global_time, self.robot_time_step)
        # Store state, action value and attention weights
        self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
        if hasattr(self.robot.policy, 'action_values'): self.action_values.append(self.robot.policy.action_values)
        if hasattr(self.robot.policy, 'get_attention_weights'): self.attention_weights.append(self.robot.policy.get_attention_weights())
        # If the robot time step is different from the environment timestep, we apply the same action for robot_timestep/timestep times
        for _ in range(self.time_step_factor):
            # Update robot state
            self.robot.step(action, self.time_step)
            # Update humans state
            self.motion_model_manager.update_humans(self.global_time, self.time_step)
            self.global_time += self.time_step
        # Compute observation
        ob = self.compute_humans_observable_state()
        # Set the state of the gym env as updated
        self.updated = True
        return ob, reward, terminated, truncated, {0: info}

    def imitation_learning_step(self):
        """
        Makes a step of the environment when the robot is moving following a human policy.
        The imitation learning step computes the reward and infos with the actual state at the end of the update.
        """
        # Store state, action value and attention weights
        self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
        # If the robot time step is different from the environment timestep, we apply the same action for robot_timestep/timestep times
        for _ in range(self.time_step_factor):
            # Update robot state
            self.motion_model_manager.update_robot(self.global_time, self.time_step)
            # Update humans state
            self.motion_model_manager.update_humans(self.global_time, self.time_step)
            self.global_time += self.time_step
        # Compute observation
        ob = self.compute_humans_observable_state()
        # Collision detection and check if reaching the goal
        collision, dmin, reaching_goal = self.check_actual_collisions_and_goal()
        # Compute Reward, truncated, terminated, and info
        reward, terminated, truncated, info = self.compute_reward_and_infos(collision, dmin, reaching_goal, self.global_time, self.robot_time_step)
        # Set the state of the gym env as updated
        self.updated = True
        return ob, reward, terminated, truncated, {0: info}

    def render(self):
        if not HEADLESS: self.render_sim()
