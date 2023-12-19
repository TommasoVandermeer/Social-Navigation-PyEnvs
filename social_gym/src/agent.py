import pygame
import math
import numpy as np
from crowd_nav.utils.state import ObservableState, FullState

class Agent():
    def __init__(self, position:list[float], yaw:float, color:tuple, radius:float, real_size:float, display_ratio:float, mass=80, desired_speed=1):
        self.position = np.array(position, dtype=np.float64)
        self.yaw = yaw
        self.color = color
        self.real_size = real_size
        self.ratio = display_ratio
        self.radius = radius
        self.obstacles = []
        self.mass = mass
        self.desired_speed = desired_speed
        self.linear_velocity = np.array([0.0,0.0])
        self.body_velocity = np.array([0.0,0.0],dtype=np.float64)
        self.angular_velocity = 0.0
        self.headed = False
        self.orca = False
        self.desired_force = np.array([0.0,0.0], dtype=np.float64)
        self.obstacle_force = np.array([0.0,0.0], dtype=np.float64)
        self.social_force = np.array([0.0,0.0], dtype=np.float64)
        self.group_force = np.array([0.0,0.0], dtype=np.float64)
        self.global_force = np.array([0.0,0.0], dtype=np.float64)
        self.torque_force = 0.0
        self.inertia = 0.5 * self.mass * self.radius * self.radius
        self.rotational_matrix = np.array([[0.0,0.0],[0.0,0.0]],dtype=np.float64)
        self.k_theta = 0.0
        self.k_omega = 0.0
        # These are initialized in the children classes
        self.goals = list()
        self.policy = None
        self.kinematics = None
        self.sensor = None
        self.visible = None

    def get_goal_position(self):
        return np.array([self.goals[0][0], self.goals[0][1]])
    
    def get_position(self):
        return self.position
    
    def get_pose(self):
        pose = np.append(self.position, self.yaw)
        return pose
    
    def set_pose(self, pose:np.array):
        self.position = pose[0:2]
        self.yaw = pose[2]

    def move(self):
        self.rect.centerx = round(self.position[0] * self.ratio)
        self.rect.centery = round((self.real_size - self.position[1]) * self.ratio)

    def update(self):
        self.move()
        self.rotate()

    def rotate(self):
        self.image = pygame.transform.rotate(self.original_image, math.degrees(self.yaw))
        self.rect = self.image.get_rect(center = tuple([self.position[0] * self.ratio, (self.real_size - self.position[1]) * self.ratio]))

    def render(self, display, scroll:np.array):
        display.blit(self.image, (self.rect.x - scroll[0], self.rect.y - scroll[1]))
    
    def get_rect(self):
        return self.rect
    
    def compute_rotational_matrix(self):
        self.rotational_matrix = np.array([[np.cos(self.yaw), -np.sin(self.yaw)],[np.sin(self.yaw), np.cos(self.yaw)]], dtype=np.float64)

    def set_parameters(self, model:str):
        if (model == 'sfm_roboticsupo'):
            # SFM Parameters
            self.goal_weight = 2.0
            self.obstacle_weight = 10.0
            self.social_weight = 15.0
            self.group_gaze_weight = 3.0
            self.group_coh_weight = 2.0
            self.group_rep_weight = 1.0
            self.relaxation_time = 0.5
            self.obstacle_sigma = 0.2
            self.agent_lambda = 2.0
            self.agent_gamma = 0.35
            self.agent_nPrime = 3.0
            self.agent_n = 2.0
        elif (model == 'sfm_helbing'):
            # SFM Parameters
            self.relaxation_time = 0.5
            self.Ai = 2000.0
            self.Aw = 2000.0
            self.Bi = 0.08
            self.Bw = 0.08
            self.k1 = 120000.0
            self.k2 = 240000.0
        elif (model == 'sfm_guo'):
            # SFM Parameters
            self.relaxation_time = 0.5
            self.Ai = 2000.0
            self.Aw = 2000.0
            self.Bi = 0.08
            self.Bw = 0.08
            self.Ci = 120.0
            self.Cw = 120.0
            self.Di = 0.6
            self.Dw = 0.6
            self.k1 = 120000.0
            self.k2 = 240000.0
        elif (model == 'sfm_moussaid'):
            # SFM Parameters
            self.relaxation_time = 0.5
            self.Ei = 360
            self.agent_lambda = 2.0
            self.gamma = 0.35
            self.ns = 2.0
            self.ns1 = 3.0
            self.Aw = 2000.0
            self.Bw = 0.08
            self.k1 = 120000.0
            self.k2 = 240000.0
        elif (model == 'hsfm_farina'):
            # HSFM Parameters
            self.relaxation_time = 0.5
            self.Ai = 2000.0
            self.Aw = 2000.0
            self.Bi = 0.08
            self.Bw = 0.08
            self.k1 = 120000.0
            self.k2 = 240000.0
            self.ko = 1.0
            self.kd = 500.0
            self.alpha = 3.0
            self.k_lambda = 0.3
            # self.group_distance_forward = 2.0
            # self.group_distance_orthogonal = 1.0
            # self.k1g = 200.0
            # self.k2g = 200.0
        elif (model == 'hsfm_guo'):
            # HSFM Parameters
            self.relaxation_time = 0.5
            self.Ai = 2000.0
            self.Aw = 2000.0
            self.Bi = 0.08
            self.Bw = 0.08
            self.Ci = 120.0
            self.Cw = 120.0
            self.Di = 0.6
            self.Dw = 0.6
            self.k1 = 120000.0
            self.k2 = 240000.0
            self.ko = 1.0
            self.kd = 500.0
            self.alpha = 3.0
            self.k_lambda = 0.3
            # self.group_distance_forward = 2.0
            # self.group_distance_orthogonal = 1.0
            # self.k1g = 200.0
            # self.k2g = 200.0
        elif (model == 'hsfm_moussaid'):
            # HSFM Parameters
            self.relaxation_time = 0.5
            self.Ei = 360
            self.agent_lambda = 2.0
            self.gamma = 0.35
            self.ns = 2.0
            self.ns1 = 3.0
            self.Aw = 2000.0
            self.Bw = 0.08
            self.k1 = 120000.0
            self.k2 = 240000.0
            self.ko = 1.0
            self.kd = 500.0
            self.alpha = 3.0
            self.k_lambda = 0.3
            # self.group_distance_forward = 2.0
            # self.group_distance_orthogonal = 1.0
            # self.k1g = 200.0
            # self.k2g = 200.0
        elif (model == 'hsfm_new'):
            # HSFM Parameters
            self.relaxation_time = 0.5
            self.Ai = 2000.0
            self.Aw = 2000.0
            self.Bi = 0.08
            self.Bw = 0.08
            self.k1 = 120000.0
            self.k2 = 240000.0
            self.ko = 1.0
            self.kd = 500.0
            self.alpha = 3.0
            self.k_lambda = 0.3
            # self.group_distance_forward = 2.0
            # self.group_distance_orthogonal = 1.0
            # self.k1g = 200.0
            # self.k2g = 200.0
        elif (model == 'hsfm_new_guo'):
            # HSFM Parameters
            self.relaxation_time = 0.5
            self.Ai = 2000.0
            self.Aw = 2000.0
            self.Bi = 0.08
            self.Bw = 0.08
            self.Ci = 120.0
            self.Cw = 120.0
            self.Di = 0.6
            self.Dw = 0.6
            self.k1 = 120000.0
            self.k2 = 240000.0
            self.ko = 1.0
            self.kd = 500.0
            self.alpha = 3.0
            self.k_lambda = 0.3 # 0.1
            # self.group_distance_forward = 2.0
            # self.group_distance_orthogonal = 1.0
            # self.k1g = 200.0
            # self.k2g = 200.0
        elif (model == 'hsfm_new_moussaid'):
            # HSFM Parameters
            self.relaxation_time = 0.5
            self.Ei = 360
            self.agent_lambda = 2.0
            self.gamma = 0.35
            self.ns = 2.0
            self.ns1 = 3.0
            self.Aw = 2000.0
            self.Bw = 0.08
            self.k1 = 120000.0
            self.k2 = 240000.0
            self.ko = 1.0
            self.kd = 500.0
            self.alpha = 3.0
            self.k_lambda = 0.3 # 0.1
            # self.group_distance_forward = 2.0
            # self.group_distance_orthogonal = 1.0
            # self.k1g = 200.0
            # self.k2g = 200.0

    ### METHODS FOR CROWDNAV POLICIES

    def get_observable_state(self):
        return ObservableState(self.position[0], self.position[1], self.linear_velocity[0], self.linear_velocity[1], self.radius)

    def get_full_state(self):
        return FullState(self.position[0], self.position[1], self.linear_velocity[0], self.linear_velocity[1], self.radius, self.goals[0][0], self.goals[0][1], self.desired_speed, self.yaw)
