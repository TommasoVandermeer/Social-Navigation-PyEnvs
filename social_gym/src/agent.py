import pygame
import math
import numpy as np
from crowd_nav.utils.state import ObservableState, FullState, ObservableStateHeaded
from social_gym.src.utils import PRECISION

class Agent():
    def __init__(self, position:list[float], yaw:float, color:tuple, radius:float, real_size:float, display_ratio:float, mass=80, desired_speed=1):
        self.position = np.array(position, dtype=PRECISION)
        self.yaw = yaw
        self.color = color
        self.real_size = real_size
        self.ratio = display_ratio
        self.radius = radius
        self.safety_space = 0
        self.obstacles = []
        self.mass = mass
        self.desired_speed = desired_speed
        self.linear_velocity = np.array([0.0,0.0],dtype=PRECISION)
        self.body_velocity = np.array([0.0,0.0],dtype=PRECISION)
        self.angular_velocity = 0.0
        self.headed = False
        self.orca = False
        self.desired_force = np.array([0.0,0.0], dtype=PRECISION)
        self.obstacle_force = np.array([0.0,0.0], dtype=PRECISION)
        self.social_force = np.array([0.0,0.0], dtype=PRECISION)
        self.group_force = np.array([0.0,0.0], dtype=PRECISION)
        self.global_force = np.array([0.0,0.0], dtype=PRECISION)
        self.torque_force = 0.0
        self.inertia = 0.5 * self.mass * self.radius * self.radius
        self.rotational_matrix = np.array([[0.0,0.0],[0.0,0.0]],dtype=PRECISION)
        self.k_theta = 0.0
        self.k_omega = 0.0
        # These are initialized in the children classes
        self.goals = list()
        self.policy = None
        self.kinematics = None
        self.sensor = None
        self.visible = None

    def get_goal_position(self):
        return np.array([self.goals[0][0], self.goals[0][1]], dtype=PRECISION)
    
    def get_position(self):
        return self.position
    
    def get_pose(self):
        pose = np.append(self.position, self.yaw)
        return pose
    
    def set_pose(self, pose:np.array):
        self.position = pose[0:2]
        self.yaw = pose[2]

    def set_goals(self, goals:list[list[float]]):
        self.goals = goals

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
        self.rotational_matrix = np.array([[np.cos(self.yaw), -np.sin(self.yaw)],[np.sin(self.yaw), np.cos(self.yaw)]], dtype=PRECISION)

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
            self.k_lambda = 0.1 # Default: 0.3
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
            self.k_lambda = 0.1 # Default: 0.3
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
            self.k_lambda = 0.1 # Default: 0.3
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
            self.k_lambda = 0.1 # Default: 0.3
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
            self.k_lambda = 0.1 # Default: 0.3
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
            self.k_lambda = 0.1 # Default: 0.3
            # self.group_distance_forward = 2.0
            # self.group_distance_orthogonal = 1.0
            # self.k1g = 200.0
            # self.k2g = 200.0

    ### METHODS FOR CROWDNAV POLICIES

    def get_observable_state(self, visible_theta_and_omega=False):
        if visible_theta_and_omega: return ObservableStateHeaded(self.position[0], self.position[1], self.linear_velocity[0], self.linear_velocity[1], self.radius, self.yaw, self.angular_velocity)
        else: return ObservableState(self.position[0], self.position[1], self.linear_velocity[0], self.linear_velocity[1], self.radius)

    def get_full_state(self):
        return FullState(self.position[0], self.position[1], self.linear_velocity[0], self.linear_velocity[1], self.radius, self.goals[0][0], self.goals[0][1], self.desired_speed, self.yaw)
    
    ### METHODS ADDED FOR PARALLELISM

    def get_safe_state(self):
        return np.array([*np.copy(self.position),self.yaw,*np.copy(self.linear_velocity),*np.copy(self.body_velocity),self.angular_velocity,
                         self.radius,self.mass,*self.goals[0],self.desired_speed], PRECISION)
    
    def set_state(self, pose_and_velocity:np.ndarray):
        # [px,py,theta,vx,vy,bvx,bvy,omega]
        self.position = pose_and_velocity[0:2]
        self.yaw = pose_and_velocity[2]
        self.linear_velocity = pose_and_velocity[3:5]
        self.body_velocity = pose_and_velocity[5:7]
        self.angular_velocity = pose_and_velocity[7]

    def get_parameters(self, model:str):
        # Params array should be of the form: [relax_t,Ai,Aw,Bi,Bw,Ci,Cw,Di,Dw,Ei,k1,k2,a_lambda,gamma,ns,ns1,ko,kd,alpha,k_lambda] (length = 20)
        params = np.zeros((20,), PRECISION)
        if (model == 'sfm_helbing'):
            params[0] = self.relaxation_time
            params[1] = self.Ai
            params[2] = self.Aw
            params[3] = self.Bi
            params[4] = self.Bw
            params[10] = self.k1
            params[11] = self.k2
        elif (model == 'sfm_guo'):
            params[0] = self.relaxation_time
            params[1] = self.Ai
            params[2] = self.Aw
            params[3] = self.Bi
            params[4] = self.Bw
            params[5] = self.Ci
            params[6] = self.Cw
            params[7] = self.Di
            params[8] = self.Dw
            params[10] = self.k1
            params[11] = self.k2
        elif (model == 'sfm_moussaid'):
            params[0] = self.relaxation_time
            params[9] = self.Ei
            params[12] = self.agent_lambda
            params[13] = self.gamma
            params[14] = self.ns
            params[15] = self.ns1
            params[2] = self.Aw
            params[4] = self.Bw
            params[10] = self.k1
            params[11] = self.k2
        elif (model == 'hsfm_farina'):
            params[0] = self.relaxation_time
            params[1] = self.Ai
            params[2] = self.Aw
            params[3] = self.Bi
            params[4] = self.Bw
            params[10] = self.k1
            params[11] = self.k2
            params[16] = self.ko
            params[17] = self.kd
            params[18] = self.alpha
            params[19] = self.k_lambda
        elif (model == 'hsfm_guo'):
            params[0] = self.relaxation_time
            params[1] = self.Ai
            params[2] = self.Aw
            params[3] = self.Bi
            params[4] = self.Bw
            params[5] = self.Ci
            params[6] = self.Cw
            params[7] = self.Di
            params[8] = self.Dw
            params[10] = self.k1
            params[11] = self.k2
            params[16] = self.ko
            params[17] = self.kd
            params[18] = self.alpha
            params[19] = self.k_lambda
        elif (model == 'hsfm_moussaid'):
            params[0] = self.relaxation_time
            params[9] = self.Ei
            params[12] = self.agent_lambda
            params[13] = self.gamma
            params[14] = self.ns
            params[15] = self.ns1
            params[2] = self.Aw
            params[4] = self.Bw
            params[10] = self.k1
            params[11] = self.k2
            params[16] = self.ko
            params[17] = self.kd
            params[18] = self.alpha
            params[19] = self.k_lambda
        elif (model == 'hsfm_new'):
            params[0] = self.relaxation_time
            params[1] = self.Ai
            params[2] = self.Aw
            params[3] = self.Bi
            params[4] = self.Bw
            params[10] = self.k1
            params[11] = self.k2
            params[16] = self.ko
            params[17] = self.kd
            params[18] = self.alpha
            params[19] = self.k_lambda
        elif (model == 'hsfm_new_guo'):
            params[0] = self.relaxation_time
            params[1] = self.Ai
            params[2] = self.Aw
            params[3] = self.Bi
            params[4] = self.Bw
            params[5] = self.Ci
            params[6] = self.Cw
            params[7] = self.Di
            params[8] = self.Dw
            params[10] = self.k1
            params[11] = self.k2
            params[16] = self.ko
            params[17] = self.kd
            params[18] = self.alpha
            params[19] = self.k_lambda
        elif (model == 'hsfm_new_moussaid'):
            params[0] = self.relaxation_time
            params[9] = self.Ei
            params[12] = self.agent_lambda
            params[13] = self.gamma
            params[14] = self.ns
            params[15] = self.ns1
            params[2] = self.Aw
            params[4] = self.Bw
            params[10] = self.k1
            params[11] = self.k2
            params[16] = self.ko
            params[17] = self.kd
            params[18] = self.alpha
            params[19] = self.k_lambda
        return params
