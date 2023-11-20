import pygame
import math
import numpy as np
from src.state import ObservableState, FullState
from src.action import ActionXY, ActionRot

class Agent():
    def __init__(self, position:list[float], yaw:float, color:tuple, radius:float, real_size:float, display_ratio:float):
        self.position = np.array(position, dtype=np.float64)
        self.yaw = yaw
        self.color = color
        self.real_size = real_size
        self.ratio = display_ratio
        self.radius = radius
        self.linear_velocity = np.array([0.0,0.0])
        self.body_velocity = np.array([0.0,0.0],dtype=np.float64)
        self.angular_velocity = 0.0
        # These are initialized in the children classes
        self.goals = None
        self.desired_speed = None
        self.policy = None
        self.kinematics = None
        self.sensor = None

    def get_observable_state(self):
        return ObservableState(self.position[0], self.position[1], self.linear_velocity[0], self.linear_velocity[1], self.radius)

    def get_full_state(self):
        return FullState(self.position[0], self.position[1], self.linear_velocity[0], self.linear_velocity[1], self.radius, self.goals[0][0], self.goals[0][1], self.desired_speed, self.yaw)

    def set_policy(self, policy):
        self.policy = policy
        self.kinematics = policy.kinematics

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.position[0] = px
        self.position[1] = py
        self.goals[0][0] = gx
        self.goals[0][1] = gy
        self.linear_velocity[0] = vx
        self.linear_velocity[1] = vy
        self.yaw = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.desired_speed = v_pref

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)
    
    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            act = np.array([action.vx, action.vy])
            position = self.position + act * delta_t
        else:
            act = np.array([np.cos(self.yaw + action.r) * action.v, np.sin(self.yaw + action.r) * action.v])
            position = self.position + act * delta_t
        return position

    def get_goal_position(self):
        return np.array(self.goals[0][0], self.goals[0][1])

    def move(self):
        self.rect.centerx = round(self.position[0] * self.ratio)
        self.rect.centery = round((self.real_size - self.position[1]) * self.ratio)

    def rotate(self):
        self.image = pygame.transform.rotate(self.original_image, math.degrees(self.yaw))
        self.rect = self.image.get_rect(center = tuple([self.position[0] * self.ratio, (self.real_size - self.position[1]) * self.ratio]))

    def render(self, display, scroll:np.array):
        display.blit(self.image, (self.rect.x - scroll[0], self.rect.y - scroll[1]))
    
    def get_rect(self):
        return self.rect