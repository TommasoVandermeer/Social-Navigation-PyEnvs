import numpy as np
import math
from social_gym.src.utils import bound_angle

class DifferentialDrive:
    """
    This class is used to simulate a differential drive.
    """
    def __init__(self, robot_width:float, max_speed=1.0):
        self.velocity = np.array([0,0], dtype = np.float64) # [Left wheels velocity, right wheels velocity]
        self.width = robot_width
        self.max_speed = max_speed

    def update_pose(self, position:np.array, yaw:float, dt:float):
        linear_velocity = np.sum(self.velocity) / 2
        new_x_pos = position[0] + linear_velocity * math.cos(yaw) * dt
        new_y_pos = position[1] + linear_velocity * math.sin(yaw) * dt
        new_position = np.array([new_x_pos, new_y_pos], dtype=np.float64)
        new_yaw = bound_angle(yaw + (self.velocity[1] - self.velocity[0]) / self.width * dt)
        return new_position, new_yaw
    
    def change_velocity(self, new_velocity:np.array):
        self.velocity = new_velocity
        # Bound velocity between [-self.max_speed, self.max_speed]
        if self.velocity[0] > self.max_speed: self.velocity[0] = self.max_speed
        if self.velocity[1] > self.max_speed: self.velocity[1] = self.max_speed
        if self.velocity[0] < -self.max_speed: self.velocity[0] = -self.max_speed
        if self.velocity[1] < -self.max_speed: self.velocity[1] = -self.max_speed
        