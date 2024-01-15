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
        linear_velocity = np.sum(self.velocity) * 0.5
        new_x_pos = position[0] + linear_velocity * math.cos(yaw) * dt
        new_y_pos = position[1] + linear_velocity * math.sin(yaw) * dt
        new_position = np.array([new_x_pos, new_y_pos], dtype=np.float64)
        new_yaw = bound_angle(yaw + (self.velocity[1] - self.velocity[0]) / self.width * dt)
        return new_position, new_yaw
    
    def change_velocity(self, new_velocity:np.array):
        self.velocity = new_velocity
        self.velocity = self.bound_velocity(self.velocity, self.max_speed)
    
    def change_velocity_linear_angular(self, linear_velocity:float, angular_rate:float):
        left_velocity = linear_velocity - 0.5 * angular_rate * self.width
        right_velocity = linear_velocity + 0.5 * angular_rate * self.width
        self.velocity = np.array([left_velocity, right_velocity], dtype=np.float64)
        self.velocity = self.bound_velocity(self.velocity, self.max_speed)

    def bound_velocity(self, velocity, max_speed):
        new_velocity = velocity.copy()
        # Bound velocity between [-max_speed, max_speed]
        if new_velocity[0] > max_speed: new_velocity[0] = max_speed
        if new_velocity[1] > max_speed: new_velocity[1] = max_speed
        if new_velocity[0] < -max_speed: new_velocity[0] = -max_speed
        if new_velocity[1] < -max_speed: new_velocity[1] = -max_speed
        return new_velocity
        