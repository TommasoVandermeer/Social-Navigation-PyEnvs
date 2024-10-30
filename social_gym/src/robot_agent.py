import pygame
import math
from social_gym.src.agent import Agent
from social_gym.src.utils import bound_angle, PRECISION
from social_gym.src.sensors import LaserSensor
from social_gym.src.actuators import DifferentialDrive
import numpy as np
import logging
from crowd_nav.utils.state import JointState
from crowd_nav.utils.action import ActionXY, ActionRot
from crowd_nav.policy_no_train.policy_factory import policy_factory

KEYS_VELOCITY_CHANGE = {"up": np.array([1.20,1.20], dtype=PRECISION), "down": np.array([-1.20,-1.20], dtype=PRECISION), "left": np.array([-1.20,0.00], dtype=PRECISION), "right": np.array([0.00,-1.20], dtype=PRECISION)}

class RobotAgent(Agent):
    def __init__(self, game, pos=[7.5,7.5], yaw=0.0, radius=0.3, goals=list(), mass=80, desired_speed=1):
        super().__init__(np.array(pos, dtype=PRECISION), yaw, (255,0,0), radius, game.real_size, game.display_to_real_ratio, mass=mass, desired_speed=desired_speed)

        self.goals = goals
        self.collisions = 0 # Unused
        self.laser = None

        self.set_radius_and_update_graphics(self.radius)

    def set_radius_and_update_graphics(self, radius:float):
        self.radius = radius
        display_radius = radius * self.ratio
        self.image = pygame.Surface((display_radius * 2, display_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, self.color, (display_radius, display_radius), display_radius)
        pygame.draw.circle(self.image, (0,0,0), (display_radius + math.cos(0.0) * display_radius, display_radius - math.sin(0.0) * display_radius), display_radius / 3)
        self.original_image = self.image
        self.image = pygame.transform.rotate(self.original_image, math.degrees(self.yaw))
        self.rect = self.image.get_rect(center = tuple([self.position[0] * self.ratio, (self.real_size - self.position[1]) * self.ratio]))

    def check_collisions(self, humans, walls):
        for human in humans:
            distance = np.linalg.norm(self.position - human.position)
            if (distance < (human.radius + self.radius)):
                direction = (self.position - human.position) / distance
                self.position = human.position + direction * (human.radius + self.radius)
                self.move()
        for wall in walls:
            closest_point, distance = wall.get_closest_point(self.position)
            if (distance < self.radius):
                direction = (self.position - closest_point) / np.linalg.norm(closest_point - self.position)
                self.position = closest_point + direction * self.radius
                self.move()
        self.move()

    def render(self, display, scroll:np.array):
        if self.laser is not None and self.laser_render: self.render_laser(display, scroll)
        display.blit(self.image, (self.rect.x - scroll[0], self.rect.y - scroll[1]))

    def render_laser(self, display, scroll:np.array):
        self.laser_surface.fill((0,0,0,0))
        self.laser_surface_rect = self.laser_surface.get_rect(center = tuple([self.position[0] * self.ratio, (self.real_size - self.position[1]) * self.ratio]))
        for k, v in self.laser_data.items():
            end_position = np.empty((2,), dtype=PRECISION)
            end_position[0] = self.laser_start_position[0] + (v + self.radius) * math.cos(k)
            end_position[1] = self.laser_start_position[1] + (v + self.radius) * math.sin(k)
            end_display_position = np.empty((2,), dtype=PRECISION)
            end_display_position[0] = end_position[0] * self.ratio
            end_display_position[1] = (self.laser.max_distance * 2 - end_position[1]) * self.ratio
            pygame.draw.line(self.laser_surface, (255,0,0,100), tuple(self.laser_start_display_position), tuple(end_display_position), 2)
        display.blit(self.laser_surface, (self.laser_surface_rect.x - scroll[0], self.laser_surface_rect.y - scroll[1]))

    def add_laser_sensor(self, range:float, samples:int, max_distance:float, uncertainty=None, render=False):
        self.laser = LaserSensor(self.position, self.yaw, range, samples, max_distance, uncertainty=uncertainty)
        # For rendering
        self.laser_render = render
        self.laser_surface = pygame.Surface((self.laser.max_distance * self.ratio * 2, self.laser.max_distance * self.ratio * 2), pygame.SRCALPHA)
        self.laser_start_display_position = np.array([self.laser_surface.get_size()[0] / 2, self.laser_surface.get_size()[1] / 2], dtype=PRECISION)
        self.laser_start_position = np.empty((2,), dtype=PRECISION)
        self.laser_start_position[0] = self.laser_start_display_position[0] / self.ratio
        self.laser_start_position[1] = -(self.laser_start_display_position[1] / self.ratio) + self.laser.max_distance * 2

    def get_laser_readings(self, humans:list, walls):
        self.laser.update_pose(self.position, self.yaw)
        readings = self.laser.get_laser_measurements(humans, walls)
        # Subtract robot radius from readings (laser is robot pose centered)
        self.laser_data = {k: v - self.radius for k, v in readings.items()}
        return self.laser_data
    
    def mount_differential_drive(self, max_speed):
        self.diff_drive = DifferentialDrive(self.radius, max_speed)

    def move_with_keys(self, direction:str, robot_dt:float):    
        if not hasattr(self, "diff_drive"): raise ValueError("Differential drive is not mounted")
        self.diff_drive.change_velocity(self.diff_drive.velocity + KEYS_VELOCITY_CHANGE[direction] * robot_dt)

    ### METHODS FOR CROWDNAV POLICIES
        
    def set_policy(self, policy):
        self.policy = policy
        self.kinematics = policy.kinematics
        if 'hsfm' in policy.name: self.headed = True
        if policy.name == 'orca': self.orca = True

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None, w=None):
        self.position[0] = px
        self.position[1] = py
        self.goals.insert(0, [gx,gy])
        if len(self.goals) > 1: self.goals.pop()
        self.goals[0] = [gx,gy]
        self.linear_velocity[0] = vx
        self.linear_velocity[1] = vy
        self.yaw = theta
        if radius is not None: self.radius = radius
        if v_pref is not None: self.desired_speed = v_pref
        if w is not None: self.angular_velocity = w

    def check_validity(self, action):
        if self.kinematics == 'holonomic': assert isinstance(action, ActionXY)
        else: assert isinstance(action, ActionRot)
    
    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            act = np.array([action.vx, action.vy], dtype=PRECISION)
            position = self.position + act * delta_t
        else:
            act = np.array([np.cos(self.yaw + action.r) * action.v, np.sin(self.yaw + action.r) * action.v], dtype=PRECISION)
            position = self.position + act * delta_t
        return position

    def step(self, action, delta_t):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        self.position = self.compute_position(action, delta_t)
        if self.kinematics == 'holonomic':
            self.linear_velocity = np.array([action.vx, action.vy], dtype=PRECISION)
        else:
            self.yaw = (self.yaw + action.r) % (2 * np.pi)
            self.linear_velocity = np.array([np.cos(self.yaw) * action.v, np.sin(self.yaw) * action.v], dtype=PRECISION)

    def act(self, ob):
        if self.policy is None: raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def configure(self, config, section):
        self.visible = config.getboolean(section, 'visible')
        self.desired_speed = config.getfloat(section, 'v_pref')
        self.radius = config.getfloat(section, 'radius')
        self.policy = policy_factory[config.get(section, 'policy')]()
        self.sensor = config.get(section, 'sensor')
        self.kinematics = self.policy.kinematics if self.policy is not None else None
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None

    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format('visible' if self.visible else 'invisible', self.kinematics))