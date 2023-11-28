import pygame
import math
from social_gym.src.agent import Agent
from social_gym.src.utils import bound_angle
from social_gym.src.state import JointState
from social_gym.policy.policy_factory import policy_factory
import numpy as np

class RobotAgent(Agent):
    def __init__(self, game, pos=[7.5,7.5], yaw=0.0, radius=0.25, goals=list()):
        super().__init__(np.array(pos, dtype=np.float64), yaw, (255,0,0), radius, game.real_size, game.display_to_real_ratio)

        display_radius = self.radius * self.ratio
        self.image = pygame.Surface((display_radius * 2, display_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, self.color, (display_radius, display_radius), display_radius)

        self.goals = goals

        self.headed = False
        self.orca = False

        self.collisions = 0 # Unused

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

    def move_with_keys(self, direction, humans, walls):    
        if direction == 'up':
            self.position[0] += math.cos(self.yaw) * 0.01
            self.position[1] += math.sin(self.yaw) * 0.01
        elif direction == 'down':
            self.position[0] -= math.cos(self.yaw) * 0.01
            self.position[1] -= math.sin(self.yaw) * 0.01
        elif direction == 'left':
            self.yaw = bound_angle(self.yaw + 0.1)
        elif direction == 'right':
            self.yaw = bound_angle(self.yaw - 0.1)
        self.check_collisions(humans, walls)

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
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