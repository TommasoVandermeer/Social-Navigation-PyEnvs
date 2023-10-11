import pygame
import math
import numpy as np
from src.utils import points_distance, points_unit_vector, vector_angle

class HumanAgent(pygame.sprite.Sprite):
    def __init__(self, game, model:str, pos:list[float], yaw:float, goals:list[list[float]], color=(0,0,0), radius=0.3, mass=75, des_speed=0.9, group_id=-1):
        super().__init__()
        self.ratio = game.display_to_real_ratio
        self.real_size = game.real_size
        self.color = color

        self.motion_model = model
        self.radius = radius 
        radius = self.radius * self.ratio
        self.desired_speed = des_speed
        self.group_id = group_id
        self.goals = goals
        self.obstacles = []

        if (self.motion_model == 'sfm'):
            # SFM Parameters
            self.goal_weight = 2.0
            self.obstacle_weight = 10.0
            self.social_weight = 30.0
            self.group_gaze_weight = 3.0
            self.group_coh_weight = 2.0
            self.group_rep_weight = 1.0
            self.relaxation_time = 0.5
            self.obstacle_sigma = 0.2
            self.agent_lambda = 2.0
            self.agent_gamma = 0.35
            self.agent_nPrime = 3.0
            self.agent_n = 2.0
            # SFM Forces
            self.desired_force = np.array([0.0,0.0])
            self.obstacle_force = np.array([0.0,0.0])
            self.social_force = np.array([0.0,0.0])
            self.group_force = np.array([0.0,0.0])
            self.global_force = np.array([0.0,0.0])
            # SFM Velocities
            self.linear_velocity = np.array([0.0,0.0])
            self.angular_velocity = 0
        elif (self.motion_model == 'hsfm'):
            self.mass = mass
            self.relaxation_time = 0.5
            self.k_orthogonal = 1.0
            self.k_damping = 500.0
            self.k_lambda = 0.3
            self.alpha = 3.0
            self.group_distance_forward = 2.0
            self.group_distance_orthogonal = 1.0
            self.k1g = 200.0
            self.k2g = 200.0
            self.Ai = 2000.0
            self.Aw = 2000.0
            self.Bi = 0.08
            self.Bw = 0.08
            self.k1 = 120000.0
            self.k2 = 240000.0

        self.image = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        #self.image.fill((0,0,0))
        pygame.draw.circle(self.image, self.color, (radius, radius), radius, int(0.05 * self.ratio))
        pygame.draw.circle(self.image, (0,0,255), (radius + math.cos(0.0) * radius, radius - math.sin(0.0) * radius), radius / 3)
        self.original_image = self.image

        # POSE must always be regarded in the real frame  
        self.position = pos
        self.yaw = yaw

        self.image = pygame.transform.rotate(self.original_image, math.degrees(self.yaw))
        self.rect = self.image.get_rect(center = tuple([self.position[0] * self.ratio, (self.real_size - self.position[1]) * self.ratio]))

    def get_rect(self):
        return self.rect
    
    def rotate(self):
        self.image = pygame.transform.rotate(self.original_image, math.degrees(self.yaw))
        self.rect = self.image.get_rect(center = tuple([self.position[0] * self.ratio, (self.real_size - self.position[1]) * self.ratio]))

    def move(self):
        self.rect.centerx = round(self.position[0] * self.ratio)
        self.rect.centery = round((self.real_size - self.position[1]) * self.ratio)

    def update(self, walls):
        self.move()
        self.rotate()
        # Update ostacles
        self.obstacles.clear()
        for wall in walls:
            self.obstacles.append(np.array(wall.get_closest_point(self.position)))