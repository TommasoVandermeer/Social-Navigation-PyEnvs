import pygame
import math
import numpy as np
from .agent import Agent

class HumanAgent(Agent):
    def __init__(self, game, label:int, model:str, pos:list[float], yaw:float, goals:list[list[float]], color=(0,0,0), radius=0.3, mass=75, des_speed=0.9, group_id=-1):
        super().__init__(pos, yaw, color, radius, game.real_size, game.display_to_real_ratio)

        self.motion_model = model
        display_radius = self.radius * self.ratio
        self.desired_speed = des_speed
        self.group_id = group_id
        self.goals = goals
        self.obstacles = []

        if (self.motion_model == 'sfm_roboticsupo'):
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
            # SFM Forces
            self.desired_force = np.array([0.0,0.0], dtype=np.float64)
            self.obstacle_force = np.array([0.0,0.0], dtype=np.float64)
            self.social_force = np.array([0.0,0.0], dtype=np.float64)
            self.group_force = np.array([0.0,0.0], dtype=np.float64)
            self.global_force = np.array([0.0,0.0], dtype=np.float64)
            # SFM Velocities
            self.linear_velocity = np.array([0.0,0.0], dtype=np.float64)
            self.angular_velocity = 0
        elif (self.motion_model == 'sfm_helbing'):
            # SFM Parameters
            self.mass = mass
            self.relaxation_time = 0.5
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
            # SFM Forces
            self.desired_force = np.array([0.0,0.0], dtype=np.float64)
            self.obstacle_force = np.array([0.0,0.0], dtype=np.float64)
            self.social_force = np.array([0.0,0.0], dtype=np.float64)
            self.group_force = np.array([0.0,0.0], dtype=np.float64)
            self.global_force = np.array([0.0,0.0], dtype=np.float64)
            # SFM Velocities
            self.linear_velocity = np.array([0.0,0.0], dtype=np.float64)
            self.angular_velocity = 0

        self.font = pygame.font.Font('fonts/Roboto-Black.ttf',int(0.25 * self.ratio))
        self.label = self.font.render(f"{label}", False, (0,0,0))

        self.image = pygame.Surface((display_radius * 2, display_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (255,255,255), (display_radius, display_radius), display_radius)
        pygame.draw.circle(self.image, self.color, (display_radius, display_radius), display_radius, int(0.05 * self.ratio))
        if 'hsfm' in self.motion_model: pygame.draw.circle(self.image, (0,0,255), (display_radius + math.cos(0.0) * display_radius, display_radius - math.sin(0.0) * display_radius), display_radius / 3)
        self.original_image = self.image

        self.image = pygame.transform.rotate(self.original_image, math.degrees(self.yaw))
        self.rect = self.image.get_rect(center = tuple([self.position[0] * self.ratio, (self.real_size - self.position[1]) * self.ratio]))
        self.label_rect = self.label.get_rect(center = tuple([self.position[0] * self.ratio, (self.real_size - self.position[1]) * self.ratio]))

    def update(self, walls):
        self.move()
        self.rotate()
        # Update ostacles
        self.obstacles.clear()
        for wall in walls:
            obstacle, distance = wall.get_closest_point(self.position)
            self.obstacles.append(obstacle)

    def render_label(self, display):
        self.label_rect.centerx = round(self.position[0] * self.ratio)
        self.label_rect.centery = round((self.real_size - self.position[1]) * self.ratio)
        display.blit(self.label, self.label_rect)