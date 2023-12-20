import pygame
import math
import numpy as np
import os
from social_gym.src.agent import Agent

class HumanAgent(Agent):
    def __init__(self, game, label:int, model:str, pos:list[float], yaw:float, goals:list[list[float]], color=(0,0,0), radius=0.3, mass=80, des_speed=1, group_id=-1):
        super().__init__(pos, yaw, color, radius, game.real_size, game.display_to_real_ratio, mass=mass, desired_speed=des_speed)

        self.motion_model = model
        display_radius = self.radius * self.ratio
        self.group_id = group_id
        self.goals = goals

        self.set_parameters(self.motion_model)

        self.font = pygame.font.Font(os.path.join(os.path.dirname(__file__),'..','fonts/Roboto-Black.ttf'),int(0.25 * self.ratio))
        self.label = self.font.render(f"{label}", False, (0,0,0))

        self.image = pygame.Surface((display_radius * 2, display_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (255,255,255), (display_radius, display_radius), display_radius)
        pygame.draw.circle(self.image, self.color, (display_radius, display_radius), display_radius, int(0.05 * self.ratio))
        if 'hsfm' in self.motion_model: pygame.draw.circle(self.image, (0,0,255), (display_radius + math.cos(0.0) * display_radius, display_radius - math.sin(0.0) * display_radius), display_radius / 3)
        self.original_image = self.image

        self.image = pygame.transform.rotate(self.original_image, math.degrees(self.yaw))
        self.rect = self.image.get_rect(center = tuple([self.position[0] * self.ratio, (self.real_size - self.position[1]) * self.ratio]))
        self.label_rect = self.label.get_rect(center = tuple([self.position[0] * self.ratio, (self.real_size - self.position[1]) * self.ratio]))

    def render(self, display, scroll:np.array):
        display.blit(self.image, (self.rect.x - scroll[0], self.rect.y - scroll[1]))
        self.render_label(display, scroll)

    def render_label(self, display, scroll:np.array):
        self.label_rect.centerx = round(self.position[0] * self.ratio)
        self.label_rect.centery = round((self.real_size - self.position[1]) * self.ratio)
        display.blit(self.label, (self.label_rect.x - scroll[0], self.label_rect.y - scroll[1]))