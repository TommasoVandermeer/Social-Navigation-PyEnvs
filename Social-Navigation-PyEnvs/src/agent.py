import pygame
import math
import numpy as np

class Agent():
    def __init__(self, position:list[float], yaw:float, color:tuple, radius:float, real_size:float, display_ratio:float):
        self.position = np.array(position, dtype=np.float64)
        self.yaw = yaw
        self.color = color
        self.real_size = real_size
        self.ratio = display_ratio
        self.radius = radius

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