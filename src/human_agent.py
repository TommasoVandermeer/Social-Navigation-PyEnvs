import pygame
import math

class HumanAgent(pygame.sprite.Sprite):
    def __init__(self, game, model:str, pos:list[float], yaw:float, color=(0,0,0), radius=0.3, mass=75):
        super().__init__()
        self.ratio = game.display_to_real_ratio

        self.motion_model = model
        self.color = color
        self.radius = radius * self.ratio
        self.mass = mass
        self.image = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        self.image.fill((255,255,255))
        pygame.draw.circle(self.image, self.color, (self.radius, self.radius), self.radius, int(0.05 * self.ratio))

        self.position = [pos[0] * self.ratio, pos[1] * self.ratio]
        self.yaw = yaw

        pygame.draw.circle(self.image, (0,0,255), (self.radius + math.cos(self.yaw) * self.radius, self.radius - math.sin(self.yaw) * self.radius), self.radius / 3)

        self.rect = self.image.get_rect(center = tuple(self.position))

        self.original_image = self.image

    def update(self):
        pass