import pygame
import math
from src.utils import bound_angle

class RobotAgent(pygame.sprite.Sprite):
    def __init__(self, game):
        super().__init__()
        self.ratio = game.display_to_real_ratio

        self.color = (255,0,0)
        self.radius = 0.25 * self.ratio
        self.image = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        self.image.fill((255,255,255))
        pygame.draw.circle(self.image, self.color, (self.radius, self.radius), self.radius)

        self.position = [game.display.get_size()[0]/2, game.display.get_size()[1]/2]
        self.yaw = 0.0

        pygame.draw.circle(self.image, (0,0,0), (self.radius + math.cos(self.yaw) * self.radius, self.radius - math.sin(self.yaw) * self.radius), self.radius / 3)

        self.rect = self.image.get_rect(center = tuple(self.position))

        self.original_image = self.image

    def update(self):
        pass

    def render(self, display):
        display.blit(self.image, self.rect)

    def move_with_keys(self, direction):
        if direction == 'up':
            self.position[0] += math.cos(self.yaw)
            self.position[1] -= math.sin(self.yaw)
            self.rect.centerx = round(self.position[0])
            self.rect.centery = round(self.position[1])
        if direction == 'down':
            self.position[0] -= math.cos(self.yaw)
            self.position[1] += math.sin(self.yaw)
            self.rect.centerx = round(self.position[0])
            self.rect.centery = round(self.position[1])
        if direction == 'left':
            self.yaw = bound_angle(self.yaw + 0.1)
            self.image = pygame.transform.rotate(self.original_image, math.degrees(self.yaw))
            self.rect = self.image.get_rect(center = tuple(self.position))
        if direction == 'right':
            self.yaw = bound_angle(self.yaw - 0.1)
            self.image = pygame.transform.rotate(self.original_image, math.degrees(self.yaw))
            self.rect = self.image.get_rect(center = tuple(self.position))
