import pygame
import math
from src.utils import bound_angle, points_distance, points_unit_vector, vector_angle
import numpy as np

class RobotAgent():
    def __init__(self, game):
        super().__init__()
        self.ratio = game.display_to_real_ratio
        self.real_size = game.real_size

        self.color = (255,0,0)
        self.radius = 0.25
        radius = self.radius * self.ratio
        self.image = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        #self.image.fill((0,0,0))
        pygame.draw.circle(self.image, self.color, (radius, radius), radius)

        # POSE must always be regarded in the real frame        
        self.position = [self.real_size / 2, self.real_size / 2]
        self.yaw = 0.0
        self.linear_velocity = np.array([0.0,0.0])
        self.angular_velocity = 0.0

        self.collisions = 0

        pygame.draw.circle(self.image, (0,0,0), (radius + math.cos(self.yaw) * radius, radius - math.sin(self.yaw) * radius), radius / 3)

        self.rect = self.image.get_rect(center = tuple([self.position[0] * self.ratio, (self.real_size - self.position[1]) * self.ratio]))

        self.original_image = self.image

    def rotate(self):
        self.image = pygame.transform.rotate(self.original_image, math.degrees(self.yaw))
        self.rect = self.image.get_rect(center = tuple([self.position[0] * self.ratio, (self.real_size - self.position[1]) * self.ratio]))

    def move(self):
        self.rect.centerx = round(self.position[0] * self.ratio)
        self.rect.centery = round((self.real_size - self.position[1]) * self.ratio)

    def check_collisions(self, humans, walls):
        for human in humans:
            if (points_distance(human.position, self.position) < (human.radius + self.radius)):
                angle = vector_angle(points_unit_vector(self.position, human.position))
                self.position[0] = human.position[0] + math.cos(angle) * (human.radius + self.radius)
                self.position[1] = human.position[1] + math.sin(angle) * (human.radius + self.radius)
                self.move()
        for wall in walls:
            closest_point = wall.get_closest_point(self.position)
            if (points_distance(self.position, closest_point) < self.radius):
                angle = angle = vector_angle(points_unit_vector(self.position, closest_point))
                self.position[0] = closest_point[0] + math.cos(angle) * (self.radius)
                self.position[1] = closest_point[1] + math.sin(angle) * (self.radius)
                self.move()

    def move_with_keys(self, direction):    
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
        self.move()
        self.rotate()

    def render(self, display):
        display.blit(self.image, self.rect)

    def update(self, humans, walls):
        self.check_collisions(humans, walls)