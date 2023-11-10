import pygame
import numpy as np

BIG_INT = 10000

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, game, vertices:list[list[float]]):
        super().__init__()

        if len(vertices) < 3:
            raise Exception("Obstacle has to have at least 3 vertices") 

        self.ratio = game.display_to_real_ratio
        self.real_size = game.real_size

        self.vertices = np.array(vertices, dtype=np.float64)
        self.segments = {}
        self.coordinates = []

        min_x = BIG_INT
        max_x = -BIG_INT
        min_y = BIG_INT
        max_y = -BIG_INT

        for i in range(len(vertices)):
            if vertices[i][0] < min_x: min_x = vertices[i][0]
            if vertices[i][0] > max_x: max_x = vertices[i][0]
            if vertices[i][1] < min_y: min_y = vertices[i][1]
            if vertices[i][1] > max_y: max_y = vertices[i][1]
            if i < len(vertices)-1: self.segments[i] = [vertices[i], vertices[i+1]]
            else: self.segments[i] = [vertices[i], vertices[0]]

        self.bounding_box_x_length = max_x - min_x
        self.bounding_box_y_length = max_y - min_y

        for i in range(len(vertices)):
            self.coordinates.append((int((vertices[i][0] - min_x) * self.ratio), int((self.bounding_box_y_length - (vertices[i][1] - min_y)) * self.ratio)))

        self.image = pygame.Surface((self.bounding_box_x_length * self.ratio, self.bounding_box_y_length * self.ratio), pygame.SRCALPHA)
        pygame.draw.polygon(self.image, (0,0,0), self.coordinates)
        self.rect = self.image.get_rect(bottomleft = (min_x * self.ratio, (self.real_size - min_y) * self.ratio))

    def render(self, display:pygame.Surface, scroll:np.array):
        display.blit(self.image, (self.rect.x - scroll[0], self.rect.y - scroll[1]))

    def get_rect(self):
        return self.rect

    def get_segments(self):
        return self.segments
    
    def get_closest_point(self, point:np.array):
        min_distance = 10000
        closest_point = np.array([0.0,0.0], dtype=np.float64)
        for key in self.segments:
            a = np.array(min(self.segments[key][0],self.segments[key][1]), dtype=np.float64)
            b = np.array(max(self.segments[key][0],self.segments[key][1]), dtype=np.float64)
            t = (np.dot(point - a, b - a)) / (np.linalg.norm(b - a) ** 2)
            t_star = min(max(0, t), 1)
            h = a + t_star * (b - a)
            distance = np.linalg.norm(h - point)
            if distance <= min_distance:
                closest_point = h
                min_distance = distance
        return closest_point, min_distance