import pygame
from src.utils import points_distance

class Obstacle(pygame.sprite.Sprite):
    def __init__(self, game, bottom_left:list[float], bottom_right:list[float], top_right:list[float], top_left:list[float]):
        super().__init__()

        self.ratio = game.display_to_real_ratio
        self.real_size = game.real_size

        # Point must be given in order and in the real frame
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
        self.top_right = top_right
        self.top_left = top_left

        ## Only if walls are not inclined
        # if ((bottom_left[0] != bottom_right[0]) and (bottom_left[1] != bottom_right[1])):
        #     raise Exception("Obstacle points given in the wrong order")
        # if ((bottom_right[0] != top_right[0]) and (bottom_right[1] != top_right[1])):
        #     raise Exception("Obstacle points given in the wrong order")
        # if ((top_right[0] != top_left[0]) and (top_right[1] != top_left[1])):
        #     raise Exception("Obstacle points given in the wrong order")
        # if ((top_left[0] != bottom_left[0]) and (top_left[1] != bottom_left[1])):
        #     raise Exception("Obstacle points given in the wrong order")
        
        self.segments = [[self.bottom_left,self.bottom_right],[self.bottom_right,self.top_right],[self.top_right,self.top_left],[self.top_left,self.bottom_left]]

        self.x_length = points_distance(self.bottom_left,self.bottom_right)
        self.y_length = points_distance(self.bottom_right,self.top_right)

        self.image = pygame.Surface((self.x_length * self.ratio, self.y_length * self.ratio))
        self.image.fill((0,0,0))

        self.rect = self.image.get_rect(bottomleft = (self.bottom_left[0] * self.ratio, (self.real_size - self.bottom_left[1]) * self.ratio))

    def get_rect(self):
        return self.rect

    def get_segments(self):
        return self.segments
    
    def get_closest_point(self, point):
        min_dist = 10000
        closest_point = [0,0]
        for line in self.segments:
            a = min(line[0],line[1])
            b = max(line[0],line[1])
            t = ((point[0] - a[0]) * (b[0] - a[0]) + (point[1] - a[1]) * (b[1] - a[1])) / ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
            t_star = min(max(0, t), 1)
            h = [a[0] + t_star * (b[0] - a[0]), a[1] + t_star * (b[1] - a[1])]
            dist = points_distance(point, h)
            if dist <= min_dist:
                closest_point = h
                min_dist = dist
        return closest_point