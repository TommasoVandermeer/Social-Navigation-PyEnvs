import pygame
import math

class HumanAgent(pygame.sprite.Sprite):
    def __init__(self, game, model:str, pos:list[float], yaw:float, color=(0,0,0), radius=0.3, mass=75, des_speed=0.9, group_id=-1):
        super().__init__()
        self.ratio = game.display_to_real_ratio
        self.real_size = game.real_size
        self.color = color

        self.motion_model = model
        self.radius = radius 
        radius = self.radius * self.ratio
        self.desired_speed = des_speed
        self.group_id = -1

        if (self.motion_model == 'sfm'):
            self.goal_weight = 2.0
            self.obstacle_weight = 10.0
            self.social_weight = 15.0
            self.group_gaze_weight = 0.0
            self.group_coh_weight = 0.0
            self.group_rep_weight = 0.0
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

        # POSE must always be regarded in the real frame  
        self.position = pos
        self.yaw = yaw

        pygame.draw.circle(self.image, (0,0,255), (radius + math.cos(self.yaw) * radius, radius - math.sin(self.yaw) * radius), radius / 3)

        self.rect = self.image.get_rect(center = tuple([self.position[0] * self.ratio, (self.real_size - self.position[1]) * self.ratio]))

        self.original_image = self.image

    def get_rect(self):
        return self.rect

    def update(self):
        pass