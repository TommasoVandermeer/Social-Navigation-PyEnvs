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

    ### Methods added for parallelization

    def get_safe_state(self):
        return np.array([*np.copy(self.position),self.yaw,*np.copy(self.linear_velocity),*np.copy(self.body_velocity),self.angular_velocity,
                         self.radius,self.mass,*self.goals[0],self.desired_speed], np.float64)
        
    def set_state(self, pose_and_velocity:np.ndarray):
        # [px,py,theta,vx,vy,bvx,bvy,omega]
        self.position = pose_and_velocity[0:2]
        self.yaw = pose_and_velocity[2]
        self.linear_velocity = pose_and_velocity[3:5]
        self.body_velocity = pose_and_velocity[5:7]
        self.angular_velocity = pose_and_velocity[7]

    def get_parameters(self, model:str):
        # Params array should be of the form: [relax_t,Ai,Aw,Bi,Bw,Ci,Cw,Di,Dw,Ei,k1,k2,a_lambda,gamma,ns,ns1,ko,kd,alpha,k_lambda] (length = 20)
        params = np.zeros((20,), np.float64)
        if (model == 'sfm_helbing'):
            params[0] = self.relaxation_time
            params[1] = self.Ai
            params[2] = self.Aw
            params[3] = self.Bi
            params[4] = self.Bw
            params[10] = self.k1
            params[11] = self.k2
        elif (model == 'sfm_guo'):
            params[0] = self.relaxation_time
            params[1] = self.Ai
            params[2] = self.Aw
            params[3] = self.Bi
            params[4] = self.Bw
            params[5] = self.Ci
            params[6] = self.Cw
            params[7] = self.Di
            params[8] = self.Dw
            params[10] = self.k1
            params[11] = self.k2
        elif (model == 'sfm_moussaid'):
            params[0] = self.relaxation_time
            params[9] = self.Ei
            params[12] = self.agent_lambda
            params[13] = self.gamma
            params[14] = self.ns
            params[15] = self.ns1
            params[2] = self.Aw
            params[4] = self.Bw
            params[10] = self.k1
            params[11] = self.k2
        elif (model == 'hsfm_farina'):
            params[0] = self.relaxation_time
            params[1] = self.Ai
            params[2] = self.Aw
            params[3] = self.Bi
            params[4] = self.Bw
            params[10] = self.k1
            params[11] = self.k2
            params[16] = self.ko
            params[17] = self.kd
            params[18] = self.alpha
            params[19] = self.k_lambda
        elif (model == 'hsfm_guo'):
            params[0] = self.relaxation_time
            params[1] = self.Ai
            params[2] = self.Aw
            params[3] = self.Bi
            params[4] = self.Bw
            params[5] = self.Ci
            params[6] = self.Cw
            params[7] = self.Di
            params[8] = self.Dw
            params[10] = self.k1
            params[11] = self.k2
            params[16] = self.ko
            params[17] = self.kd
            params[18] = self.alpha
            params[19] = self.k_lambda
        elif (model == 'hsfm_moussaid'):
            params[0] = self.relaxation_time
            params[9] = self.Ei
            params[12] = self.agent_lambda
            params[13] = self.gamma
            params[14] = self.ns
            params[15] = self.ns1
            params[2] = self.Aw
            params[4] = self.Bw
            params[10] = self.k1
            params[11] = self.k2
            params[16] = self.ko
            params[17] = self.kd
            params[18] = self.alpha
            params[19] = self.k_lambda
        elif (model == 'hsfm_new'):
            params[0] = self.relaxation_time
            params[1] = self.Ai
            params[2] = self.Aw
            params[3] = self.Bi
            params[4] = self.Bw
            params[10] = self.k1
            params[11] = self.k2
            params[16] = self.ko
            params[17] = self.kd
            params[18] = self.alpha
            params[19] = self.k_lambda
        elif (model == 'hsfm_new_guo'):
            params[0] = self.relaxation_time
            params[1] = self.Ai
            params[2] = self.Aw
            params[3] = self.Bi
            params[4] = self.Bw
            params[5] = self.Ci
            params[6] = self.Cw
            params[7] = self.Di
            params[8] = self.Dw
            params[10] = self.k1
            params[11] = self.k2
            params[16] = self.ko
            params[17] = self.kd
            params[18] = self.alpha
            params[19] = self.k_lambda
        elif (model == 'hsfm_new_moussaid'):
            params[0] = self.relaxation_time
            params[9] = self.Ei
            params[12] = self.agent_lambda
            params[13] = self.gamma
            params[14] = self.ns
            params[15] = self.ns1
            params[2] = self.Aw
            params[4] = self.Bw
            params[10] = self.k1
            params[11] = self.k2
            params[16] = self.ko
            params[17] = self.kd
            params[18] = self.alpha
            params[19] = self.k_lambda
        return params