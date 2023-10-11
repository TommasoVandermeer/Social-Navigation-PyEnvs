import pygame
import sys
from random import randint
from src.human_agent import HumanAgent
from src.robot_agent import RobotAgent
from src.obstacle import Obstacle
from src import sfm
import math

WINDOW_SIZE = 600
DISPLAY_SIZE = 1000
REAL_SIZE = 10
MAX_FPS = 60

class SfmGame:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((WINDOW_SIZE,WINDOW_SIZE))
        self.display = pygame.Surface((DISPLAY_SIZE, DISPLAY_SIZE))
        self.display_to_real_ratio = DISPLAY_SIZE / REAL_SIZE
        self.real_size = REAL_SIZE
        self.clock = pygame.time.Clock()
        self.game_caption = 'Social Navigation SFM'
        self.motion_model = 'sfm'

        self.font = pygame.font.Font('fonts/Roboto-Black.ttf', 50)
        self.fps_text = self.font.render(f"FPS: {round(self.clock.get_fps())}", False, (0,0,0))
        self.fps_text_rect = self.fps_text.get_rect(topleft = (DISPLAY_SIZE - DISPLAY_SIZE/4, DISPLAY_SIZE/30))

        pygame.display.set_caption('Social Navigation SFM')

        # Obstacles
        self.walls = pygame.sprite.Group()
        self.walls.add(Obstacle(self, [1,1], [1.5,1], [1.5,3], [1,3]))
        self.walls.add(Obstacle(self, [3,9], [6,9], [6,9.5], [3,9.5]))

        # Humans
        self.humans = pygame.sprite.Group()
        self.humans.add(HumanAgent(self, self.motion_model, [REAL_SIZE/1.5,REAL_SIZE/1.5], -math.pi, [[5,5],[8,2]]))
        self.humans.add(HumanAgent(self, self.motion_model, [REAL_SIZE/4,REAL_SIZE/4], 0.0, [[5,5],[8,2]]))
        self.humans.add(HumanAgent(self, self.motion_model, [REAL_SIZE/5,REAL_SIZE/2], -math.pi, [[3,7],[8,8]], group_id=1))
        self.humans.add(HumanAgent(self, self.motion_model, [REAL_SIZE/5,REAL_SIZE/3], 0.0, [[3,7],[8,8]], group_id=1))

        # Robot
        self.robot = RobotAgent(self)

    def render(self):
        self.display.fill((255,255,255))
        self.humans.draw(self.display)
        self.robot.render(self.display)
        self.walls.draw(self.display)

        self.display.blit(self.fps_text,self.fps_text_rect)

        pygame.transform.scale(self.display, (WINDOW_SIZE, WINDOW_SIZE), self.screen)
        pygame.display.update()

    def update(self):
        sfm.compute_forces(self.humans.sprites(), self.robot)
        sfm.update_positions(self.humans.sprites(), 1/MAX_FPS)

        humans, walls = self.get_entities()

        self.humans.update(walls)
        self.robot.update(humans, walls)
        self.fps_text = self.font.render(f"FPS: {round(self.clock.get_fps())}", False, (0,0,0))

    def get_entities(self):
        humans = []
        walls = []
        for human in self.humans:
            humans.append(human)
        for wall in self.walls:
            walls.append(wall)
        return humans, walls

    def move_robot(self):
        if pygame.key.get_pressed()[pygame.K_UP]: self.robot.move_with_keys('up')
        if pygame.key.get_pressed()[pygame.K_DOWN]: self.robot.move_with_keys('down')
        if pygame.key.get_pressed()[pygame.K_LEFT]: self.robot.move_with_keys('left')
        if pygame.key.get_pressed()[pygame.K_RIGHT]: self.robot.move_with_keys('right')

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.move_robot()
            self.update()
            self.render()

            self.clock.tick(MAX_FPS)

SfmGame().run()