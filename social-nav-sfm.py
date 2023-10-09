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

        pygame.display.set_caption('Social Navigation SFM')

        # Humans
        self.humans = pygame.sprite.Group()
        self.humans.add(HumanAgent(self, self.motion_model, [REAL_SIZE/1.5,REAL_SIZE/1.5], -math.pi))
        self.humans.add(HumanAgent(self, self.motion_model, [REAL_SIZE/4,REAL_SIZE/4], 0.0))

        # Robot
        self.robot = RobotAgent(self)

        # Obstacles
        self.walls = pygame.sprite.Group()
        self.walls.add(Obstacle(self, [1,1], [1.5,1], [1.5,3], [1,3]))

    def render(self):
        self.display.fill((255,255,255))
        self.humans.draw(self.display)
        self.robot.render(self.display)
        self.walls.draw(self.display)

        pygame.transform.scale(self.display, (WINDOW_SIZE, WINDOW_SIZE), self.screen)
        pygame.display.update()

    def update(self):
        self.humans.update()
        self.robot.update()

    def get_entities(self):
        humans = []
        walls = []
        for human in self.humans:
            humans.append(human)
        for wall in self.walls:
            walls.append(wall)
        return humans, walls

    def move_robot(self):
        humans, walls = self.get_entities()
        if pygame.key.get_pressed()[pygame.K_UP]: self.robot.move_with_keys('up', humans, walls)
        if pygame.key.get_pressed()[pygame.K_DOWN]: self.robot.move_with_keys('down', humans, walls)
        if pygame.key.get_pressed()[pygame.K_LEFT]: self.robot.move_with_keys('left', humans, walls)
        if pygame.key.get_pressed()[pygame.K_RIGHT]: self.robot.move_with_keys('right', humans, walls)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.move_robot()
            self.update()
            self.render()

            #print(self.clock.get_fps())
            self.clock.tick(MAX_FPS)

SfmGame().run()