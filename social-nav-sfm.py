import pygame
import sys
from random import randint
from src.human_agent import HumanAgent
from src.robot_agent import RobotAgent
from src.obstacle import Obstacle
from config.config_sfm_roboticsupo import initialize
from src import sfm_roboticsupo
import math
from timeit import default_timer as timer

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

        self.font = pygame.font.Font('fonts/Roboto-Black.ttf', 50)
        self.fps_text = self.font.render(f"FPS: {round(self.clock.get_fps())}", False, (0,0,0))
        self.fps_text_rect = self.fps_text.get_rect(topleft = (DISPLAY_SIZE - DISPLAY_SIZE/4, DISPLAY_SIZE/30))

        pygame.display.set_caption('Social Navigation')

        walls, humans, motion_model = initialize()
        self.motion_model = motion_model
        # Obstacles
        self.walls = pygame.sprite.Group()
        # Humans
        self.humans = pygame.sprite.Group()
        
        for wall in walls:
            self.walls.add(Obstacle(self, wall[0], wall[1], wall[2], wall[3]))

        for key in humans:
            init_position = humans[key]["pos"]
            init_yaw = humans[key]["yaw"]
            goals = humans[key]["goals"]
            if "color" in humans[key]: color = humans[key]["color"]
            else: color = (0,0,0)
            if "radius" in humans[key]: radius = humans[key]["radius"]
            else: radius = 0.3
            if "mass" in humans[key]: mass = humans[key]["mass"]
            else: mass = 75
            if "des_speed" in humans[key]: des_speed = humans[key]["des_speed"]
            else: des_speed = 0.9
            if "group_id" in humans[key]: group_id = humans[key]["group_id"]
            else: group_id = -1
            self.humans.add(HumanAgent(self, key, self.motion_model, init_position, init_yaw, goals, color, radius, mass, des_speed, group_id))

        # Robot
        self.robot = RobotAgent(self)

        # Update time
        self.last_update = timer()

    def render(self):
        self.display.fill((255,255,255))
        self.humans.draw(self.display)
        self.robot.render(self.display)
        self.walls.draw(self.display)

        for human in self.humans.sprites():
            human.render_label(self.display)

        self.display.blit(self.fps_text,self.fps_text_rect)

        pygame.transform.scale(self.display, (WINDOW_SIZE, WINDOW_SIZE), self.screen)
        pygame.display.update()

    def update(self):
        sfm_roboticsupo.compute_forces(self.humans.sprites(), self.robot)
        sfm_roboticsupo.update_positions(self.humans.sprites(), timer() - self.last_update)
        self.last_update = timer()

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