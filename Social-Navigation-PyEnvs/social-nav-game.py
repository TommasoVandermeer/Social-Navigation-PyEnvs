import pygame
import sys
from src.motion_models import sfm_helbing, sfm_roboticsupo
from src.human_agent import HumanAgent
from src.robot_agent import RobotAgent
from src.obstacle import Obstacle
from src.utils import round_time
# from config.config import initialize
from config.config_circular_crossing_sfm_helbing_5_7m import initialize
# from config.config_circular_crossing_sfm_roboticsupo_14_4m import initialize
# from config.config_circular_crossing_sfm_helbing_14_4m import initialize
# from config.config_circular_crossing_rand_sfm_helbing_14_7m import initialize
# from config.config_circular_crossing_sfm_helbing_21_7m import initialize
# from config.config_circular_crossing_sfm_helbing_28_7m import initialize
# from config.config_circular_crossing_sfm_helbing_35_7m import initialize
# from config.config_circular_crossing_sfm_helbing_42_7m import initialize
# from config.config_circular_crossing_sfm_helbing_49_7m import initialize

WINDOW_SIZE = 700
DISPLAY_SIZE = 1000
REAL_SIZE = 15
MAX_FPS = 60
SAMPLING_TIME = 1 / MAX_FPS

class SfmGame:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((WINDOW_SIZE,WINDOW_SIZE))
        self.display = pygame.Surface((DISPLAY_SIZE, DISPLAY_SIZE))
        self.display_to_real_ratio = DISPLAY_SIZE / REAL_SIZE
        self.real_size = REAL_SIZE
        self.clock = pygame.time.Clock()

        self.font = pygame.font.Font('fonts/Roboto-Black.ttf', int(0.05 * WINDOW_SIZE))
        self.fps_text = self.font.render(f"FPS: {round(self.clock.get_fps())}", False, (0,0,255))
        self.fps_text_rect = self.fps_text.get_rect(topright = (DISPLAY_SIZE - DISPLAY_SIZE/30, DISPLAY_SIZE/60))
        self.x_axis_label = self.font.render("X", False, (0,0,255))
        self.x_axis_label_rect = self.x_axis_label.get_rect(center = (DISPLAY_SIZE - DISPLAY_SIZE/20, DISPLAY_SIZE - DISPLAY_SIZE/20))
        self.y_axis_label = self.font.render("Y", False, (0,0,255))
        self.y_axis_label_rect = self.y_axis_label.get_rect(center = (DISPLAY_SIZE/20, DISPLAY_SIZE/20))
        self.real_time = self.font.render(f"Real time: 0", False, (0,0,255))
        self.real_time_rect = self.real_time.get_rect(topright = ((DISPLAY_SIZE - DISPLAY_SIZE/12, DISPLAY_SIZE/20)))
        self.simulation_time = self.font.render(f"Sim. time: 0", False, (0,0,255))
        self.simulation_time_rect = self.real_time.get_rect(topright = ((DISPLAY_SIZE - DISPLAY_SIZE/11.5, DISPLAY_SIZE/12)))
        self.real_time_factor = self.font.render(f"Time fact.: 0", False, (0,0,255))
        self.real_time_factor_rect = self.real_time.get_rect(topright = ((DISPLAY_SIZE - DISPLAY_SIZE/13.25, DISPLAY_SIZE/8.5)))

        pygame.display.set_caption('Social Navigation')

        walls, humans, motion_model, insert_robot, grid = initialize()
        self.motion_model = motion_model
        self.insert_robot = insert_robot
        self.grid = grid

        # Grid
        if self.grid: 
            self.grid_lines = []
            for i in range(REAL_SIZE):
                self.grid_lines.append([[(i+1) * self.display_to_real_ratio, 0],[(i+1) * self.display_to_real_ratio, DISPLAY_SIZE]])
                self.grid_lines.append([[0, (i+1) * self.display_to_real_ratio],[DISPLAY_SIZE, (i+1) * self.display_to_real_ratio]])
            self.grid_surface = pygame.Surface((DISPLAY_SIZE,DISPLAY_SIZE))
            self.grid_surface.fill((255,255,255))
            for line in self.grid_lines:
                pygame.draw.aaline(self.grid_surface, (0,0,0), line[0], line[1])
            self.grid_surface.set_alpha(50)

        # Obstacles
        self.walls = pygame.sprite.Group()
        # Humans
        self.humans = []
        
        for wall in walls:
            self.walls.add(Obstacle(self, wall))

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
            self.humans.append(HumanAgent(self, key, self.motion_model, init_position, init_yaw, goals, color, radius, mass, des_speed, group_id))

        # Robot
        if self.insert_robot: self.robot = RobotAgent(self)

        # Simulation variables
        self.n_updates = 0
        self.real_t = 0.0
        self.sim_t = 0.0


    def render(self):
        self.display.fill((255,255,255))
        if self.grid: self.display.blit(self.grid_surface, (0,0))

        for human in self.humans:
            human.render(self.display)
            human.render_label(self.display)
        if self.insert_robot: self.robot.render(self.display)
        self.walls.draw(self.display)

        self.real_t = round_time(pygame.time.get_ticks() / 1000)
        self.sim_t = round_time(self.n_updates * SAMPLING_TIME)

        self.fps_text = self.font.render(f"FPS: {round(self.clock.get_fps())}", False, (0,0,255))
        self.real_time = self.font.render(f"Real time: {self.real_t}", False, (0,0,255))
        self.simulation_time = self.font.render(f"Sim. time: {self.sim_t}", False, (0,0,255))
        self.real_time_factor = self.font.render(f"Time fact.: {round(self.sim_t/ self.real_t, 2)}", False, (0,0,255))
        self.display.blit(self.fps_text,self.fps_text_rect)
        self.display.blit(self.real_time,self.real_time_rect)
        self.display.blit(self.simulation_time,self.simulation_time_rect)
        self.display.blit(self.real_time_factor,self.real_time_factor_rect)
        self.display.blit(self.x_axis_label,self.x_axis_label_rect)
        self.display.blit(self.y_axis_label,self.y_axis_label_rect)

        pygame.transform.scale(self.display, (WINDOW_SIZE, WINDOW_SIZE), self.screen)
        pygame.display.update()

    def update(self):
        self.n_updates += 1

        if self.motion_model == "sfm_roboticsupo":
            if self.insert_robot: sfm_roboticsupo.compute_forces(self.humans, self.robot)
            else: sfm_roboticsupo.compute_forces_no_robot(self.humans)
            sfm_roboticsupo.update_positions(self.humans, SAMPLING_TIME)
        elif self.motion_model == "sfm_helbing":
            if self.insert_robot: sfm_helbing.compute_forces(self.humans, self.robot)
            else: sfm_helbing.compute_forces_no_robot(self.humans)
            sfm_helbing.update_positions(self.humans, SAMPLING_TIME)

        for human in self.humans:
            human.update(self.walls.sprites())

        if self.insert_robot: self.robot.update(self.humans, self.walls.sprites())

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

            if self.insert_robot: self.move_robot()
            self.update()
            self.render()

            self.clock.tick(MAX_FPS)

SfmGame().run()