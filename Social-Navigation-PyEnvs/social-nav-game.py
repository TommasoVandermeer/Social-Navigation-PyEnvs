import pygame
import sys
from src.motion_models import sfm_helbing, sfm_roboticsupo
from src.human_agent import HumanAgent
from src.robot_agent import RobotAgent
from src.obstacle import Obstacle
from src.utils import round_time
import numpy as np
import matplotlib.pyplot as plt
# from config.config import initialize
# from config.config_test1_integration import initialize
from config.config_test2_integration import initialize
# from config.config_test3_integration import initialize
# from config.config_test4_integration import initialize

### GLOBAL VARIABLES
WINDOW_SIZE = 700
DISPLAY_SIZE = 1000
REAL_SIZE = 15
MAX_FPS = 60
SAMPLING_TIME = 1 / MAX_FPS
### TEST VARIABLES
FINAL_T = 40
N_UPDATES = int(FINAL_T / SAMPLING_TIME)

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

        # Obstacles
        self.walls = pygame.sprite.Group()
        # Humans
        self.humans = []
        # Reset simulation to initialize all agent states
        self.reset()

    def reset(self):
        self.humans.clear()
        self.walls.empty()

        data = initialize()
        if "motion_model" in data.keys(): self.motion_model = data["motion_model"]
        else: self.motion_model = "sfm_helbing"
        if "runge_kutta" in data.keys(): self.runge_kutta = data["runge_kutta"]
        else: self.runge_kutta = False
        if "insert_robot" in data.keys(): self.insert_robot = data["insert_robot"]
        else: self.insert_robot = False
        if "grid" in data.keys(): self.grid = data["grid"]
        else: self.grid = True
        if "test" in data.keys(): self.test = data["test"]
        else: self.test = False
        
        for wall in data["walls"]:
            self.walls.add(Obstacle(self, wall))

        for key in data["humans"]:
            init_position = data["humans"][key]["pos"].copy()
            init_yaw = data["humans"][key]["yaw"]
            goals = data["humans"][key]["goals"].copy()
            if "color" in data["humans"][key]: color = data["humans"][key]["color"]
            else: color = (0,0,0)
            if "radius" in data["humans"][key]: radius = data["humans"][key]["radius"]
            else: radius = 0.3
            if "mass" in data["humans"][key]: mass = data["humans"][key]["mass"]
            else: mass = 75
            if "des_speed" in data["humans"][key]: des_speed = data["humans"][key]["des_speed"]
            else: des_speed = 0.9
            if "group_id" in data["humans"][key]: group_id = data["humans"][key]["group_id"]
            else: group_id = -1
            self.humans.append(HumanAgent(self, key, self.motion_model, init_position, init_yaw, goals, color, radius, mass, des_speed, group_id))
        
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

        # Robot
        if self.insert_robot: self.robot = RobotAgent(self)

        # Simulation variables
        self.n_updates = 0
        self.real_t = 0.0
        self.sim_t = 0.0
        self.last_reset = round_time(pygame.time.get_ticks() / 1000)

    def render(self):
        self.display.fill((255,255,255))
        if self.grid: self.display.blit(self.grid_surface, (0,0))

        for human in self.humans:
            human.render(self.display)
            human.render_label(self.display)
        if self.insert_robot: self.robot.render(self.display)
        self.walls.draw(self.display)

        self.fps_text = self.font.render(f"FPS: {round(self.clock.get_fps())}", False, (0,0,255))
        self.real_time = self.font.render(f"Real time: {self.real_t}", False, (0,0,255))
        self.simulation_time = self.font.render(f"Sim. time: {self.sim_t}", False, (0,0,255))
        self.real_time_factor = self.font.render(f"Time fact.: {round(self.sim_t/ (self.real_t + 0.00000001), 2)}", False, (0,0,255))
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
            if self.runge_kutta: sfm_roboticsupo.update_positions_RK45(self.humans, self.sim_t, SAMPLING_TIME)
            else: sfm_roboticsupo.update_positions(self.humans, SAMPLING_TIME)
        elif self.motion_model == "sfm_helbing":
            if self.insert_robot: sfm_helbing.compute_forces(self.humans, self.robot)
            else: sfm_helbing.compute_forces_no_robot(self.humans)
            if self.runge_kutta: sfm_helbing.update_positions_RK45(self.humans, self.sim_t, SAMPLING_TIME)
            else: sfm_helbing.update_positions(self.humans, SAMPLING_TIME)

        self.real_t = round_time((pygame.time.get_ticks() / 1000) - self.last_reset)
        self.sim_t = round_time(self.n_updates * SAMPLING_TIME)

        for human in self.humans:
            human.update(self.walls.sprites())

        if self.insert_robot: self.robot.update(self.humans, self.walls.sprites())

    def move_robot_with_keys(self):
        if pygame.key.get_pressed()[pygame.K_UP]: self.robot.move_with_keys('up')
        if pygame.key.get_pressed()[pygame.K_DOWN]: self.robot.move_with_keys('down')
        if pygame.key.get_pressed()[pygame.K_LEFT]: self.robot.move_with_keys('left')
        if pygame.key.get_pressed()[pygame.K_RIGHT]: self.robot.move_with_keys('right')
    
    def get_human_states(self):
        # State: [x, y, yaw, Vx, Vy, Omega]
        state = np.empty([len(self.humans),6],dtype=np.float64)
        for i in range(len(self.humans)):
            human_state = np.array([self.humans[i].position[0],self.humans[i].position[1],self.humans[i].yaw,self.humans[i].linear_velocity[0],self.humans[i].linear_velocity[1],self.humans[i].angular_velocity], dtype=np.float64)
            state[i] = human_state
        return state

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            if self.insert_robot: self.move_robot_with_keys()
            self.update()
            self.render()

            self.clock.tick(MAX_FPS)
    
    def run_test(self):
        ## Euler
        self.runge_kutta = False
        x_pose_humans_euler = {}
        y_pose_humans_euler = {}
        for step in range(N_UPDATES+1):
            if step != 0: self.update()
            for i in range(len(self.humans)):
                if step == 0:
                    x_pose_humans_euler[i] = [self.humans[i].position[0]]
                    y_pose_humans_euler[i] = [self.humans[i].position[1]]
                else:
                    x_pose_humans_euler[i].append(self.humans[i].position[0])
                    y_pose_humans_euler[i].append(self.humans[i].position[1])
        ## Runge kutta
        self.reset()
        self.runge_kutta = True
        x_pose_humans_rk45 = {}
        y_pose_humans_rk45 = {}
        for step in range(N_UPDATES+1):
            if step != 0: self.update()
            for i in range(len(self.humans)):
                if step == 0:
                    x_pose_humans_rk45[i] = [self.humans[i].position[0]]
                    y_pose_humans_rk45[i] = [self.humans[i].position[1]]
                else:
                    x_pose_humans_rk45[i].append(self.humans[i].position[0])
                    y_pose_humans_rk45[i].append(self.humans[i].position[1])
        ## Print
        figure, axs = plt.subplots(1, 2)
        figure.suptitle('Human agents position over simulation')
        axs[0].set_title('Euler integration')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].set_xlim([0,REAL_SIZE])
        axs[0].set_ylim([0,REAL_SIZE])
        for i in range(len(self.humans)):
            axs[0].plot(x_pose_humans_euler[i],y_pose_humans_euler[i])
        axs[1].set_title('Runge-Kutta 45 integration')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')
        axs[1].set_xlim([0,REAL_SIZE])
        axs[1].set_ylim([0,REAL_SIZE])
        for i in range(len(self.humans)):
            axs[1].plot(x_pose_humans_rk45[i],y_pose_humans_rk45[i])
        ## Display graphics
        plt.show()

simulator = SfmGame()
if not simulator.test: simulator.run()
else: simulator.run_test()