import pygame
from src.motion_model_manager import MotionModelManager
from src.human_agent import HumanAgent
from src.robot_agent import RobotAgent
from src.obstacle import Obstacle
from src.utils import round_time, bound_angle
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

### GLOBAL VARIABLES
WINDOW_SIZE = 700
DISPLAY_SIZE = 1000
REAL_SIZE = 15
MAX_FPS = 60
SAMPLING_TIME = 1 / MAX_FPS
MOTION_MODELS = ["sfm_roboticsupo","sfm_helbing","sfm_guo","sfm_moussaid","hsfm_farina","hsfm_guo",
                 "hsfm_moussaid","hsfm_new","hsfm_new_guo","hsfm_new_moussaid"]
COLORS = list(mcolors.TABLEAU_COLORS.values())

class SocialNav:
    def __init__(self, config_data, mode="custom_config"):
        pygame.init()
        self.pygame_init = True

        self.display_to_real_ratio = DISPLAY_SIZE / REAL_SIZE
        self.real_size = REAL_SIZE
        self.clock = pygame.time.Clock()
        self.walls = pygame.sprite.Group()
        self.humans = []
        self.mode = mode

        if mode == "custom_config": self.config_data = config_data
        elif mode == "circular_crossing": self.config_data = self.generate_circular_crossing_setting(config_data)
        else: raise Exception(f"Mode '{mode}' does not exist")

        self.reset(restart_gui=True)

    def init_gui(self):
        self.screen = pygame.display.set_mode((WINDOW_SIZE,WINDOW_SIZE))
        self.display = pygame.Surface((DISPLAY_SIZE, DISPLAY_SIZE))
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
        self.pause_text = self.font.render("PAUSED", False, ((0,0,255)))
        self.pause_text_rect = self.pause_text.get_rect(center = (DISPLAY_SIZE/2, DISPLAY_SIZE/20))
        pygame.display.set_caption('Social Navigation')

    def reset(self, restart_gui=False):
        if not self.pygame_init: pygame.init()
        self.humans.clear()
        self.walls.empty()

        if "headless" in self.config_data.keys(): self.headless = self.config_data["headless"]
        else: self.headless = False
        if "motion_model" in self.config_data.keys(): self.motion_model = self.config_data["motion_model"]
        else: self.motion_model = "sfm_helbing"
        if "runge_kutta" in self.config_data.keys(): self.runge_kutta = self.config_data["runge_kutta"]
        else: self.runge_kutta = False
        if "insert_robot" in self.config_data.keys(): self.insert_robot = self.config_data["insert_robot"]
        else: self.insert_robot = False
        if "grid" in self.config_data.keys(): self.grid = self.config_data["grid"]
        else: self.grid = True
        
        if not self.headless and restart_gui: self.init_gui()

        for wall in self.config_data["walls"]:
            self.walls.add(Obstacle(self, wall))

        for key in self.config_data["humans"]:
            init_position = self.config_data["humans"][key]["pos"].copy()
            init_yaw = self.config_data["humans"][key]["yaw"]
            goals = self.config_data["humans"][key]["goals"].copy()
            if "color" in self.config_data["humans"][key]: color = self.config_data["humans"][key]["color"]
            else: color = (0,0,0)
            if "radius" in self.config_data["humans"][key]: radius = self.config_data["humans"][key]["radius"]
            else: radius = 0.3
            if "mass" in self.config_data["humans"][key]: mass = self.config_data["humans"][key]["mass"]
            else: mass = 75
            if "des_speed" in self.config_data["humans"][key]: des_speed = self.config_data["humans"][key]["des_speed"]
            else: des_speed = 0.9
            if "group_id" in self.config_data["humans"][key]: group_id = self.config_data["humans"][key]["group_id"]
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
        self.robot = RobotAgent(self)

        # Human motion model
        self.motion_model_manager = MotionModelManager(self.motion_model, self.insert_robot, self.runge_kutta, self.humans, self.robot)

        # Simulation variables
        self.n_updates = 0
        self.real_t = 0.0
        self.sim_t = 0.0
        self.last_reset = round_time(pygame.time.get_ticks() / 1000)
        self.last_pause_start = 0.0
        self.paused_time = 0.0
        self.paused = False

    def generate_circular_crossing_setting(self, config_data:list):
        radius = config_data[0]
        n_actors = config_data[1]
        rand = config_data[2]
        model = config_data[3]
        headless = config_data[4]
        runge_kutta = config_data[5]
        insert_robot = config_data[6]
        center = np.array([self.real_size/2,self.real_size/2],dtype=np.float64)
        humans = {}
        if not rand:
            arch = (2 * math.pi) / (n_actors)
            for i in range(n_actors):
                center_pos = [radius * math.cos(arch * i), radius * math.sin(arch * i)]
                humans[i] = {"pos": [center[0] + center_pos[0], center[1] + center_pos[1]],
                             "yaw": bound_angle(-math.pi + arch * i),
                             "goals": [[center[0] - center_pos[0], center[1] - center_pos[1]], [center[0] + center_pos[0], center[1] + center_pos[1]]]}
        else:
            pass
        data = {"motion_model": model, "headless": headless, "runge_kutta": runge_kutta, "insert_robot": insert_robot, "grid": True, "walls": [], "humans": humans}
        return data

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

        if self.paused: self.display.blit(self.pause_text, self.pause_text_rect)

        pygame.transform.scale(self.display, (WINDOW_SIZE, WINDOW_SIZE), self.screen)
        pygame.display.update()

    def update(self):
        self.n_updates += 1

        self.motion_model_manager.update(self.sim_t, SAMPLING_TIME)

        self.real_t = round_time((pygame.time.get_ticks() / 1000) - self.last_reset - self.paused_time)
        self.sim_t = round_time(self.n_updates * SAMPLING_TIME)

        for human in self.humans: human.update(self.walls.sprites())

        if self.insert_robot: self.robot.update(self.humans, self.walls.sprites())

    def move_robot_with_keys(self):
        if pygame.key.get_pressed()[pygame.K_UP]: self.robot.move_with_keys('up')
        if pygame.key.get_pressed()[pygame.K_DOWN]: self.robot.move_with_keys('down')
        if pygame.key.get_pressed()[pygame.K_LEFT]: self.robot.move_with_keys('left')
        if pygame.key.get_pressed()[pygame.K_RIGHT]: self.robot.move_with_keys('right')

    def run(self):
        self.active = True
        self.human_states = np.array([self.motion_model_manager.get_human_states()], dtype=np.float64)
        while self.active:
            if not self.paused:
                if self.insert_robot: self.move_robot_with_keys()
                self.update()
                if not self.headless: self.render()
                self.human_states = np.append(self.human_states, [self.motion_model_manager.get_human_states()], axis=0)

            else:
                if not self.headless: self.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.active = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.paused = not self.paused
                    if self.paused: self.last_pause_start = round_time(pygame.time.get_ticks() / 1000)
                    else: self.paused_time += round_time((pygame.time.get_ticks() / 1000) - self.last_pause_start)

            self.clock.tick(MAX_FPS)
    
    def run_k_steps(self, steps, quit=True):
        human_states = np.empty((steps,len(self.humans),6), dtype=np.float64)
        for step in range(steps):
            self.update()
            if not self.headless: self.render()
            human_states[step] = self.motion_model_manager.get_human_states()
        if not self.headless and quit: pygame.quit(); self.pygame_init = False
        return human_states

    def run_single_test(self, n_updates):
        start_time = round_time((pygame.time.get_ticks() / 1000))
        human_states = np.append([self.motion_model_manager.get_human_states()], self.run_k_steps(n_updates, quit=False), axis=0)
        test_time = round_time((pygame.time.get_ticks() / 1000) - start_time - self.paused_time)
        return human_states, test_time

    def run_multiple_models_test(self, final_time=40, models=MOTION_MODELS, plot_sample_time=3, two_integrations=False):
        n_updates = int(final_time / SAMPLING_TIME)
        if not two_integrations:
            self.human_states = np.empty((len(models),n_updates+1,len(self.humans),6), dtype=np.float64)
            test_times = np.empty((len(models),), dtype=np.float64)
            for i in range(len(models)):
                self.reset()
                self.motion_model_manager.set_motion_model(models[i])
                for human in self.humans: human.set_parameters(models[i])
                self.human_states[i], test_times[i] = self.run_single_test(n_updates)
                figure, ax = plt.subplots()
                figure.suptitle(f'Human agents\' position over simulation | T = {final_time} | dt = {round(SAMPLING_TIME, 4)} | Model = {models[i]}')
                if self.motion_model_manager.runge_kutta == False: integration_title = "Euler"
                else: integration_title = "Runge-Kutta-45"
                ax.set(xlabel='X',ylabel='Y',title=f'{integration_title} | Elapsed time = {test_times[i]}',xlim=[0,REAL_SIZE],ylim=[0,REAL_SIZE])
                self.plot_agents_position(ax,self.human_states[i],plot_sample_time,models[i])
        else:
            self.human_states = np.empty((len(models),2,n_updates+1,len(self.humans),6), dtype=np.float64)
            test_times = np.empty((len(models),2), dtype=np.float64)
            for i in range(len(models)):
                self.reset()
                self.motion_model_manager.set_motion_model(models[i])
                self.motion_model_manager.runge_kutta = False
                for human in self.humans: human.set_parameters(models[i])
                self.human_states[i,0], test_times[i,0] = self.run_single_test(n_updates)
                self.reset()
                self.motion_model_manager.set_motion_model(models[i])
                self.motion_model_manager.runge_kutta = True
                for human in self.humans: human.set_parameters(models[i])
                self.human_states[i,1], test_times[i,1] = self.run_single_test(n_updates)
                figure, ax = plt.subplots(1,2)
                figure.suptitle(f'Humans\' position over simulation | T = {final_time} | dt = {round(SAMPLING_TIME, 4)} | Model = {models[i]}')
                ax[0].set(xlabel='X',ylabel='Y',title=f'Euler | Elapsed time = {test_times[i,0]}',xlim=[0,REAL_SIZE],ylim=[0,REAL_SIZE])
                self.plot_agents_position(ax[0],self.human_states[i,0],plot_sample_time,models[i])
                ax[1].set(xlabel='X',ylabel='Y',title=f'Runge-Kutta-45 | Elapsed time = {test_times[i,1]}',xlim=[0,REAL_SIZE],ylim=[0,REAL_SIZE])
                self.plot_agents_position(ax[1],self.human_states[i,1],plot_sample_time,models[i])

        if not self.headless: pygame.quit(); self.pygame_init = False
        plt.show()

    def run_integration_test(self, final_time=40):
        n_updates = int(final_time / SAMPLING_TIME)
        self.human_states = np.empty((2,n_updates+1,len(self.humans),6), dtype=np.float64)
        ## Euler
        self.motion_model_manager.runge_kutta = False
        self.human_states[0], euler_time = self.run_single_test(n_updates)
        ## Runge kutta
        self.reset()
        self.motion_model_manager.runge_kutta = True
        self.human_states[1], rk45_time = self.run_single_test(n_updates)
        if not self.headless: pygame.quit(); self.pygame_init = False
        ## Print
        figure, axs = plt.subplots(1, 2)
        figure.suptitle(f'Human agents\' position over simulation | T = {final_time} | dt = {round(SAMPLING_TIME, 4)} | motion model: {self.motion_model}')
        axs[0].set_title(f'Euler - Elapsed time: {euler_time}')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        axs[0].set_xlim([0,REAL_SIZE])
        axs[0].set_ylim([0,REAL_SIZE])
        self.print_walls_on_plot(axs[0])
        for i in range(len(self.humans)):
            axs[0].plot(self.human_states[0,:,i,0],self.human_states[0,:,i,1])
        axs[1].set_title(f'Runge-Kutta 45 - Elapsed time: {rk45_time}')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Y')
        axs[1].set_xlim([0,REAL_SIZE])
        axs[1].set_ylim([0,REAL_SIZE])
        self.print_walls_on_plot(axs[1])
        for i in range(len(self.humans)):
            axs[1].plot(self.human_states[1,:,i,0],self.human_states[1,:,i,1])
        ## Display graphics
        plt.show()

    def print_walls_on_plot(self, ax):
        for i in range(len(self.walls)):
                ax.fill(self.walls.sprites()[i].vertices[:,0], self.walls.sprites()[i].vertices[:,1], facecolor='black', edgecolor='black')

    def plot_agents_position(self, ax, human_states, plot_sample_time, model):
        ax.axis('equal')
        self.print_walls_on_plot(ax)
        for j in range(len(self.humans)):
            color_idx = j % len(COLORS)
            ax.plot(human_states[:,j,0],human_states[:,j,1], color=COLORS[color_idx], linewidth=0.5, zorder=0)
            for k in range(0,len(human_states),int(plot_sample_time / SAMPLING_TIME)):
                if "hsfm" in model:
                    head = plt.Circle((human_states[k,j,0] + math.cos(human_states[k,j,2]) * self.humans[j].radius, human_states[k,j,1] + math.sin(human_states[k,j,2]) * self.humans[j].radius), 0.1, color=COLORS[color_idx], zorder=1)
                    ax.add_patch(head)
                circle = plt.Circle((human_states[k,j,0],human_states[k,j,1]),self.humans[j].radius, edgecolor=COLORS[color_idx], facecolor="white", fill=True, zorder=1)
                ax.add_patch(circle)
                ax.text(human_states[k,j,0],human_states[k,j,1], f"{k*SAMPLING_TIME}", color=COLORS[color_idx], va="center", ha="center", fontsize="x-small", zorder=1)
            goals = np.array(self.humans[j].goals, dtype=np.float64).copy()
            for k in range(len(goals)):
                if goals[k,0] == human_states[0,j,0] and goals[k,1] == human_states[0,j,1]: 
                    goals = np.delete(goals, k, 0)
                    break
            ax.scatter(goals[:,0], goals[:,1], marker="*", color=COLORS[color_idx], zorder=2)