import pygame
from social_gym.src.motion_model_manager import MotionModelManager, N_GENERAL_STATES, N_HEADED_STATES, N_NOT_HEADED_STATES
from social_gym.src.human_agent import HumanAgent
from social_gym.src.robot_agent import RobotAgent
from social_gym.src.obstacle import Obstacle
from social_gym.src.utils import round_time, bound_angle
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

### GLOBAL VARIABLES
WINDOW_SIZE = 700
DISPLAY_SIZE = 1000
REAL_SIZE = 15
MAX_FPS = 60
SAMPLING_TIME = 1 / MAX_FPS
N_UPDATES_AVERAGE_TIME = 20
MOTION_MODELS = ["sfm_roboticsupo","sfm_helbing","sfm_guo","sfm_moussaid","hsfm_farina","hsfm_guo",
                 "hsfm_moussaid","hsfm_new","hsfm_new_guo","hsfm_new_moussaid"]
COLORS = list(mcolors.TABLEAU_COLORS.values())
ZOOM_BOUNDS = [0.5, 2]
SCROLL_BOUNDS = [-500,500]

class SocialNavSim:
    def __init__(self, config_data, scenario="custom_config"):
        pygame.init()
        self.pygame_init = True

        self.display_to_real_ratio = DISPLAY_SIZE / REAL_SIZE
        self.display_to_window_ratio = DISPLAY_SIZE / WINDOW_SIZE
        self.real_size = REAL_SIZE
        self.clock = pygame.time.Clock()
        self.walls = pygame.sprite.Group()
        self.humans = []
        self.mode = scenario

        if scenario == "custom_config": self.config_data = config_data
        elif scenario == "circular_crossing": self.config_data = self.generate_circular_crossing_setting(config_data)
        else: raise Exception(f"Scenario '{scenario}' does not exist")

        self.reset_sim(restart_gui=True)

    def init_gui(self):
        self.screen = pygame.display.set_mode((WINDOW_SIZE,WINDOW_SIZE))
        self.display = pygame.Surface((DISPLAY_SIZE, DISPLAY_SIZE))
        self.font = pygame.font.Font(os.path.join(os.path.dirname(__file__),'fonts/Roboto-Black.ttf'), int(0.035 * WINDOW_SIZE))
        self.small_font = pygame.font.Font(os.path.join(os.path.dirname(__file__),'fonts/Roboto-Black.ttf'), int(0.0245 * WINDOW_SIZE))
        self.fps_text = self.font.render(f"FPS: {round(self.clock.get_fps())}", False, (0,0,255))
        self.fps_text_rect = self.fps_text.get_rect(topright = (WINDOW_SIZE - WINDOW_SIZE/30, WINDOW_SIZE/60))
        self.x_axis_label = self.font.render("X", False, (0,0,255))
        self.x_axis_label_rect = self.x_axis_label.get_rect(center = (WINDOW_SIZE - WINDOW_SIZE/20, WINDOW_SIZE - WINDOW_SIZE/20))
        self.y_axis_label = self.font.render("Y", False, (0,0,255))
        self.y_axis_label_rect = self.y_axis_label.get_rect(center = (WINDOW_SIZE/20, WINDOW_SIZE/20))
        self.real_time = self.font.render(f"Real time: 0", False, (0,0,255))
        self.real_time_rect = self.real_time.get_rect(topright = ((WINDOW_SIZE - WINDOW_SIZE/12, WINDOW_SIZE/20)))
        self.simulation_time = self.font.render(f"Sim. time: 0", False, (0,0,255))
        self.simulation_time_rect = self.real_time.get_rect(topright = ((WINDOW_SIZE - WINDOW_SIZE/11.5, WINDOW_SIZE/12)))
        self.real_time_factor = self.font.render(f"Time fact.: 0", False, (0,0,255))
        self.real_time_factor_rect = self.real_time.get_rect(topright = ((WINDOW_SIZE - WINDOW_SIZE/13.25, WINDOW_SIZE/8.5)))
        self.pause_text = self.font.render("PAUSED", False, ((0,0,255)))
        self.pause_text_rect = self.pause_text.get_rect(center = (WINDOW_SIZE/2, WINDOW_SIZE/20))
        self.rewind_text = self.small_font.render("PRESS Z TO REWIND", False, ((0,0,255)))
        self.rewind_text_rect = self.rewind_text.get_rect(bottomleft = (WINDOW_SIZE/25, WINDOW_SIZE - WINDOW_SIZE/10))
        self.reset_text = self.small_font.render("PRESS R TO RESET", False, ((0,0,255)))
        self.reset_text_rect = self.reset_text.get_rect(bottomright = (WINDOW_SIZE - WINDOW_SIZE/25, WINDOW_SIZE - WINDOW_SIZE/10))
        self.speedup_text = self.small_font.render("PRESS S TO SPEED UP", False, ((0,0,255)))
        self.speedup_text_rect = self.reset_text.get_rect(midbottom = (WINDOW_SIZE/2, WINDOW_SIZE - WINDOW_SIZE/10))
        pygame.display.set_caption('Social Navigation')

    def set_time_step(self, time_step:float):
        global SAMPLING_TIME, MAX_FPS
        SAMPLING_TIME = time_step
        MAX_FPS = 1 / SAMPLING_TIME
        self.robot.time_step = time_step
        self.robot.policy.time_step = time_step

    def reset_sim(self, restart_gui=False, reset_robot=True):
        if not self.pygame_init: pygame.init(); self.pygame_init = True
        self.humans.clear()
        self.walls.empty()

        if "headless" in self.config_data.keys(): self.headless = self.config_data["headless"]
        else: self.headless = False
        if "motion_model" in self.config_data.keys(): self.motion_model = self.config_data["motion_model"]
        else: self.motion_model = "sfm_helbing"
        if "runge_kutta" in self.config_data.keys(): self.runge_kutta = self.config_data["runge_kutta"]
        else: self.runge_kutta = False
        if "grid" in self.config_data.keys(): self.grid = self.config_data["grid"]
        else: self.grid = True
        if "robot_visible" in self.config_data.keys(): self.robot_visible = self.config_data["robot_visible"]
        else: self.robot_visible = False

        if not self.headless and restart_gui: self.init_gui()

        # Robot
        if reset_robot:
            if "robot" in self.config_data.keys(): 
                self.robot = RobotAgent(self, self.config_data["robot"]["pos"].copy(), self.config_data["robot"]["yaw"], self.config_data["robot"]["radius"], self.config_data["robot"]["goals"].copy())
                self.insert_robot = True
            else: 
                self.robot = RobotAgent(self)
                self.insert_robot = False
        
        # Obstacles
        for wall in self.config_data["walls"]:
            self.walls.add(Obstacle(self, wall))

        # Humans
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
            whole_size = int(DISPLAY_SIZE * ZOOM_BOUNDS[1] + (SCROLL_BOUNDS[1]-SCROLL_BOUNDS[0]) * self.display_to_window_ratio)
            for i in range(int((DISPLAY_SIZE * ZOOM_BOUNDS[1] + SCROLL_BOUNDS[1] * self.display_to_window_ratio) / self.display_to_real_ratio)): # Positive lines (in Pygame frame)
                self.grid_lines.append([[0,whole_size + SCROLL_BOUNDS[0] * self.display_to_window_ratio + DISPLAY_SIZE * (1 - ZOOM_BOUNDS[1]) - ((i+1) * self.display_to_real_ratio)],[whole_size,whole_size + SCROLL_BOUNDS[0] * self.display_to_window_ratio + DISPLAY_SIZE * (1 - ZOOM_BOUNDS[1]) - ((i+1) * self.display_to_real_ratio)]]) # x line
                self.grid_lines.append([[-SCROLL_BOUNDS[0] * self.display_to_window_ratio + ((i+1) * self.display_to_real_ratio),whole_size],[-SCROLL_BOUNDS[0] * self.display_to_window_ratio + ((i+1) * self.display_to_real_ratio), 0]]) # y line
            for i in range(int((-SCROLL_BOUNDS[0] * self.display_to_window_ratio + DISPLAY_SIZE * (ZOOM_BOUNDS[1] - 1)) / self.display_to_real_ratio)): # Negative lines (in Pygame frame)
                self.grid_lines.append([[0,whole_size + SCROLL_BOUNDS[0] * self.display_to_window_ratio + DISPLAY_SIZE * (1 - ZOOM_BOUNDS[1]) + (i+1) * self.display_to_real_ratio],[whole_size,whole_size + SCROLL_BOUNDS[0] * self.display_to_window_ratio + DISPLAY_SIZE * (1 - ZOOM_BOUNDS[1]) + (i+1) * self.display_to_real_ratio]]) # x line
                self.grid_lines.append([[-SCROLL_BOUNDS[0] * self.display_to_window_ratio - (i+1) * self.display_to_real_ratio,whole_size],[-SCROLL_BOUNDS[0] * self.display_to_window_ratio - (i+1) * self.display_to_real_ratio, 0]]) # y line
            self.grid_lines.append([[0,whole_size + SCROLL_BOUNDS[0] * self.display_to_window_ratio + DISPLAY_SIZE * (1 - ZOOM_BOUNDS[1])],[whole_size,whole_size + SCROLL_BOUNDS[0] * self.display_to_window_ratio + DISPLAY_SIZE * (1 - ZOOM_BOUNDS[1])]]) # X AXIS
            self.grid_lines.append([[-SCROLL_BOUNDS[0] * self.display_to_window_ratio,whole_size],[-SCROLL_BOUNDS[0] * self.display_to_window_ratio, 0]]) # Y AXIS
            self.grid_surface = pygame.Surface((whole_size, whole_size), pygame.SRCALPHA)

            for i, line in enumerate(self.grid_lines):
                if i >= len(self.grid_lines)-2: pygame.draw.line(self.grid_surface, (0,0,255,255), line[0], line[1], 5)
                else: pygame.draw.line(self.grid_surface, (0,0,0,50), line[0], line[1], 3)
            
        # Scroll and zoom
        self.scroll = np.array([0.0,0.0], dtype=np.float16)
        self.display_scroll = np.array([0.0,0.0], dtype=np.float16)
        self.zoom = 1

        # Simulation stats
        self.show_stats = True

        # Human motion model
        self.motion_model_manager = MotionModelManager(self.motion_model, self.robot_visible, self.runge_kutta, self.humans, self.robot, self.walls.sprites())

        # Simulation variables
        self.n_updates = 0
        self.real_t = 0.0
        self.sim_t = 0.0
        self.last_reset = round_time(pygame.time.get_ticks() / 1000)
        self.last_pause_start = 0.0
        self.paused_time = 0.0
        self.paused = False
        self.updates_time = 0.0
        self.previous_updates_time = 0.0

    def generate_circular_crossing_setting(self, config_data:list):
        radius = config_data[0]
        n_actors = config_data[1]
        rand = config_data[2]
        model = config_data[3]
        headless = config_data[4]
        runge_kutta = config_data[5]
        insert_robot = config_data[6]
        randomize_human_attributes = config_data[7]
        robot_visible = config_data[8]
        center = np.array([self.real_size/2,self.real_size/2],dtype=np.float64)
        humans = {}
        humans_des_speed = []
        humans_radius = []
        if randomize_human_attributes:
            for i in range(n_actors):
                humans_des_speed.append(np.random.uniform(0.5, 1.5))
                humans_radius.append(np.random.uniform(0.3, 0.5))
        else: #TODO: Load it from config
            for i in range(n_actors):
                humans_des_speed.append(1.0)
                humans_radius.append(0.3)
        if not rand:
            if not insert_robot:
                angle = (2 * math.pi) / (n_actors)
                for i in range(n_actors):
                    center_pos = [radius * math.cos(angle * i), radius * math.sin(angle * i)]
                    humans[i] = {"pos": [center[0] + center_pos[0], center[1] + center_pos[1]],
                                "yaw": bound_angle(-math.pi + angle * i),
                                "goals": [[center[0] - center_pos[0], center[1] - center_pos[1]], [center[0] + center_pos[0], center[1] + center_pos[1]]],
                                "des_speed": humans_des_speed[i],
                                "radius": humans_radius[i]}
            else:
                robot = {"pos": [center[0], center[1]-radius], "yaw": math.pi / 2, "radius": 0.25, "goals": [[center[0], center[1]+radius]]}
                angle = (2 * math.pi) / (n_actors + 1)
                for i in range(n_actors):
                    center_pos = [radius * math.cos(-math.pi/2 + angle * i+1), radius * math.sin(-math.pi/2 + angle * i+1)]
                    humans[i] = {"pos": [center[0] + center_pos[0], center[1] + center_pos[1]],
                                "yaw": bound_angle(math.pi/2 + angle * i+1),
                                "goals": [[center[0] - center_pos[0], center[1] - center_pos[1]], [center[0] + center_pos[0], center[1] + center_pos[1]]],
                                "des_speed": humans_des_speed[i],
                                "radius": humans_radius[i]}
        else:
            humans_pos = []
            if not insert_robot:
                for i in range(n_actors):
                    while True:
                        angle = np.random.random() * np.pi * 2
                        pos_noise = np.array([(np.random.random() - 0.5) * humans_des_speed[i], (np.random.random() - 0.5) * humans_des_speed[i]], dtype=np.float64)
                        pos = np.array([center[0] + radius * np.cos(angle) + pos_noise[0], center[1] + radius * np.sin(angle) + pos_noise[1]], dtype=np.float64)
                        collide = False 
                        for j in range(len(humans_pos)):
                            min_dist = humans_radius[i] + humans_radius[j] + 0.2 # This last element is kind of a discomfort distance
                            other_human_pos = np.array([humans_pos[j][0],humans_pos[j][1]], dtype=np.float64)
                            other_human_goal = np.array([-humans_pos[j][0] + 2 * center[0],-humans_pos[j][1] + 2 * center[0]], dtype=np.float64)
                            if np.linalg.norm(pos - other_human_pos) < min_dist or np.linalg.norm(pos - other_human_goal) < min_dist:
                                collide = True
                                break
                        if not collide: 
                            humans_pos.append([pos[0], pos[1]])
                            humans[i] = {"pos": [pos[0], pos[1]],
                                        "yaw": bound_angle(math.pi + angle),
                                        "goals": [[center[0] * 2 - pos[0], center[1] * 2 - pos[1]], [pos[0], pos[1]]],
                                        "des_speed": humans_des_speed[i],
                                        "radius": humans_radius[i]}
                            break
            else:
                robot = {"pos": [center[0], center[1]-radius], "yaw": math.pi / 2, "radius": 0.25, "goals": [[center[0], center[1]+radius]]}
                for i in range(n_actors):
                    while True:
                        angle = np.random.random() * np.pi * 2
                        pos_noise = np.array([(np.random.random() - 0.5) * humans_des_speed[i], (np.random.random() - 0.5) * humans_des_speed[i]], dtype=np.float64)
                        pos = np.array([center[0] + radius * np.cos(angle) + pos_noise[0], center[1] + radius * np.sin(angle) + pos_noise[1]], dtype=np.float64)
                        collide = False 
                        for j in range(len(humans_pos)):
                            min_dist = humans_radius[i] + humans_radius[j] + 0.2 # This last element is kind of a discomfort distance
                            other_human_pos = np.array([humans_pos[j][0],humans_pos[j][1]], dtype=np.float64)
                            other_human_goal = np.array([-humans_pos[j][0] + 2 * center[0],-humans_pos[j][1] + 2 * center[0]], dtype=np.float64)
                            if np.linalg.norm(pos - other_human_pos) < min_dist or np.linalg.norm(pos - other_human_goal) < min_dist:
                                collide = True
                                break
                        robot_pos = np.array([center[0], center[1]-radius], dtype=np.float64)
                        robot_goal = np.array([center[0], center[1]+radius], dtype=np.float64)
                        if np.linalg.norm(pos - robot_pos) < humans_radius[i] + 0.25 + 0.2 or np.linalg.norm(pos - robot_goal) < humans_radius[i] + 0.25 + 0.2:
                            collide = True
                        if not collide: 
                            humans_pos.append([pos[0], pos[1]])
                            humans[i] = {"pos": [pos[0], pos[1]],
                                        "yaw": bound_angle(math.pi + angle),
                                        "goals": [[center[0] * 2 - pos[0], center[1] * 2 - pos[1]], [pos[0], pos[1]]],
                                        "des_speed": humans_des_speed[i],
                                        "radius": humans_radius[i]}
                            break
        if insert_robot: data = {"motion_model": model, "headless": headless, "runge_kutta": runge_kutta, "robot_visible": robot_visible, "grid": True, "walls": [], "humans": humans, "robot": robot}
        else: data = {"motion_model": model, "headless": headless, "runge_kutta": runge_kutta, "robot_visible": False, "grid": True, "walls": [], "humans": humans}
        self.config_data = data
        return data

    def generate_square_crossing_human(self, config_data:list):
        ### TO BE IMPLEMENTED
        pass

    def render_sim(self):
        self.display = pygame.Surface((int(DISPLAY_SIZE / self.zoom),int(DISPLAY_SIZE / self.zoom))) # For zooming
        self.display.fill((255,255,255))

        # Change based on scroll and zoom
        if self.grid: self.display.blit(self.grid_surface, (SCROLL_BOUNDS[0] * self.display_to_window_ratio - self.display_scroll[0], SCROLL_BOUNDS[0] * self.display_to_window_ratio - self.display_scroll[1]))
        for wall in self.walls.sprites(): wall.render(self.display, self.display_scroll)
        for human in self.humans: human.update(); human.render(self.display, self.display_scroll)
        if self.insert_robot: self.robot.update(); self.robot.render(self.display, self.display_scroll)
        pygame.transform.scale(self.display, (WINDOW_SIZE, WINDOW_SIZE), self.screen)

        # Fixed on screen
        if self.show_stats:
            self.fps_text = self.font.render(f"FPS: {round(self.clock.get_fps())}", False, (0,0,255))
            self.real_time = self.font.render(f"Real time: {self.real_t}", False, (0,0,255))
            self.simulation_time = self.font.render(f"Sim. time: {self.sim_t}", False, (0,0,255))
            if self.n_updates < N_UPDATES_AVERAGE_TIME * 2: self.real_time_factor = self.font.render(f"Time fact.: {round(self.sim_t/ (self.real_t + 0.00000001), 2)}", False, (0,0,255))
            else: self.real_time_factor = self.font.render(f"Time fact.: {round((SAMPLING_TIME * N_UPDATES_AVERAGE_TIME) / (self.updates_time - self.previous_updates_time), 2)}", False, (0,0,255))
            self.screen.blit(self.fps_text,self.fps_text_rect)
            self.screen.blit(self.real_time,self.real_time_rect)
            self.screen.blit(self.simulation_time,self.simulation_time_rect)
            self.screen.blit(self.real_time_factor,self.real_time_factor_rect)
            self.screen.blit(self.x_axis_label,self.x_axis_label_rect)
            self.screen.blit(self.y_axis_label,self.y_axis_label_rect)
            if self.paused: 
                self.screen.blit(self.pause_text, self.pause_text_rect)
                self.screen.blit(self.rewind_text, self.rewind_text_rect)
                self.screen.blit(self.reset_text, self.reset_text_rect)
                self.screen.blit(self.speedup_text, self.speedup_text_rect)

        pygame.display.update()

    def update(self):
        self.n_updates += 1
        self.motion_model_manager.update(self.sim_t, SAMPLING_TIME)
        self.real_t = round_time((pygame.time.get_ticks() / 1000) - self.last_reset - self.paused_time)
        self.sim_t = round_time(self.n_updates * SAMPLING_TIME)
        if self.n_updates % N_UPDATES_AVERAGE_TIME == 0: self.previous_updates_time = self.updates_time; self.updates_time = (pygame.time.get_ticks() / 1000) - self.last_reset - self.paused_time

    def move_robot_with_keys(self, humans, walls):
        if pygame.key.get_pressed()[pygame.K_UP]: self.robot.move_with_keys('up', humans, walls)
        if pygame.key.get_pressed()[pygame.K_DOWN]: self.robot.move_with_keys('down', humans, walls)
        if pygame.key.get_pressed()[pygame.K_LEFT]: self.robot.move_with_keys('left', humans, walls)
        if pygame.key.get_pressed()[pygame.K_RIGHT]: self.robot.move_with_keys('right', humans, walls)

    def rewind_human_state(self):
        if len(self.human_states) > 0:
            self.n_updates -= 1
            state = self.human_states[self.n_updates]
            self.motion_model_manager.set_human_states(state)
            self.human_states = self.human_states[:-1]

    def run_live(self):
        self.active = True
        if self.motion_model_manager.headed: self.human_states = np.array([self.motion_model_manager.get_human_states(include_goal=True, headed= True)], dtype=np.float64)
        else: self.human_states = np.array([self.motion_model_manager.get_human_states(include_goal=True, headed= False)], dtype=np.float64)
        while self.active:
            if not self.paused:
                if self.insert_robot: self.move_robot_with_keys(self.humans, self.walls)
                self.update()
                if not self.headless: self.render_sim()
                if self.motion_model_manager.headed: self.human_states = np.append(self.human_states, [self.motion_model_manager.get_human_states(include_goal=True, headed= True)], axis=0)
                else: self.human_states = np.append(self.human_states, [self.motion_model_manager.get_human_states(include_goal=True, headed= False)], axis=0)
            else:
                if not self.headless: 
                    self.render_sim()
                    # Rewind
                    if pygame.key.get_pressed()[pygame.K_z]: 
                        r_is_pressed = True
                        while r_is_pressed:
                            self.rewind_human_state()
                            self.render_sim()
                            pygame.event.get(); r_is_pressed = pygame.key.get_pressed()[pygame.K_z]
                    # Reset
                    if pygame.key.get_pressed()[pygame.K_r]: 
                        self.reset_sim()
                        if self.motion_model_manager.headed: self.human_states = np.array([self.motion_model_manager.get_human_states(include_goal=True, headed= True)], dtype=np.float64)
                        else: self.human_states = np.array([self.motion_model_manager.get_human_states(include_goal=True, headed= False)], dtype=np.float64)
                    # Speed up (or Resume)
                    if pygame.key.get_pressed()[pygame.K_s]:
                        self.paused_time += round_time((pygame.time.get_ticks() / 1000) - self.last_pause_start)
                        s_is_pressed = True
                        while s_is_pressed:
                            self.update()
                            self.render_sim()
                            if self.motion_model_manager.headed: self.human_states = np.append(self.human_states, [self.motion_model_manager.get_human_states(include_goal=True, headed= True)], axis=0)
                            else: self.human_states = np.append(self.human_states, [self.motion_model_manager.get_human_states(include_goal=True, headed= False)], axis=0)
                            pygame.event.get(); s_is_pressed = pygame.key.get_pressed()[pygame.K_s]
                        self.last_pause_start = round_time(pygame.time.get_ticks() / 1000)
            for event in pygame.event.get():
                # Exit
                if event.type == pygame.QUIT:
                    self.active = False
                    pygame.quit()
                # Pause
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    if self.paused: self.last_pause_start = round_time(pygame.time.get_ticks() / 1000)
                    else: self.paused_time += round_time((pygame.time.get_ticks() / 1000) - self.last_pause_start)
                # Reset scroll and zoom
                if event.type == pygame.KEYDOWN and event.key == pygame.K_o:
                    self.scroll -= self.scroll
                    self.display_scroll = self.scroll * self.display_to_window_ratio
                    self.zoom = 1
                # Hide/Show simulation stats
                if event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                    self.show_stats = not self.show_stats
                # Scroll
                if event.type == pygame.MOUSEMOTION and event.buttons[1]:
                    self.scroll -= event.rel
                    self.scroll[0] = min(SCROLL_BOUNDS[1],max(SCROLL_BOUNDS[0], self.scroll[0]))
                    self.scroll[1] = min(SCROLL_BOUNDS[1],max(SCROLL_BOUNDS[0], self.scroll[1]))
                    self.display_scroll = self.scroll * self.display_to_window_ratio
                # Zoom
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 4:
                    self.zoom += 0.1
                    self.zoom = min(self.zoom, ZOOM_BOUNDS[1])
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 5:
                    self.zoom -= 0.1
                    self.zoom = max(ZOOM_BOUNDS[0], self.zoom)
            self.clock.tick(MAX_FPS)
    
    def run_from_precomputed_states(self, human_states):
        self.config_data["headless"] = False
        self.reset_sim(restart_gui=True)
        self.updates_time = SAMPLING_TIME * N_UPDATES_AVERAGE_TIME # Just to not give error
        for i in range(len(human_states)):
            self.motion_model_manager.set_human_states(human_states[i], just_visual=True)
            self.n_updates += 1
            self.real_t = round_time((pygame.time.get_ticks() / 1000) - self.last_reset)
            self.sim_t = round_time(self.n_updates * SAMPLING_TIME)
            self.render_sim()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); self.pygame_init = False
            self.clock.tick(MAX_FPS)
        pygame.quit(); self.pygame_init = False

    def run_k_steps(self, steps, quit=True):
        human_states = np.empty((steps,len(self.humans),N_GENERAL_STATES), dtype=np.float64)
        for step in range(steps):
            self.update()
            if not self.headless: self.render_sim()
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
            self.human_states = np.empty((len(models),n_updates+1,len(self.humans),N_GENERAL_STATES), dtype=np.float64)
            test_times = np.empty((len(models),), dtype=np.float64)
            for i in range(len(models)):
                self.reset_sim()
                self.motion_model_manager.set_motion_model(models[i])
                self.human_states[i], test_times[i] = self.run_single_test(n_updates)
                figure, ax = plt.subplots()
                figure.suptitle(f'Human agents\' position over simulation | T = {final_time} | dt = {round(SAMPLING_TIME, 4)} | Model = {models[i]}')
                if self.motion_model_manager.runge_kutta == False: integration_title = "Euler"
                else: integration_title = "Runge-Kutta-45"
                ax.set(xlabel='X',ylabel='Y',title=f'{integration_title} | Elapsed time = {test_times[i]}',xlim=[0,REAL_SIZE],ylim=[0,REAL_SIZE])
                self.plot_agents_position_with_sample(ax,self.human_states[i],plot_sample_time,models[i])
        else:
            self.human_states = np.empty((len(models),2,n_updates+1,len(self.humans),N_GENERAL_STATES), dtype=np.float64)
            test_times = np.empty((len(models),2), dtype=np.float64)
            for i in range(len(models)):
                self.reset_sim()
                self.motion_model_manager.set_motion_model(models[i])
                self.motion_model_manager.runge_kutta = False
                self.human_states[i,0], test_times[i,0] = self.run_single_test(n_updates)
                self.reset_sim()
                self.motion_model_manager.set_motion_model(models[i])
                self.motion_model_manager.runge_kutta = True
                self.human_states[i,1], test_times[i,1] = self.run_single_test(n_updates)
                figure, ax = plt.subplots(1,2)
                figure.suptitle(f'Humans\' position over simulation | T = {final_time} | dt = {round(SAMPLING_TIME, 4)} | Model = {models[i]}')
                ax[0].set(xlabel='X',ylabel='Y',title=f'Euler | Elapsed time = {test_times[i,0]}',xlim=[0,REAL_SIZE],ylim=[0,REAL_SIZE])
                self.plot_agents_position_with_sample(ax[0],self.human_states[i,0],plot_sample_time,models[i])
                ax[1].set(xlabel='X',ylabel='Y',title=f'Runge-Kutta-45 | Elapsed time = {test_times[i,1]}',xlim=[0,REAL_SIZE],ylim=[0,REAL_SIZE])
                self.plot_agents_position_with_sample(ax[1],self.human_states[i,1],plot_sample_time,models[i])

        if not self.headless: pygame.quit(); self.pygame_init = False
        plt.show()

    def run_integration_test(self, final_time=40):
        n_updates = int(final_time / SAMPLING_TIME)
        self.human_states = np.empty((2,n_updates+1,len(self.humans),N_GENERAL_STATES), dtype=np.float64)
        ## Euler
        self.motion_model_manager.runge_kutta = False
        self.human_states[0], euler_time = self.run_single_test(n_updates)
        ## Runge kutta
        self.reset_sim()
        self.motion_model_manager.runge_kutta = True
        self.human_states[1], rk45_time = self.run_single_test(n_updates)
        if not self.headless: pygame.quit(); self.pygame_init = False
        ## Print
        figure, axs = plt.subplots(1, 2)
        figure.suptitle(f'Human agents\' position over simulation | T = {final_time} | dt = {round(SAMPLING_TIME, 4)} | motion model: {self.motion_model}')
        axs[0].set(xlabel='X',ylabel='Y',title=f'Euler | Elapsed time = {euler_time}',xlim=[0,REAL_SIZE],ylim=[0,REAL_SIZE])
        self.plot_agents_trajectory(axs[0], self.human_states[0])
        axs[1].set(xlabel='X',ylabel='Y',title=f'Runge-Kutta 45 | Elapsed time = {rk45_time}',xlim=[0,REAL_SIZE],ylim=[0,REAL_SIZE])
        self.plot_agents_trajectory(axs[1], self.human_states[1])
        ## Display graphics
        plt.show()

    def run_complete_rk45_simulation(self, sampling_time=1/60, final_time=40, plot_sample_time=3):
        self.reset_sim()
        global SAMPLING_TIME; SAMPLING_TIME = sampling_time
        start_time = round_time((pygame.time.get_ticks() / 1000))
        self.human_states = self.motion_model_manager.complete_rk45_simulation(0.0, sampling_time, final_time)
        test_time = round_time((pygame.time.get_ticks() / 1000) - start_time - self.paused_time)
        figure, ax = plt.subplots()
        figure.suptitle(f'Human agents\' position over simulation | T = {final_time} | dt = {round(sampling_time, 4)} | Model = {self.motion_model_manager.motion_model_title}')
        ax.set(xlabel='X',ylabel='Y',title=f'Complete Runge-kutta-45 | Elapsed time = {test_time}',xlim=[0,REAL_SIZE],ylim=[0,REAL_SIZE])
        self.plot_agents_position_with_sample(ax,self.human_states,plot_sample_time,self.motion_model_manager.motion_model_title)
        if not self.headless: pygame.quit(); self.pygame_init = False
        plt.show()

    def print_walls_on_plot(self, ax):
        for i in range(len(self.walls)):
                ax.fill(self.walls.sprites()[i].vertices[:,0], self.walls.sprites()[i].vertices[:,1], facecolor='black', edgecolor='black')

    def plot_agents_position_with_sample(self, ax, human_states, plot_sample_time:float, model:str):
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
                ax.text(human_states[k,j,0],human_states[k,j,1], f"{k*SAMPLING_TIME}", color=COLORS[color_idx], va="center", ha="center", fontsize="xx-small", zorder=1)
            goals = np.array(self.humans[j].goals, dtype=np.float64).copy()
            for k in range(len(goals)):
                if goals[k,0] == human_states[0,j,0] and goals[k,1] == human_states[0,j,1]: 
                    goals = np.delete(goals, k, 0)
                    break
            ax.scatter(goals[:,0], goals[:,1], marker="*", color=COLORS[color_idx], zorder=2)

    def plot_agents_trajectory(self, ax, human_states):
        ax.axis('equal')
        self.print_walls_on_plot(ax)
        for i in range(len(self.humans)):
            ax.plot(human_states[:,i,0],human_states[:,i,1])