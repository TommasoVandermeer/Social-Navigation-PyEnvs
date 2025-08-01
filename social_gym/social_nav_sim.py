import pygame
from crowd_nav.utils.state import ObservableState, ObservableStateHeaded
from social_gym.src.info import *
from social_gym.src.motion_model_manager import MotionModelManager, N_GENERAL_STATES
from social_gym.src.human_agent import HumanAgent
from social_gym.src.robot_agent import RobotAgent
from social_gym.src.obstacle import Obstacle
from social_gym.src.utils import round_time, bound_angle, point_to_segment_dist, is_multiple, PRECISION
from crowd_nav.policy.policy_factory import policy_factory
import torch
import configparser
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
ROBOT_SAMPLING_TIME = 1/4 # Must be a multiple of SAMPLING_TIME
N_UPDATES_AVERAGE_TIME = 20
MOTION_MODELS = ["sfm_roboticsupo","sfm_helbing","sfm_guo","sfm_moussaid","hsfm_farina","hsfm_guo",
                 "hsfm_moussaid","hsfm_new","hsfm_new_guo","hsfm_new_moussaid","orca"]
COLORS = list(mcolors.TABLEAU_COLORS.values())
ZOOM_BOUNDS = [0.5, 2]
SCROLL_BOUNDS = [-500,500]

class SocialNavSim:
    def __init__(self, config_data, scenario = "custom_config", parallelize_robot = False, parallelize_humans = False):
        pygame.init()
        self.pygame_init = True

        self.display_to_real_ratio = DISPLAY_SIZE / REAL_SIZE
        self.display_to_window_ratio = DISPLAY_SIZE / WINDOW_SIZE
        self.real_size = REAL_SIZE
        self.clock = pygame.time.Clock()
        self.walls = pygame.sprite.Group()
        self.humans = []
        self.mode = scenario
        self.parallelize_robot = parallelize_robot
        self.parallelize_humans = parallelize_humans

        if scenario == "custom_config": self.config_data = config_data
        elif scenario == "circular_crossing": self.config_data = self.generate_circular_crossing_setting(**config_data)
        elif scenario == "parallel_traffic": self.config_data = self.generate_parallel_traffic_scenario(**config_data)
        elif scenario == "circular_crossing_with_static_obstacles": self.config_data = self.generate_circular_crossing_with_static_obstacles(**config_data)
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
        self.robot_env_same_timestep = (SAMPLING_TIME == ROBOT_SAMPLING_TIME)

    def set_robot_time_step(self, time_step:float):
        if not is_multiple(time_step, SAMPLING_TIME): raise ValueError("Robot time step must be a multiple of the environment sampling time")
        else:
            global ROBOT_SAMPLING_TIME
            ROBOT_SAMPLING_TIME = time_step
            self.robot.time_step = time_step
            if self.robot.policy is not None: self.robot.policy.time_step = time_step
        self.robot_env_same_timestep = (SAMPLING_TIME == ROBOT_SAMPLING_TIME)

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
        self.updated = True
        
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
                if i >= len(self.grid_lines)-2: pygame.draw.line(self.grid_surface, (0,0,255,255), line[0], line[1], 4)
                else: pygame.draw.line(self.grid_surface, (0,0,0,50), line[0], line[1], 3)
            
        # Scroll and zoom
        self.scroll = np.array([-350.0,+350.0], dtype=np.float16) # Default [0,0]
        self.display_scroll = self.scroll * self.display_to_window_ratio
        self.zoom = 1

        # Simulation stats
        self.show_stats = True

        # Human motion model
        if self.motion_model == "orca": self.parallelize_humans = False
        if hasattr(self, "motion_model_manager") and self.motion_model_manager.robot_motion_model_title is not None:
            robot_motion_model_title = self.motion_model_manager.robot_motion_model_title
            robot_runge_kutta = self.motion_model_manager.robot_runge_kutta
            self.motion_model_manager = MotionModelManager(self.motion_model, self.robot_visible, self.runge_kutta, self.humans, self.robot, self.walls.sprites(), parallelize = self.parallelize_humans)
            self.motion_model_manager.set_robot_motion_model(robot_motion_model_title, robot_runge_kutta)
            self.robot_crowdnav_policy = False
            self.robot_controlled = True
        else: 
            self.motion_model_manager = MotionModelManager(self.motion_model, self.robot_visible, self.runge_kutta, self.humans, self.robot, self.walls.sprites(), parallelize = self.parallelize_humans)
            self.robot_controlled = False
        if hasattr(self, "parallel_traffic_humans_respawn") and self.parallel_traffic_humans_respawn: 
            self.motion_model_manager.parallel_traffic_humans_respawn = True
            self.motion_model_manager.respawn_bounds = self.respawn_bounds

        # Simulation variables
        self.robot_env_same_timestep = (SAMPLING_TIME == ROBOT_SAMPLING_TIME)
        self.n_updates = 0
        self.real_t = 0.0
        self.sim_t = 0.0
        self.last_reset = round_time(pygame.time.get_ticks() / 1000)
        self.last_pause_start = 0.0
        self.paused_time = 0.0
        self.paused = False
        self.updates_time = 0.0
        self.previous_updates_time = 0.0

    def generate_circular_crossing_setting(self, **kwargs):
        ## Get input data
        insert_robot = kwargs["insert_robot"] if "insert_robot" in kwargs else False
        model = kwargs["human_policy"] if "human_policy" in kwargs else "sfm_guo"
        headless = kwargs["headless"] if "headless" in kwargs else False
        runge_kutta = kwargs["runge_kutta"] if "runge_kutta" in kwargs else False
        robot_visible = kwargs["robot_visible"] if "robot_visible" in kwargs else False
        robot_r = kwargs["robot_radius"] if "robot_radius" in kwargs else 0.3
        radius = kwargs["circle_radius"] if "circle_radius" in kwargs else 7
        n_actors = kwargs["n_actors"] if "n_actors" in kwargs else 10
        randomize_human_attributes = kwargs["randomize_human_attributes"] if "randomize_human_attributes" in kwargs else False
        rand = kwargs["randomize_human_positions"] if "randomize_human_positions" in kwargs else False
        # Generate humans initial condition
        center = np.array([0,0],dtype=PRECISION) # [self.real_size/2,self.real_size/2]
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
                robot = {"pos": [center[0], center[1]-radius], "yaw": math.pi / 2, "radius": robot_r, "goals": [[center[0], center[1]+radius],[center[0],center[1]-radius]]}
                angle = (2 * math.pi) / (n_actors + 1)
                for i in range(n_actors):
                    center_pos = [radius * math.cos(-(math.pi/2) + angle * (i+1)), radius * math.sin(-(math.pi/2) + angle * (i+1))]
                    humans[i] = {"pos": [center[0] + center_pos[0], center[1] + center_pos[1]],
                                "yaw": bound_angle((math.pi/2) + angle * (i+1)),
                                "goals": [[center[0] - center_pos[0], center[1] - center_pos[1]], [center[0] + center_pos[0], center[1] + center_pos[1]]],
                                "des_speed": humans_des_speed[i],
                                "radius": humans_radius[i]}
        else:
            humans_pos = []
            if not insert_robot:
                for i in range(n_actors):
                    while True:
                        angle = np.random.random() * np.pi * 2
                        pos_noise = np.array([(np.random.random() - 0.5) * humans_des_speed[i], (np.random.random() - 0.5) * humans_des_speed[i]], dtype=PRECISION)
                        pos = np.array([center[0] + radius * np.cos(angle) + pos_noise[0], center[1] + radius * np.sin(angle) + pos_noise[1]], dtype=PRECISION)
                        collide = False 
                        for j in range(len(humans_pos)):
                            min_dist = humans_radius[i] + humans_radius[j] + 0.2 # This last element is kind of a discomfort distance
                            other_human_pos = np.array([humans_pos[j][0],humans_pos[j][1]], dtype=PRECISION)
                            other_human_goal = np.array([-humans_pos[j][0] + 2 * center[0],-humans_pos[j][1] + 2 * center[0]], dtype=PRECISION)
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
                robot = {"pos": [center[0], center[1]-radius], "yaw": math.pi / 2, "radius": robot_r, "goals": [[center[0], center[1]+radius],[center[0],center[1]-radius]]}
                for i in range(n_actors):
                    while True:
                        angle = np.random.random() * np.pi * 2
                        pos_noise = np.array([(np.random.random() - 0.5) * humans_des_speed[i], (np.random.random() - 0.5) * humans_des_speed[i]], dtype=PRECISION)
                        pos = np.array([center[0] + radius * np.cos(angle) + pos_noise[0], center[1] + radius * np.sin(angle) + pos_noise[1]], dtype=PRECISION)
                        collide = False 
                        for j in range(len(humans_pos)):
                            min_dist = humans_radius[i] + humans_radius[j] + 0.2 # This last element is kind of a discomfort distance
                            other_human_pos = np.array([humans_pos[j][0],humans_pos[j][1]], dtype=PRECISION)
                            other_human_goal = np.array([-humans_pos[j][0] + 2 * center[0],-humans_pos[j][1] + 2 * center[0]], dtype=PRECISION)
                            if np.linalg.norm(pos - other_human_pos) < min_dist or np.linalg.norm(pos - other_human_goal) < min_dist:
                                collide = True
                                break
                        robot_pos = np.array([center[0], center[1]-radius], dtype=PRECISION)
                        robot_goal = np.array([center[0], center[1]+radius], dtype=PRECISION)
                        if np.linalg.norm(pos - robot_pos) < humans_radius[i] + robot_r + 0.2 or np.linalg.norm(pos - robot_goal) < humans_radius[i] + robot_r + 0.2:
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

    def generate_parallel_traffic_scenario(self, **kwargs):
        ## Get input data
        insert_robot = kwargs["insert_robot"] if "insert_robot" in kwargs else False
        human_policy = kwargs["human_policy"] if "human_policy" in kwargs else "sfm_guo"
        headless = kwargs["headless"] if "headless" in kwargs else False
        runge_kutta = kwargs["runge_kutta"] if "runge_kutta" in kwargs else False
        robot_visible = kwargs["robot_visible"] if "robot_visible" in kwargs else False
        robot_radius = kwargs["robot_radius"] if "robot_radius" in kwargs else 0.3
        traffic_length = kwargs["traffic_length"] if "traffic_length" in kwargs else 14
        traffic_height = kwargs["traffic_height"] if "traffic_height" in kwargs else 3
        n_actors = kwargs["n_actors"] if "n_actors" in kwargs else 10
        randomize_human_attributes = kwargs["randomize_human_attributes"] if "randomize_human_attributes" in kwargs else False
        ## Generate robot initial condition
        if insert_robot: robot = {"pos": [- (traffic_length / 2) + 1, 0], "yaw": 0.0, "radius": robot_radius, "goals": [[(traffic_length / 2) - 1, 0],[- (traffic_length / 2) + 1, 0]]}
        ## Generate humans initial condition
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
        humans_area = 0
        for r in humans_radius: humans_area += math.pi * (r**2)
        if humans_area > traffic_length * traffic_height * 0.4: raise ValueError("Number of humans specified is too big for desided traffic height and length")
        humans_pos = []
        for i in range(n_actors):
            while True:
                # a = 0 + humans_radius[i]
                # b = traffic_length - humans_radius[i]
                a = -(traffic_length/2 ) + humans_radius[i]
                b = traffic_length/2 - humans_radius[i]
                pos = np.array([(b - a) * np.random.random() + a, (np.random.random() - 0.5) * traffic_height], dtype=PRECISION)
                # pos = np.array([b - ((b - a) * np.random.random()) / 8, (np.random.random() - 0.5) * traffic_height], dtype=PRECISION) # Agents much closer to b boundary
                collide = False
                for j in range(len(humans_pos)):
                    other_human_pos = humans_pos[j]
                    if np.linalg.norm(pos - other_human_pos) - humans_radius[i] - humans_radius[j] - 0.1 < 0: # This is  discomfort distance
                        collide = True 
                        break
                if insert_robot and np.linalg.norm(pos - np.array(robot["pos"], PRECISION)) - humans_radius[i] - robot["radius"] - 0.1 < 0: # This is  discomfort distance
                    collide = True
                if not collide:
                    humans_pos.append(pos)
                    humans[i] = {"pos": [pos[0], pos[1]],
                                 "yaw": bound_angle(-math.pi),
                                 "goals": [[-(traffic_length / 2)-3, pos[1]]],
                                 "des_speed": humans_des_speed[i],
                                 "radius": humans_radius[i]}
                    break
        ## Generate final data
        if insert_robot: data = {"motion_model": human_policy, "headless": headless, "runge_kutta": runge_kutta, "robot_visible": robot_visible, "grid": True, "walls": [], "humans": humans, "robot": robot}
        else: data = {"motion_model": human_policy, "headless": headless, "runge_kutta": runge_kutta, "robot_visible": False, "grid": True, "walls": [], "humans": humans}
        ## Set parallel traffic humans respawn
        self.parallel_traffic_humans_respawn = True
        self.respawn_bounds = ((traffic_length / 2), (traffic_height / 2))
        self.config_data = data
        return data

    def generate_circular_crossing_with_static_obstacles(self, **kwargs):
        ## Get input data
        insert_robot = kwargs["insert_robot"] if "insert_robot" in kwargs else False
        model = kwargs["human_policy"] if "human_policy" in kwargs else "sfm_guo"
        headless = kwargs["headless"] if "headless" in kwargs else False
        runge_kutta = kwargs["runge_kutta"] if "runge_kutta" in kwargs else False
        robot_visible = kwargs["robot_visible"] if "robot_visible" in kwargs else False
        robot_r = kwargs["robot_radius"] if "robot_radius" in kwargs else 0.3
        radius = kwargs["circle_radius"] if "circle_radius" in kwargs else 7
        n_actors = kwargs["n_actors"] if "n_actors" in kwargs else 10
        # Generate humans initial condition
        assert radius > 5, "Radius must be greater than 5 for this scenario"
        inner_circle_radius = radius - 3
        center = np.array([0,0],dtype=PRECISION) # [self.real_size/2,self.real_size/2]
        humans = {}
        humans_des_speed = []
        humans_radius = []
        for i in range(n_actors):
            if i < 3: 
                humans_des_speed.append(0.0)
                humans_radius.append(1 + (np.random.random()-1) * 0.4)
            else: 
                humans_des_speed.append(1.0)
                humans_radius.append(0.3)
        humans_pos = []
        robot_pos = np.array([center[0], center[1]-radius], dtype=PRECISION)
        robot_goal = np.array([center[0], center[1]+radius], dtype=PRECISION)
        for i in range(n_actors):
            while True:
                if i < 3:
                    angle = (np.pi / int(n_actors / 2)) * (-0.5 + 2 * i + (np.random.random() - 0.5) * 0.5)
                    pos_noise = np.array([(np.random.random() - 0.5) * 0.1, (np.random.random() - 0.5) * 0.1], dtype=PRECISION)
                    pos = np.array([center[0] + inner_circle_radius * np.cos(angle) + pos_noise[0], center[1] + inner_circle_radius * np.sin(angle) + pos_noise[1]], dtype=PRECISION)
                else:
                    angle = (np.pi / int(n_actors / 2)) * (0.5 + 2 * i + (np.random.random() - 0.5) * 0.5)
                    pos_noise = np.array([(np.random.random() - 0.5) * 0.7, (np.random.random() - 0.5) * 0.7], dtype=PRECISION)
                    pos = np.array([center[0] + radius * np.cos(angle) + pos_noise[0], center[1] + radius * np.sin(angle) + pos_noise[1]], dtype=PRECISION)
                collide = False 
                for j in range(len(humans_pos)):
                    min_dist = humans_radius[i] + humans_radius[j] + 0.2 # This last element is kind of a discomfort distance
                    other_human_pos = np.array([humans_pos[j][0],humans_pos[j][1]], dtype=PRECISION)
                    if j < 3: 
                        other_human_goal = np.array([humans_pos[j][0],humans_pos[j][1]], dtype=PRECISION)
                    else:
                        other_human_goal = np.array([-humans_pos[j][0] + 2 * center[0],-humans_pos[j][1] + 2 * center[0]], dtype=PRECISION)
                    if np.linalg.norm(pos - other_human_pos) < min_dist or np.linalg.norm(pos - other_human_goal) < min_dist:
                        collide = True
                        break
                if np.linalg.norm(pos - robot_pos) < humans_radius[i] + robot_r + 0.2 or np.linalg.norm(pos - robot_goal) < humans_radius[i] + robot_r + 0.2:
                    collide = True
                if not collide: 
                    humans_pos.append([pos[0], pos[1]])
                    if i < 3: 
                        human_goals = [[pos[0], pos[1]], [pos[0], pos[1]]]
                    else:
                        human_goals = [[center[0] * 2 - pos[0], center[1] * 2 - pos[1]], [pos[0], pos[1]]]
                    humans[i] = {"pos": [pos[0], pos[1]],
                                "yaw": bound_angle(math.pi + angle),
                                "goals": human_goals,
                                "des_speed": humans_des_speed[i],
                                "radius": humans_radius[i]}
                    break
        if insert_robot: 
            robot = {"pos": [center[0], center[1]-radius], "yaw": math.pi / 2, "radius": robot_r, "goals": [[center[0], center[1]+radius],[center[0],center[1]-radius]]}
            data = {"motion_model": model, "headless": headless, "runge_kutta": runge_kutta, "robot_visible": robot_visible, "grid": True, "walls": [], "humans": humans, "robot": robot}
        else: data = {"motion_model": model, "headless": headless, "runge_kutta": runge_kutta, "robot_visible": False, "grid": True, "walls": [], "humans": humans}
        self.config_data = data
        return data

    def render_sim(self):
        self.display = pygame.Surface((int(DISPLAY_SIZE / self.zoom),int(DISPLAY_SIZE / self.zoom))) # For zooming
        self.display.fill((255,255,255))

        # Change based on scroll and zoom
        if self.grid: self.display.blit(self.grid_surface, (SCROLL_BOUNDS[0] * self.display_to_window_ratio - self.display_scroll[0], SCROLL_BOUNDS[0] * self.display_to_window_ratio - self.display_scroll[1]))
        for wall in self.walls.sprites(): wall.render(self.display, self.display_scroll)
        # for human in self.humans: human.update(); human.render(self.display, self.display_scroll)
        for i, human in enumerate(self.humans):
            if self.mode != 'circular_crossing_with_static_obstacles':
                human.update()
                human.render(self.display, self.display_scroll)
            else:
                if i < 3: 
                    self.display.blit(human.outer_circle, (human.rect.x - self.display_scroll[0], human.rect.y - self.display_scroll[1]))
                else:
                    human.update()
                    human.render(self.display, self.display_scroll)
            
        if self.insert_robot: self.robot.update(); self.robot.render(self.display, self.display_scroll)
        pygame.transform.scale(self.display, (WINDOW_SIZE, WINDOW_SIZE), self.screen)

        # Fixed on screen
        if self.show_stats:
            self.fps_text = self.font.render(f"FPS: {round(self.clock.get_fps())}", False, (0,0,255))
            self.real_time = self.font.render(f"Real time: {self.real_t}", False, (0,0,255))
            self.simulation_time = self.font.render(f"Sim. time: {round_time(self.sim_t)}", False, (0,0,255))
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
        ############################
        # self.n_updates += 1
        # if self.insert_robot: self.control_robot()
        # self.motion_model_manager.update_humans(self.sim_t, SAMPLING_TIME)
        # self.update_times()
        ############################
        self.n_updates += 1
        if self.insert_robot: 
            robot_state = self.robot.get_safe_state()
            self.control_robot()
            new_robot_state = self.robot.get_safe_state()
            self.robot.set_state(robot_state)
        self.motion_model_manager.update_humans(self.sim_t, SAMPLING_TIME)
        if self.insert_robot:
            self.robot.set_state(new_robot_state)
        self.update_times()

    def update_times(self):
        self.real_t = round_time((pygame.time.get_ticks() / 1000) - self.last_reset - self.paused_time)
        self.sim_t = self.n_updates * SAMPLING_TIME
        if self.n_updates % N_UPDATES_AVERAGE_TIME == 0: self.previous_updates_time = self.updates_time; self.updates_time = (pygame.time.get_ticks() / 1000) - self.last_reset - self.paused_time

    def control_robot(self):
        # Every ROBOT_SAMPLING_TIME we update the robot's action, but each SAMPLING_TIME we update its position
        if is_multiple(self.sim_t, ROBOT_SAMPLING_TIME): 
            if not self.robot_controlled:
                if not hasattr(self.robot, "diff_drive"): self.robot.mount_differential_drive(1)
                if pygame.key.get_pressed()[pygame.K_UP]: self.robot.move_with_keys('up', ROBOT_SAMPLING_TIME)
                if pygame.key.get_pressed()[pygame.K_DOWN]: self.robot.move_with_keys('down', ROBOT_SAMPLING_TIME)
                if pygame.key.get_pressed()[pygame.K_LEFT]: self.robot.move_with_keys('left', ROBOT_SAMPLING_TIME)
                if pygame.key.get_pressed()[pygame.K_RIGHT]: self.robot.move_with_keys('right', ROBOT_SAMPLING_TIME)
                self.robot.position, self.robot.yaw = self.robot.diff_drive.update_pose(self.robot.position, self.robot.yaw, SAMPLING_TIME)
                self.robot.check_collisions(self.humans, self.walls)
            else:
                if self.robot_crowdnav_policy:
                    if hasattr(self.robot.policy, "with_theta_and_omega_visible") and self.robot.policy.with_theta_and_omega_visible: ob = [human.get_observable_state(visible_theta_and_omega=True) for human in self.humans]
                    else: ob = [human.get_observable_state() for human in self.humans]
                    action = self.robot.act(ob)
                    self.robot.step(action, SAMPLING_TIME)
                    if (np.linalg.norm(self.robot.position - self.robot.get_goal_position()) < self.robot.radius) and (len(self.robot.goals)>1):
                        robot_goal = self.robot.goals[0]
                        self.robot.goals.remove(robot_goal)
                        self.robot.goals.append(robot_goal)
                else:
                    if self.robot_env_same_timestep: self.motion_model_manager.update_robot(self.sim_t, SAMPLING_TIME)
                    else:
                        self.motion_model_manager.update_robot_pose(SAMPLING_TIME)
                        self.motion_model_manager.update_robot(self.sim_t, ROBOT_SAMPLING_TIME, just_velocities=True)
                self.updated = True
            if self.robot.laser is not None: measuremenets = self.robot.get_laser_readings(self.humans, self.walls)
        else: 
            if not self.robot_controlled: self.robot.position, self.robot.yaw = self.robot.diff_drive.update_pose(self.robot.position, self.robot.yaw, SAMPLING_TIME)
            else: self.motion_model_manager.update_robot_pose(SAMPLING_TIME) # We just update the position

    def rewind_states(self, self_states=True, human_states=None, robot_poses=None):
        if self_states:
            if len(self.human_states) > 0:
                self.n_updates -= 1
                state = self.human_states[self.n_updates]
                self.motion_model_manager.set_human_states(state)
                self.human_states = self.human_states[:-1]
                if self.insert_robot: 
                    self.motion_model_manager.set_robot_state(self.robot_states[self.n_updates])
                    self.robot_states = self.robot_states[:-1]
        else:
            if self.n_updates > 0:
                self.n_updates -= 1
                if robot_poses is not None: self.robot.set_pose(robot_poses[self.n_updates])
                self.motion_model_manager.set_human_states(human_states[self.n_updates], just_visual=True)

    def save_simulation_states(self):
        if self.motion_model_manager.headed: self.human_states = np.append(self.human_states, [self.motion_model_manager.get_human_states(include_goal=True, headed= True)], axis=0)
        else: self.human_states = np.append(self.human_states, [self.motion_model_manager.get_human_states(include_goal=True, headed= False)], axis=0)
        if self.insert_robot and self.robot.headed: self.robot_states = np.append(self.robot_states, [self.motion_model_manager.get_robot_state(headed=True)], axis=0)
        elif self.insert_robot and not(self.robot.headed): self.robot_states = np.append(self.robot_states, [self.motion_model_manager.get_robot_state(headed=False)], axis=0)
        else: pass

    def run_live(self):
        self.active = True
        if self.motion_model_manager.headed: self.human_states = np.array([self.motion_model_manager.get_human_states(include_goal=True, headed= True)], dtype=PRECISION)
        else: self.human_states = np.array([self.motion_model_manager.get_human_states(include_goal=True, headed= False)], dtype=PRECISION)
        if self.insert_robot and self.robot.headed: self.robot_states = np.array([self.motion_model_manager.get_robot_state(headed=True)], dtype=PRECISION)
        elif self.insert_robot and not(self.robot.headed): self.robot_states = np.array([self.motion_model_manager.get_robot_state(headed=False)], dtype=PRECISION)
        while self.active:
            if not self.paused:
                self.update()
                if not self.headless: self.render_sim()
                self.save_simulation_states()
            else:
                if not self.headless: 
                    self.render_sim()
                    # Rewind
                    if pygame.key.get_pressed()[pygame.K_z]: 
                        r_is_pressed = True
                        while r_is_pressed:
                            self.rewind_states()
                            self.render_sim()
                            pygame.event.get(); r_is_pressed = pygame.key.get_pressed()[pygame.K_z]
                    # Reset
                    if pygame.key.get_pressed()[pygame.K_r]:
                        self.reset_sim()
                        if self.motion_model_manager.headed: self.human_states = np.array([self.motion_model_manager.get_human_states(include_goal=True, headed= True)], dtype=PRECISION)
                        else: self.human_states = np.array([self.motion_model_manager.get_human_states(include_goal=True, headed= False)], dtype=PRECISION)
                        if self.insert_robot and self.robot.headed: self.robot_states = np.array([self.motion_model_manager.get_robot_state(headed=True)], dtype=PRECISION)
                        elif self.insert_robot and not(self.robot.headed): self.robot_states = np.array([self.motion_model_manager.get_robot_state(headed=False)], dtype=PRECISION)
                    # Speed up (or Resume)
                    if pygame.key.get_pressed()[pygame.K_s]:
                        self.paused_time += round_time((pygame.time.get_ticks() / 1000) - self.last_pause_start)
                        s_is_pressed = True
                        while s_is_pressed:
                            self.update()
                            self.render_sim()
                            self.save_simulation_states()
                            pygame.event.get(); s_is_pressed = pygame.key.get_pressed()[pygame.K_s]
                        self.last_pause_start = round_time(pygame.time.get_ticks() / 1000)
            self.event_handling()
            self.clock.tick(MAX_FPS)
    
    def run_from_precomputed_states(self, human_states, robot_poses=None):
        self.config_data["headless"] = False
        self.reset_sim(restart_gui=True)
        self.paused = True; self.last_pause_start = round_time(pygame.time.get_ticks() / 1000)
        while self.n_updates < len(human_states)-1:
            if not self.paused:
                if robot_poses is not None: self.robot.set_pose(robot_poses[self.n_updates])
                self.motion_model_manager.set_human_states(human_states[self.n_updates], just_visual=True)
                self.n_updates += 1
                self.update_times()
                self.render_sim()
            else:
                self.render_sim()
                # Rewind
                if pygame.key.get_pressed()[pygame.K_z]: 
                    r_is_pressed = True
                    while r_is_pressed:
                        self.rewind_states(self_states=False, human_states=human_states, robot_poses=robot_poses)
                        self.render_sim()
                        pygame.event.get(); r_is_pressed = pygame.key.get_pressed()[pygame.K_z]
                # Reset
                if pygame.key.get_pressed()[pygame.K_r]: 
                    self.n_updates = 0
                    self.paused = False
                    continue
                # Speed up (or Resume)
                if pygame.key.get_pressed()[pygame.K_s]:
                    self.paused_time += round_time((pygame.time.get_ticks() / 1000) - self.last_pause_start)
                    s_is_pressed = True
                    while (s_is_pressed) and (self.n_updates < len(human_states)-1):
                        if robot_poses is not None: self.robot.set_pose(robot_poses[self.n_updates])
                        self.motion_model_manager.set_human_states(human_states[self.n_updates], just_visual=True)
                        self.n_updates += 1
                        self.update_times()
                        self.render_sim()
                        pygame.event.get(); s_is_pressed = pygame.key.get_pressed()[pygame.K_s]
                    self.last_pause_start = round_time(pygame.time.get_ticks() / 1000)
            self.event_handling()
            self.clock.tick(MAX_FPS)
        pygame.quit(); self.pygame_init = False

    def event_handling(self):
        for event in pygame.event.get():
            # Exit
            if event.type == pygame.QUIT:
                self.active = False
                self.pygame_init = False
                pygame.quit()
            # Pause
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.paused = not self.paused
                if self.paused: self.last_pause_start = round_time(pygame.time.get_ticks() / 1000)
                else: self.paused_time += round_time((pygame.time.get_ticks() / 1000) - self.last_pause_start)
            # Reset scroll and zoom
            if event.type == pygame.KEYDOWN and event.key == pygame.K_o:
                self.scroll = np.array([-350.0,+350.0], dtype=np.float16) # Default [0,0]
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

    def run_k_steps(self, steps, quit=True, additional_info=False, stop_when_collision_or_goal=False, save_states_time_step=SAMPLING_TIME):
        """
        Method used to run k steps in the simulator.

        params:
        - steps: number of steps to run.
        - quit: wether to quit pygame at the end of the method or not.
        - additional_info: if true, collision, success, truncated are computed based on the robot performance.
        - stop_when_collision_or_goal: if true, the simulation is stopped if the robot collides or reaches goal
        """
        # Check if the time_step to save states is a multiple of the environment sampling time
        if not is_multiple(save_states_time_step, SAMPLING_TIME): raise ValueError(f"Time step to save states must be a multiple of environment sampling time: {SAMPLING_TIME}")
        # Variables where to save the states
        human_states = np.array([self.motion_model_manager.get_human_states()], dtype=PRECISION)
        if self.insert_robot: robot_states = np.array([self.motion_model_manager.get_robot_state()], dtype=PRECISION)
        # Additional infos
        if stop_when_collision_or_goal and not additional_info: raise ValueError("Cannot stop the episode if you don't compute additional info")
        collision = False
        success = False
        truncated = False
        time_to_goal = None
        # Main loop
        for step in range(steps):
            if stop_when_collision_or_goal and (collision or success): break
            self.update()
            if not self.headless: self.render_sim()
            save_states = is_multiple(self.sim_t, save_states_time_step)
            if save_states: human_states = np.append(human_states, [self.motion_model_manager.get_human_states()], axis=0)
            if self.insert_robot: 
                if save_states: robot_states = np.append(robot_states, [self.motion_model_manager.get_robot_state()], axis=0)
                if additional_info:
                    # Collision
                    for human in self.humans:
                        if np.linalg.norm(human.position - self.robot.position) < (human.radius + self.robot.radius): collision = True
                    # Success
                    if len(robot_states) > 1 and not np.array_equal(robot_states[len(robot_states)-1][6:8],robot_states[len(robot_states)-2][6:8]): # If the robot's goal has changed
                        time_to_goal = self.n_updates * SAMPLING_TIME
                        success = True
                    # Truncated
                    if step == steps - 1 and not collision and not success: truncated = True
        # Closure
        if not self.headless and quit: pygame.quit(); self.pygame_init = False
        if self.insert_robot and additional_info: return human_states, robot_states, collision, time_to_goal, success, truncated
        elif self.insert_robot and not additional_info: return human_states, robot_states
        else: return human_states

    def run_single_test(self, n_updates):
        start_time = round_time((pygame.time.get_ticks() / 1000))
        human_states = self.run_k_steps(n_updates, quit=False, save_states_time_step=SAMPLING_TIME)
        test_time = round_time((pygame.time.get_ticks() / 1000) - start_time - self.paused_time)
        return human_states, test_time

    def run_multiple_models_test(self, final_time=40, models=MOTION_MODELS, plot_sample_time=3, two_integrations=False):
        n_updates = int(final_time / SAMPLING_TIME)
        if not two_integrations:
            self.human_states = np.empty((len(models),n_updates+1,len(self.humans),N_GENERAL_STATES), dtype=PRECISION)
            test_times = np.empty((len(models),), dtype=PRECISION)
            for i in range(len(models)):
                self.reset_sim()
                self.motion_model_manager.set_human_motion_model(models[i])
                self.human_states[i], test_times[i] = self.run_single_test(n_updates)
                figure, ax = plt.subplots(figsize=(7,7))
                figure.tight_layout()
                # figure.suptitle(f'Human agents\' position over simulation | T = {final_time} | dt = {round(SAMPLING_TIME, 4)} | Model = {models[i]}')
                if self.motion_model_manager.runge_kutta == False: integration_title = "Euler"
                else: integration_title = "Runge-Kutta-45"
                # ax.set(xlabel='X',ylabel='Y',title=f'{integration_title} | Elapsed time = {test_times[i]}',xlim=[0,REAL_SIZE],ylim=[0,REAL_SIZE])
                self.plot_agents_position_with_sample(ax,self.human_states[i],plot_sample_time,models[i])
        else:
            self.human_states = np.empty((len(models),2,n_updates+1,len(self.humans),N_GENERAL_STATES), dtype=PRECISION)
            test_times = np.empty((len(models),2), dtype=PRECISION)
            for i in range(len(models)):
                self.reset_sim()
                self.motion_model_manager.set_human_motion_model(models[i])
                self.motion_model_manager.runge_kutta = False
                self.human_states[i,0], test_times[i,0] = self.run_single_test(n_updates)
                self.reset_sim()
                self.motion_model_manager.set_human_motion_model(models[i])
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
        self.human_states = np.empty((2,n_updates+1,len(self.humans),N_GENERAL_STATES), dtype=PRECISION)
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

    def run_and_plot_trajectories_humans_and_robot(self, final_time=40, plot_sample_time=3, ax=None, show=True):
        n_updates = int(final_time / SAMPLING_TIME)
        human_states, robot_states = self.run_k_steps(n_updates, quit=False, additional_info=False, save_states_time_step=SAMPLING_TIME)
        if ax is None: 
            figure, ax = plt.subplots()
            figure.tight_layout()
            if self.mode == "circular_crossing": figure.set_size_inches(10,10)
            elif self.mode == "parallel_traffic": figure.set_size_inches(10,5)
            # figure.suptitle(f'Human agents\' position over simulation | T = {final_time} | dt = {round(SAMPLING_TIME, 4)} | Model = {self.motion_model}')
        self.plot_humans_and_robot_trajectories(ax, human_states, robot_states, plot_sample_time=plot_sample_time, show=show)

    def plot_humans_and_robot_trajectories(self, ax, human_states, robot_states=None, plot_sample_time=3, show=True, plot_goals=True, dt_between_states=SAMPLING_TIME):
        ax.set(xlabel='X',ylabel='Y',xlim=[0,REAL_SIZE],ylim=[0,REAL_SIZE])
        self.plot_agents_position_with_sample(ax,human_states,plot_sample_time,self.motion_model, plot_goals=plot_goals, dt_between_states=dt_between_states)
        if self.insert_robot: self.plot_agents_position_with_sample(ax,robot_states,plot_sample_time,self.motion_model, is_robot=True, plot_goals=plot_goals, dt_between_states=dt_between_states)
        if not self.headless: pygame.quit(); self.pygame_init = False
        if show: plt.show()

    def print_walls_on_plot(self, ax):
        for i in range(len(self.walls)):
                ax.fill(self.walls.sprites()[i].vertices[:,0], self.walls.sprites()[i].vertices[:,1], facecolor='black', edgecolor='black')

    def plot_agents_position_with_sample(self, ax, states, plot_sample_time:float, model:str, is_robot=False, plot_goals=True, dt_between_states=SAMPLING_TIME):
        ax.axis('equal')
        self.print_walls_on_plot(ax)
        if is_robot:
            color = "red"
            ax.plot(states[:,0],states[:,1], color=color, linewidth=0.5, zorder=0)
            for k in range(0,len(states),int(plot_sample_time / dt_between_states)):
                circle = plt.Circle((states[k,0],states[k,1]),self.robot.radius, edgecolor="black", facecolor=color, fill=True, zorder=1)
                ax.add_patch(circle)
                size = 15 if (k*dt_between_states).is_integer() else 10
                num = int(k*dt_between_states) if (k*dt_between_states).is_integer() else (k*dt_between_states)
                ax.text(states[k,0],states[k,1], f"{num}", color="black", va="center", ha="center", size=size, zorder=1, weight='bold')
                if plot_goals:
                    goals = np.array(self.robot.goals, dtype=PRECISION).copy()
                    for k in range(len(goals)):
                        if goals[k,0] == states[0,0] and goals[k,1] == states[0,1]: 
                            goals = np.delete(goals, k, 0)
                            break
                    ax.scatter(goals[:,0], goals[:,1], marker="*", color=color, zorder=2)
        else:
            for j in range(len(self.humans)):
                color_idx = j % len(COLORS)
                ax.plot(states[:,j,0],states[:,j,1], color=COLORS[color_idx], linewidth=0.5, zorder=0)
                for k in range(0,len(states),int(plot_sample_time / dt_between_states)):
                    if "hsfm" in model:
                        head = plt.Circle((states[k,j,0] + math.cos(states[k,j,2]) * self.humans[j].radius, states[k,j,1] + math.sin(states[k,j,2]) * self.humans[j].radius), 0.1, color=COLORS[color_idx], zorder=1)
                        ax.add_patch(head)
                    circle = plt.Circle((states[k,j,0],states[k,j,1]),self.humans[j].radius, edgecolor=COLORS[color_idx], facecolor="white", fill=True, zorder=1)
                    ax.add_patch(circle)
                    size = 15 if (k*dt_between_states).is_integer() else 10
                    num = int(k*dt_between_states) if (k*dt_between_states).is_integer() else (k*dt_between_states)
                    ax.text(states[k,j,0],states[k,j,1], f"{num}", color=COLORS[color_idx], va="center", ha="center", size=size, zorder=1, weight='bold')
                if plot_goals:
                    goals = np.array(self.humans[j].goals, dtype=PRECISION).copy()
                    for k in range(len(goals)):
                        if goals[k,0] == states[0,j,0] and goals[k,1] == states[0,j,1]: 
                            goals = np.delete(goals, k, 0)
                            break
                    ax.scatter(goals[:,0], goals[:,1], marker="*", color=COLORS[color_idx], zorder=2)

    def plot_agents_trajectory(self, ax, human_states):
        ax.axis('equal')
        self.print_walls_on_plot(ax)
        for i in range(len(self.humans)):
            ax.plot(human_states[:,i,0],human_states[:,i,1])

    def set_human_motion_model_as_robot_policy(self, policy_name, runge_kutta):
        """
        Sets a human motion model as the policy the robot will follow to move towards its goals.

        params:
        - policy_name (str): name of the policy.
        - runge_kutta (bool): if True, integration uses Runge-Kutta-45. Only valid for NON-crowdnav policies.
        - safety_space (float): Only used for ORCA, it's an additional constant added to the radius of each agent to increase security in the trajectories
        
        output: None
        """
        self.motion_model_manager.set_robot_motion_model(policy_name, runge_kutta)

    ### METHODS FOR CROWDNAV ROBOT POLICIES

    def set_robot_policy(self, policy_name:str, runge_kutta=False, crowdnav_policy=False, model_dir=None, il=False):
        """"
        Sets the policy the robot will follow to move towards its goals.

        params:
        - policy_name (str): name of the policy.
        - runge_kutta (bool): if True, integration uses Runge-Kutta-45. Only valid for NON-crowdnav policies.
        - crowdnav_policy (bool): specify if the policy is a crowdnav one (True) or a human motion model of MotionModelManager (false)
        - model_dir (str): specifies the directory where the crowdnav model is stored
        - il (bool): if True, the Imitation Learning model is taken. Else, the reinforcement learning model is selected.

        output: None
        """
        self.robot_controlled = True
        if crowdnav_policy:
            # Set policy
            policy = policy_factory[policy_name]()
            if model_dir is not None:
                if not il: model_weights = os.path.join(model_dir, 'rl_model.pth')
                else: model_weights = os.path.join(model_dir, 'il_model.pth')
                policy_config_file = os.path.join(model_dir, os.path.basename('policy.config'))
                policy_config = configparser.RawConfigParser()
                policy_config.read(policy_config_file)
                policy.configure(policy_config)
            if policy.trainable: 
                policy.get_model().load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))
                if policy_name == 'lstm_rl': policy.model.lstm.eval()
            policy.set_phase('test')
            policy.set_device('cpu')
            policy.set_env(self)
            # Configure robot
            if model_dir is not None:
                env_config_file = os.path.join(model_dir, os.path.basename('env.config'))
                env_config = configparser.RawConfigParser()
                env_config.read(env_config_file)
                self.robot.configure(env_config, 'robot')
                self.time_limit = env_config.getint('env', 'time_limit')
                self.success_reward = env_config.getfloat('reward', 'success_reward')
                self.collision_penalty = env_config.getfloat('reward', 'collision_penalty')
                self.discomfort_dist = env_config.getfloat('reward', 'discomfort_dist')
                self.discomfort_penalty_factor = env_config.getfloat('reward', 'discomfort_penalty_factor')
            else:
                self.robot.desired_speed = 1
            self.robot.set_policy(policy)
            self.robot.policy.time_step = ROBOT_SAMPLING_TIME
            self.robot_crowdnav_policy = True
        else: 
            self.set_human_motion_model_as_robot_policy(policy_name, runge_kutta)
            self.robot_crowdnav_policy = False
        if self.parallelize_robot: 
            if crowdnav_policy and hasattr(self.robot, "policy"): self.robot.policy.parallelize = True
            self.robot.parallelize = True

    def transform_human_states(self, state:np.array, theta_and_omega_visible=False):
        """
        Transforms the states from the configuration of MotionModelManager to the
        configuration used in CrowdNav.

        params:
        - state: np.array((n_humans,4))
        - theta_and_omega_visible: bool indicating if the theta and omega of the humans are visible in the state
        
        output:
        - output_state: list(ObservableState) or list(ObservableStateHeaded) [x, y, yaw, Vx, Vy, Omega, Gx, Gy]
        """
        output_state = []
        if theta_and_omega_visible: 
            for i, human_state in enumerate(state): output_state.append(ObservableStateHeaded(human_state[0],human_state[1],human_state[3],human_state[4],self.humans[i].radius, human_state[2], human_state[5]))
        else: 
            for i, human_state in enumerate(state): output_state.append(ObservableState(human_state[0],human_state[1],human_state[2],human_state[3],self.humans[i].radius))
        return output_state

    def collision_detection_and_reaching_goal(self, action, time_step):
        """"
        This function, based on the next robot action, computes if there will be a collision 
        (assuming that humans move at a constant velocity which is the last observed one) and if the robot
        reaches the goal.

        params:
        - action: robot action (either a crowdnav action or a np.array)
        - time_step: time step to use for the computation

        returns:
        - collision: bool telling if the robot will collide
        - dmin: minimum distance of the robot from humans
        - reaching_goal: bool telling if the robot will reach the goal
        """
        # Collision detection
        dmin = float('inf')
        collision = False
        for human in self.humans:
            difference = human.position - self.robot.position
            if isinstance(action, np.ndarray):
                robot_velocity = action
            else:
                if self.robot.kinematics == 'holonomic': robot_velocity = np.array([action.vx, action.vy])
                else: robot_velocity = np.array([action.v * np.cos(action.r + self.robot.theta), action.v * np.sin(action.r + self.robot.theta)])
            velocity_difference = human.linear_velocity - robot_velocity
            e = difference + velocity_difference * time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(difference[0], difference[1], e[0], e[1], 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0: collision = True; break
            elif (closest_dist >= 0) and (closest_dist < dmin): dmin = closest_dist
        # Check if reaching the goal
        if isinstance(action, np.ndarray): end_position = self.robot.position + action * time_step
        else: end_position = self.robot.compute_position(action, time_step)
        reaching_goal = np.linalg.norm(end_position - self.robot.get_goal_position()) < self.robot.radius
        return collision, dmin, reaching_goal

    def compute_reward_and_infos(self, collision, dmin, reaching_goal, current_time, time_step):
        """"
        Computes the reward, if the episode is truncated or terminated and info.

        params:
        - collision: bool telling if the robot will collide
        - dmin: minimum distance of the robot from humans
        - reaching_goal: bool telling if the robot will reach the goal
        - current_time: float indicating the current time
        - time_step: time step to use for the computation

        returns:
        - reward (float)
        - terminated (bool)
        - truncated (bool)
        - info (one object between Timeout(), Collision(), ReachGoal(), Danger())
        """
        # Compute Reward, truncated, terminated, and info
        if current_time >= self.time_limit - 1:
            reward = 0
            truncated = True
            terminated = False
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            truncated = False
            terminated = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            truncated = False
            terminated = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * time_step # adjust the reward based on FPS
            truncated = False
            terminated = False
            info = Danger(dmin)
        else:
            reward = 0
            truncated = False
            terminated = False
            info = Nothing()
        return reward, terminated, truncated, info

    def onestep_lookahead(self, action, time_step=ROBOT_SAMPLING_TIME):
        """
        This method is just required to use the trained policies of the robot.
        """
        # Detect collision and reaching goal
        collision, dmin, reaching_goal = self.collision_detection_and_reaching_goal(action, time_step)
        # Compute reward, terminated, truncated, and info
        reward, _, _, _ = self.compute_reward_and_infos(collision, dmin, reaching_goal, self.sim_t, time_step)
        # Next humans' state computation
        if not self.updated: ob = self.last_observation.copy()
        else:
            if self.robot.policy.query_env:
                if self.robot.policy.with_theta_and_omega_visible: ob = self.transform_human_states(self.motion_model_manager.get_next_human_observable_states(time_step, theta_and_omega_visible=True), theta_and_omega_visible=True)
                else: ob = self.transform_human_states(self.motion_model_manager.get_next_human_observable_states(time_step))
            else:
                ob = self.propagate_humans_state_with_constant_velocity_model(time_step)
            self.last_observation = ob.copy()
        self.updated = False
        return ob, reward

    def propagate_humans_state_with_constant_velocity_model(self, time_step):
        """
        This method propagates the humans' states with the constant velocity model.
        """
        propagated_obs = []
        if self.robot.policy.with_theta_and_omega_visible:
            for human in self.humans:
                next_position = human.position + human.linear_velocity * time_step
                next_angle = human.yaw + human.angular_velocity * time_step
                propagated_obs.append(ObservableStateHeaded(*next_position,*human.linear_velocity,human.radius, next_angle, human.angular_velocity))
        else:
            for human in self.humans:
                next_position = human.position + human.linear_velocity * time_step
                propagated_obs.append(ObservableState(*next_position,*human.linear_velocity,human.radius))
        return propagated_obs
    
