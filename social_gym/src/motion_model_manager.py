from social_gym.src.utils import bound_angle
from social_gym.src.human_agent import HumanAgent
from social_gym.src.robot_agent import RobotAgent
from social_gym.src.agent import Agent
from social_gym.src.social_momentum import *
from social_gym.src.forces_parallel import update_humans_parallel
from scipy.integrate import solve_ivp
import rvo2
import socialforce
import numpy as np

N_GENERAL_STATES = 8
N_HEADED_STATES = 6
N_NOT_HEADED_STATES = 4
ORCA_DEFAULTS = [10,10,5,5] # neighbor_dist, max_neighbors, time_horizon, time_horizon_obstacles
SFMS = ["sfm_helbing","sfm_guo","sfm_moussaid",
        "hsfm_farina","hsfm_guo","hsfm_moussaid",
        "hsfm_new","hsfm_new_guo","hsfm_new_moussaid"]

class Group:
    def __init__(self):
        self.group_agents = []
        self.center = np.array([0.0,0.0],dtype=np.float64)

    def append_agent(self, agent:int):
        self.group_agents.append(agent)

    def compute_center(self):
        self.center /= len(self.group_agents)

    def num_agents(self):
        return len(self.group_agents)

class MotionModelManager:
    def __init__(self, motion_model_title:str, consider_robot:bool, runge_kutta:bool, humans:list[HumanAgent], robot:RobotAgent, walls:list, parallelize = False):
        self.consider_robot = consider_robot
        self.runge_kutta = runge_kutta
        self.update_targets = True
        self.humans = humans
        self.robot = robot
        self.walls = walls
        self.parallel = parallelize
        self.orca = False # ORCA bool controller
        self.sm = False # Social Momentum bool controller
        self.sf = False # CrowdNav SocialForce bool controller
        if runge_kutta and motion_model_title=="orca": raise NotImplementedError
        self.parallel_traffic_humans_respawn = False
        self.set_human_motion_model(motion_model_title)
        self.robot_motion_model_title = None # Robot policy can be set later

    ### METHODS FOR BOTH HUMANS AND ROBOT
    
    def bound_velocity(self, velocity:np.array, desired_speed:float):
        velocity_norm = np.linalg.norm(velocity)
        if (velocity_norm > desired_speed): velocity = (velocity / velocity_norm) * desired_speed
        return velocity
    
    def rewind_goals(self, agent:Agent, goal:list):
        if goal not in agent.goals: agent.set_goals([goal]) # If the goal is not in the list, we overwrite the goal list (used for parallel traffic scenario, where goals list is dynamic)
        else:
            if ((agent.goals) and (agent.goals[0] != goal)):
                while agent.goals[0] != goal:
                    goal_back = agent.goals[0]
                    agent.goals.remove(goal_back)
                    agent.goals.append(goal_back)
    
    def update_goals(self, agent:Agent):
        if ((agent.goals) and (np.linalg.norm(agent.goals[0] - agent.position) < agent.radius)):
            goal = agent.goals[0]
            agent.goals.remove(goal)
            agent.goals.append(goal)

    def euler_not_headed_single_agent_update(self, agent:Agent, dt:float, include_mass:bool, just_velocities=False):
        if not just_velocities: agent.position += agent.linear_velocity * dt
        if include_mass: agent.linear_velocity += (agent.global_force / agent.mass) * dt
        else: agent.linear_velocity += agent.global_force * dt
        agent.linear_velocity = self.bound_velocity(agent.linear_velocity, agent.desired_speed)

    def euler_headed_single_agent_update(self, agent:Agent, dt:float, just_velocities=False):
        if not just_velocities:
            agent.position += agent.linear_velocity * dt
            agent.yaw = bound_angle(agent.yaw + agent.angular_velocity * dt)
        agent.body_velocity += (agent.global_force / agent.mass) * dt
        agent.angular_velocity += (agent.torque_force / agent.inertia) * dt
        agent.body_velocity = self.bound_velocity(agent.body_velocity, agent.desired_speed)
        self.headed_agent_update_linear_velocity(agent) # We update the linear velocity here becaue CrowdNav uses linear velocity for Observable and Full states      

    def set_new_not_headed_state_from_rk45_solution(self, agent:Agent, y:np.array):
        if len(y) != N_NOT_HEADED_STATES: raise ValueError(f'The passed solution size is not correct, it should be {N_NOT_HEADED_STATES}')
        agent.position[0] = y[0]
        agent.position[1] = y[1]
        agent.linear_velocity[0] = y[2]
        agent.linear_velocity[1] = y[3]
        agent.linear_velocity = self.bound_velocity(agent.linear_velocity, agent.desired_speed)

    def set_new_headed_state_from_rk45_solution(self, agent:Agent, y:np.array):
        if len(y) != N_HEADED_STATES: raise ValueError(f'The passed solution size is not correct, it should be {N_HEADED_STATES}')
        agent.position[0] = y[0]
        agent.position[1] = y[1]
        agent.yaw = bound_angle(y[2])
        agent.body_velocity[0] = y[3]
        agent.body_velocity[1] = y[4]
        agent.body_velocity = self.bound_velocity(agent.body_velocity, agent.desired_speed)
        agent.angular_velocity = y[5]

    def set_state_orca(self, agent_idx:int, robot_sim=False):
        if not robot_sim: # Humans ORCA simulation
            if agent_idx < len(self.humans):
                self.sim.setAgentPosition(self.agents[agent_idx], (self.humans[agent_idx].position[0], self.humans[agent_idx].position[1]))
                self.sim.setAgentVelocity(self.agents[agent_idx], (self.humans[agent_idx].linear_velocity[0], self.humans[agent_idx].linear_velocity[1]))
                self.update_goals_orca(agent_idx, robot_sim=False)
            else:
                self.sim.setAgentPosition(self.agents[agent_idx], (self.robot.position[0], self.robot.position[1]))
                self.sim.setAgentVelocity(self.agents[agent_idx], (self.robot.linear_velocity[0], self.robot.linear_velocity[1]))
                self.update_goals_orca(agent_idx, robot_sim=False, robot=True)
        else: # Robot ORCA simulation
            if agent_idx < len(self.humans):
                self.robot_sim.setAgentPosition(self.robot_sim_agents[agent_idx], (self.humans[agent_idx].position[0], self.humans[agent_idx].position[1]))
                self.robot_sim.setAgentVelocity(self.robot_sim_agents[agent_idx], (self.humans[agent_idx].linear_velocity[0], self.humans[agent_idx].linear_velocity[1]))
                self.robot_sim.setAgentPrefVelocity(self.robot_sim_agents[agent_idx], (0,0))
            else:
                self.robot_sim.setAgentPosition(self.robot_sim_agents[agent_idx], (self.robot.position[0], self.robot.position[1]))
                self.robot_sim.setAgentVelocity(self.robot_sim_agents[agent_idx], (self.robot.linear_velocity[0], self.robot.linear_velocity[1]))
                self.update_goals_orca(agent_idx, robot_sim=True, robot=True)

    def update_goals_orca(self, agent_idx:int, robot_sim=False, robot=False):
        if not robot:
            if self.update_targets: self.update_goals(self.humans[agent_idx])
            goal_position = np.array(self.humans[agent_idx].goals[0], dtype=np.float64)
            difference = goal_position - self.humans[agent_idx].position
            norm = np.linalg.norm(difference)
            agent_pref_speed = difference / norm if norm > self.humans[agent_idx].desired_speed else difference
            if not robot_sim: self.sim.setAgentPrefVelocity(self.agents[agent_idx], (agent_pref_speed[0], agent_pref_speed[1]))
            else: self.robot_sim.setAgentPrefVelocity(self.robot_sim_agents[agent_idx], (agent_pref_speed[0], agent_pref_speed[1]))
        else:
            if self.update_targets: self.update_goals(self.robot) # Pay attention to this
            goal_position = np.array(self.robot.goals[0], dtype=np.float64)
            difference = goal_position - self.robot.position
            norm = np.linalg.norm(difference)
            robot_pref_speed = difference / norm if norm > self.robot.desired_speed else difference
            if not robot_sim: self.sim.setAgentPrefVelocity(self.agents[agent_idx], (robot_pref_speed[0], robot_pref_speed[1]))
            else: self.robot_sim.setAgentPrefVelocity(self.robot_sim_agents[agent_idx], (robot_pref_speed[0], robot_pref_speed[1]))

    def headed_agent_update_linear_velocity(self, agent:Agent):
        agent.compute_rotational_matrix()
        agent.linear_velocity = np.matmul(agent.rotational_matrix, agent.body_velocity)

    def set_safety_space(self, safety_space:float):
        # Humans safety space
        if self.motion_model_title is not None:
            if "sfm" in self.motion_model_title:
                for iii, human in enumerate(self.humans): 
                    human.safety_space = 0.01 + safety_space
                    if self.parallel: self.safety_space[iii] = 0.01 + safety_space
            elif self.motion_model_title == "orca":
                if self.sim is not None:
                    for i, agent in enumerate(self.agents):
                        if i < len(self.humans): self.sim.setAgentRadius(i, self.humans[i].radius + 0.01 + safety_space)
                        else: self.sim.setAgentRadius(i, self.robot.radius + 0.01 + safety_space)
            else: raise NotImplementedError(f"Model {self.motion_model_title} is not implemented for humans")
        # Robot safety space
        if self.robot_motion_model_title is not None:
            if "sfm" in self.robot_motion_model_title:
                self.robot.safety_space = 0.01 + safety_space
                if self.parallel and self.consider_robot: self.safety_space[len(self.humans)] = 0.01 + safety_space
            elif self.robot_motion_model_title == "orca":
                if self.robot_sim is not None: # Add safety space to Robot ORCA simulation
                    for i, agent in enumerate(self.robot_sim_agents):
                        if i < len(self.humans): self.robot_sim.setAgentRadius(i, self.humans[i].radius + 0.01 + safety_space)
                        else: self.robot_sim.setAgentRadius(i, self.robot.radius + 0.01 + safety_space)
            else: raise NotImplementedError(f"Model {self.motion_model_title} is not implemented for robot")

    def check_pair_agents_social_force_parameters(self, agent1:Agent, agent2:Agent):
        """
        Checks if the parameters of the motion models guiding the two agents are the same. 
        The agents should have the same motion model, otherwise this check does not make sense
        (different parameters).

        params:
        - agent1: First agent (both robot or human)
        - agent2: Second agent (both robot or human)

        output:
        bool indicating wether the parameters are equal or not.
        """
        if agent1.radius != agent2.radius: return False
        if agent1.mass != agent2.mass: return False
        if self.motion_model_title == "sfm_helbing" or self.motion_model_title == "hsfm_farina" or self.motion_model_title == "hsfm_new":
            # Ai, Bi, k1, k2
            if agent1.Ai != agent2.Ai: return False
            if agent1.Bi != agent2.Bi: return False
            if agent1.k1 != agent2.k1: return False
            if agent1.k2 != agent2.k2: return False
        elif self.motion_model_title == "sfm_guo" or self.motion_model_title == "hsfm_guo" or self.motion_model_title == "hsfm_new_guo": 
            # Ai, Bi, Ci, Di, k1, k2
            if agent1.Ai != agent2.Ai: return False
            if agent1.Bi != agent2.Bi: return False
            if agent1.Ci != agent2.Ci: return False
            if agent1.Di != agent2.Di: return False
            if agent1.k1 != agent2.k1: return False
            if agent1.k2 != agent2.k2: return False
        elif self.motion_model_title == "sfm_moussaid" or self.motion_model_title == "hsfm_moussaid" or self.motion_model_title == "hsfm_new_moussaid":  
            # agent_lambda, gamma, Ei, ns1, ns, k1, k2
            if agent1.agent_lambda != agent2.agent_lambda: return False
            if agent1.gamma != agent2.gamma: return False
            if agent1.Ei != agent2.Ei: return False
            if agent1.ns1 != agent2.ns1: return False
            if agent1.ns != agent2.ns: return False
            if agent1.k1 != agent2.k1: return False
            if agent1.k2 != agent2.k2: return False
        elif self.motion_model_title == "sfm_roboticsupo":
            # agent_lambda, agent_gamma, agent_nPrime, agent_n, social_weight
            if agent1.agent_lambda != agent2.agent_lambda: return False
            if agent1.agent_gamma != agent2.agent_gamma: return False
            if agent1.agent_nPrime != agent2.agent_nPrime: return False
            if agent1.agent_n != agent2.agent_n: return False
            if agent1.social_weight != agent2.social_weight: return False
        else: raise NotImplementedError(f"The {self.motion_model_title} model is not implemented")
        return True

    ### METHODS ONLY FOR HUMANS

    def set_human_motion_model(self, motion_model_title:str):
        self.motion_model_title = motion_model_title
        global compute_desired_force, compute_obstacle_force, compute_social_force, compute_group_force, compute_torque_force, compute_all_social_forces
        if self.motion_model_title == "sfm_helbing": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_helbing as compute_social_force, compute_all_social_forces, compute_group_force_dummy as compute_group_force; self.headed = False; self.include_mass = True; self.type = 0
        elif self.motion_model_title == "sfm_guo": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_guo as compute_obstacle_force, compute_social_force_guo as compute_social_force, compute_all_social_forces, compute_group_force_dummy as compute_group_force; self.headed = False; self.include_mass = True; self.type = 1
        elif self.motion_model_title == "sfm_moussaid": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_moussaid as compute_social_force, compute_all_social_forces, compute_group_force_dummy as compute_group_force; self.headed = False; self.include_mass = True; self.type = 2
        elif self.motion_model_title == "sfm_roboticsupo": from social_gym.src.forces import compute_desired_force_roboticsupo as compute_desired_force, compute_obstacle_force_roboticsupo as compute_obstacle_force, compute_social_force_roboticsupo as compute_social_force, compute_all_social_forces, compute_group_force_roboticsupo as compute_group_force; self.headed = False; self.include_mass = False; self.type = 3
        elif self.motion_model_title == "hsfm_farina": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_helbing as compute_social_force, compute_all_social_forces, compute_torque_force_farina as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True; self.type = 0
        elif self.motion_model_title == "hsfm_guo": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_guo as compute_obstacle_force, compute_social_force_guo as compute_social_force, compute_all_social_forces, compute_torque_force_farina as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True; self.type = 1
        elif self.motion_model_title == "hsfm_moussaid": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_moussaid as compute_social_force, compute_all_social_forces, compute_torque_force_farina as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True; self.type = 2
        elif self.motion_model_title == "hsfm_new": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_helbing as compute_social_force, compute_all_social_forces, compute_torque_force_new as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True; self.type = 0
        elif self.motion_model_title == "hsfm_new_guo": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_guo as compute_obstacle_force, compute_social_force_guo as compute_social_force, compute_all_social_forces, compute_torque_force_new as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True; self.type = 1
        elif self.motion_model_title == "hsfm_new_moussaid": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_moussaid as compute_social_force, compute_all_social_forces, compute_torque_force_new as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True; self.type = 2
        elif self.motion_model_title == "orca": 
            self.orca = True; self.headed = False; self.include_mass = False
            self.sim = rvo2.PyRVOSimulator(1/60, ORCA_DEFAULTS[0], ORCA_DEFAULTS[1], ORCA_DEFAULTS[2], ORCA_DEFAULTS[3], 0.3, 1) # dt is set at each update
            self.agents = []
            for i, agent in enumerate(self.humans):
                # Adding agents to the simulator: parameters ((poistion_x, position_y), agents_neighbor_dist, agents_max_neighbors, safety_time_horizon, safety_time_horizon_obstacles, agents_radius, agent_max_speed, (velocity_x, velocity_y))
                self.agents.append(self.sim.addAgent((agent.position[0], agent.position[1]), ORCA_DEFAULTS[0], ORCA_DEFAULTS[1], ORCA_DEFAULTS[2], ORCA_DEFAULTS[3], agent.radius + 0.01, agent.desired_speed, (agent.linear_velocity[0], agent.linear_velocity[1])))
                self.update_goals_orca(i)
            if self.consider_robot: self.agents.append(self.sim.addAgent((self.robot.position[0], self.robot.position[1]), ORCA_DEFAULTS[0], ORCA_DEFAULTS[1], ORCA_DEFAULTS[2], ORCA_DEFAULTS[3], self.robot.radius + 0.01, self.robot.desired_speed, (self.robot.linear_velocity[0], self.robot.linear_velocity[1])))
            for wall in self.walls:
                self.sim.addObstacle(list(wall.vertices))
            self.sim.processObstacles()
        elif self.motion_model_title == "social_momentum": 
            self.sm = True; self.headed = False; self.orca = False; self.include_mass = False
            self.n_actions = 20
            self.actions_angles = [((2*math.pi) / self.n_actions) * i for i in range(self.n_actions)]
            for human in self.humans: human.action_set = [np.array([math.cos(angle), math.sin(angle)], dtype=np.float64) * human.desired_speed for angle in self.actions_angles]
        elif self.motion_model_title == "socialforce": 
            # raise NotImplementedError("The model is currently being implemented and will be available soon.")
            self.sf = True; self.headed = False; self.orca = False; self.include_mass = False
            initial_state = np.array([[human.position[0], human.position[1], human.linear_velocity[0], human.linear_velocity[1], human.goals[0][0], human.goals[0][1]] for human in self.humans], dtype=np.float64)
            if self.consider_robot: initial_state = np.append(initial_state, [[self.robot.position[0], self.robot.position[1], self.robot.linear_velocity[0], self.robot.linear_velocity[1], self.robot.goals[0][0], self.robot.goals[0][1]]], axis = 0)
            self.sf_sim = socialforce.Simulator(initial_state, delta_t=0.25, v0 = 10, sigma = 0.3)
        else: raise Exception(f"The human motion model '{self.motion_model_title}' does not exist")
        if self.motion_model_title in SFMS: 
            for human in self.humans: human.set_parameters(motion_model_title)
            if self.parallel:
                self.sfm_type = SFMS.index(motion_model_title)
                self.safety_space = np.zeros(len(self.humans)+int(self.consider_robot), np.float64)
                self.states = np.array([human.get_safe_state() for human in self.humans], np.float64)
                if self.consider_robot: self.states = np.append(self.states, [self.robot.get_safe_state()], axis = 0)
                self.params = np.array([human.get_parameters(motion_model_title) for human in self.humans], np.float64)
                # Transform goals and obstacles in np.ndarray
                max_n_goals = np.max([len(human.goals) for human in self.humans])
                self.goals = np.empty((len(self.humans),max_n_goals,2), np.float64)
                self.goals[:,:,:] = np.NaN
                for i, human in enumerate(self.humans):
                    for j, goal in enumerate(human.goals):
                        self.goals[i,j] = np.array(goal.copy(), np.float64)
                if len(self.walls) > 0:
                    max_n_segments = np.max([len(obs.segments) for obs in self.walls])
                    self.obstacles = np.empty((len(self.walls),max_n_segments,2,2), np.float64)
                    self.obstacles[:,:,:,:] = np.NaN
                    for i, obs in enumerate(self.walls):
                        for j, segment in obs.segments.items():
                            self.obstacles[i,j,0] = np.array([segment[0][0], segment[0][1]], np.float64)
                            self.obstacles[i,j,1] = np.array([segment[1][0], segment[1][1]], np.float64)
                else: self.obstacles = None  
            # Check wether all agents parameters are equal, because if so, computation can be fastened
            self.all_equal_humans = True
            for i, human in enumerate(self.humans):
                for j, other_human in enumerate(self.humans):
                    if i >= j: continue
                    self.all_equal_humans = self.check_pair_agents_social_force_parameters(human, other_human)
                    if not self.all_equal_humans: break

    def get_human_states(self, include_goal=True, headed=False):
        if include_goal:
            if headed:
                # State: [x, y, yaw, BVx, BVy, Omega, Gx, Gy] - Pose (x,y,yaw), Body velocity (body_linear_x,body_linear_y,angular), and Goal (goal_x, goal_y)
                state = np.empty([len(self.humans),N_GENERAL_STATES],dtype=np.float64)
                for i in range(len(self.humans)):
                    human_state = np.array([self.humans[i].position[0],self.humans[i].position[1],self.humans[i].yaw,self.humans[i].body_velocity[0],self.humans[i].body_velocity[1],self.humans[i].angular_velocity,self.humans[i].goals[0][0],self.humans[i].goals[0][1]], dtype=np.float64)
                    state[i] = human_state
            else:
                # State: [x, y, yaw, Vx, Vy, Omega, Gx, Gy] - Pose (x,y,yaw), Velocity (linear_x,linear_y,angular), and Goal (goal_x, goal_y)
                state = np.empty([len(self.humans),N_GENERAL_STATES],dtype=np.float64)
                for i in range(len(self.humans)):
                    human_state = np.array([self.humans[i].position[0],self.humans[i].position[1],self.humans[i].yaw,self.humans[i].linear_velocity[0],self.humans[i].linear_velocity[1],self.humans[i].angular_velocity,self.humans[i].goals[0][0],self.humans[i].goals[0][1]], dtype=np.float64)
                    state[i] = human_state
        else:
            if headed:
                # State: [x, y, yaw, BVx, BVy, Omega] - Pose (x,y,yaw) and Velocity (body_linear_x,body_linear_y,angular)
                state = np.empty([len(self.humans),N_HEADED_STATES],dtype=np.float64)
                for i in range(len(self.humans)):
                    human_state = np.array([self.humans[i].position[0],self.humans[i].position[1],self.humans[i].yaw,self.humans[i].body_velocity[0],self.humans[i].body_velocity[1],self.humans[i].angular_velocity], dtype=np.float64)
                    state[i] = human_state
            else:
                # State: [x, y, Vx, Vy] - Position (x,y) and Velocity (linear_x,linear_y)
                state = np.empty([len(self.humans),N_NOT_HEADED_STATES],dtype=np.float64)
                for i in range(len(self.humans)):
                    human_state = np.array([self.humans[i].position[0],self.humans[i].position[1],self.humans[i].linear_velocity[0],self.humans[i].linear_velocity[1]], dtype=np.float64)
                    state[i] = human_state
        return state

    def set_human_states(self, state:np.array, just_visual=False):
        if not just_visual:
            # State must be of the form: [x, y, yaw, Vx, Vy, Omega, Gx, Gy] or [x, y, yaw, BVx, BVy, Omega, Gx, Gy]
            for i in range(len(self.humans)):
                self.humans[i].position[0] = state[i,0]
                self.humans[i].position[1] = state[i,1]
                self.humans[i].yaw = state[i,2]
                if self.headed:
                    self.humans[i].body_velocity[0] = state[i,3]
                    self.humans[i].body_velocity[1] = state[i,4]
                else:
                    self.humans[i].linear_velocity[0] = state[i,3]
                    self.humans[i].linear_velocity[1] = state[i,4]
                self.humans[i].angular_velocity = state[i,5]
                self.rewind_goals(self.humans[i], [state[i,6],state[i,7]])
                if self.orca: self.set_state_orca(i)
                if self.sf: self.sf_sim.state[:len(self.humans), :6] = [[human.position[0], human.position[1], human.linear_velocity[0], human.linear_velocity[1], human.goals[0][0], human.goals[0][1]] for human in self.humans]
                if self.parallel and not self.orca and not self.sm and not self.sf:
                    if self.headed: self.states[i] = np.array([*state[i,0:3],*self.states[i,3:5],*state[i,3:6],*self.states[i,8:10],*state[i,6:8],self.states[i,-1]], np.float64)
                    else: self.states[i] = np.array([*state[i,0:3],*state[i,3:5],*self.states[i,5:7],state[i,5],*self.states[i,8:10],*state[i,6:8],self.states[i,-1]], np.float64)
                    # Goals update logic
                    if not any(np.array_equal(goal, state[i,6:8]) for goal in self.goals[i]): self.goals[i] = state[i,6:8] # If the goal is not in the list, we insert it at the beginning (used for parallel traffic scenario, where goals list is dynamic)
                    else:
                        if not np.array_equal(self.goals[i,0], state[i,6:8]):
                            if np.isnan(self.goals[i]).any(): first_nan_idx = np.argwhere(np.isnan(self.goals[i]))[0][0]
                            else: first_nan_idx = len(self.goals[i])
                            while not np.array_equal(self.goals[i,0], state[i,6:8]):
                                reached_goal = np.copy(self.goals[i][0])
                                for gidx in range(first_nan_idx):
                                    if gidx < first_nan_idx - 1: self.goals[i][gidx,:] = self.goals[i][gidx+1,:]
                                    else: self.goals[i][gidx,:] = reached_goal
        else:
            # We only care about position and yaw [x, y, yaw], state can be of any form
            for i in range(len(self.humans)):
                self.humans[i].position[0] = state[i,0]
                self.humans[i].position[1] = state[i,1]
                self.humans[i].yaw = state[i,2]
                self.humans[i].update()

    def update_humans(self, t:float, dt:float, post_update=True):
        ### Update humans
        if not self.orca and not self.sm and not self.sf: ## SFM & HSFM (both Euler and RK45)
            if not self.runge_kutta:
                if self.parallel:
                    if self.consider_robot: self.states[-1] = self.robot.get_safe_state()
                    self.states = update_humans_parallel(self.sfm_type, self.states, self.goals, self.obstacles, self.params, dt, self.safety_space, all_params_equal=self.all_equal_humans, last_is_robot=self.consider_robot)
                    for i, human in enumerate(self.humans): 
                        human.set_state(self.states[i,0:8])
                        # Update human goal for state change
                        if not np.array_equal(np.array(human.goals[0], np.float64), self.states[i,10:12]):
                            goal = human.goals[0]
                            human.goals.remove(goal)
                            human.goals.append(goal)
                else:
                    self.compute_forces()
                    if not self.headed: # SFM Euler
                        for agent in self.humans: self.euler_not_headed_single_agent_update(agent, dt, self.include_mass)
                    else: # HSFM Euler
                        for agent in self.humans: self.euler_headed_single_agent_update(agent, dt)
            else:
                if not self.headed: # SFM RK45
                    current_state = np.reshape(self.get_human_states(include_goal=False, headed=False), (len(self.humans) * N_NOT_HEADED_STATES,))
                    solution = solve_ivp(self.f_rk45_not_headed, (t, t+dt), current_state, method='RK45')
                    for i, human in enumerate(self.humans): self.set_new_not_headed_state_from_rk45_solution(human, solution.y[i*N_NOT_HEADED_STATES:i*N_NOT_HEADED_STATES+N_NOT_HEADED_STATES,-1])
                else: # HSFM RK45
                    current_state = np.reshape(self.get_human_states(include_goal=False, headed=True), (len(self.humans) * N_HEADED_STATES,))
                    solution = solve_ivp(self.f_rk45_headed, (t, t+dt), current_state, method='RK45')
                    for i, human in enumerate(self.humans):
                        self.set_new_headed_state_from_rk45_solution(human, solution.y[i*N_HEADED_STATES:i*N_HEADED_STATES+N_HEADED_STATES,-1])
                        self.headed_agent_update_linear_velocity(human) # We update the linear velocity here becaue CrowdNav uses linear velocity for Observable and Full states      
        elif self.orca and not self.sm and not self.sf: ## ORCA (only Euler)
            self.sim.setTimeStep(dt)
            self.sim.doStep()
            for i, agent in enumerate(self.agents):
                if i == len(self.humans): self.set_state_orca(i); continue # Robot
                self.humans[i].linear_velocity[0] = self.sim.getAgentVelocity(agent)[0]
                self.humans[i].linear_velocity[1] = self.sim.getAgentVelocity(agent)[1]
                self.humans[i].position[0] = self.sim.getAgentPosition(agent)[0]
                self.humans[i].position[1] = self.sim.getAgentPosition(agent)[1]
                self.update_goals_orca(i)
        elif not self.orca and self.sm and not self.sf: ## SOCIAL MOMENTUM
            actions = []
            for i, human in enumerate(self.humans):
                self.update_goals(human)
                collision_free_actions = filter_action_set_for_collisions(i, self.humans, self.robot, human.action_set, self.consider_robot, dt)
                reactive_agents = update_reactive_agents(i, self.humans, self.robot, self.consider_robot)
                actions.append(optimize_momentum(human, reactive_agents, collision_free_actions, dt))
            for i, human in enumerate(self.humans):
                human.position += human.linear_velocity * dt
                human.linear_velocity = actions[i] 
        elif not self.orca and not self.sm and self.sf: ## CROWDNAV SOCIALFORCE
            if self.consider_robot: self.sf_sim.state[len(self.humans), :6] = [self.robot.position[0], self.robot.position[1], self.robot.linear_velocity[0], self.robot.linear_velocity[1], self.robot.goals[0][0], self.robot.goals[0][1]]
            self.sf_sim.delta_t = dt
            self.sf_sim.step()
            for i, human in enumerate(self.humans):
                human.linear_velocity[0] = self.sf_sim.state[i, 2]
                human.linear_velocity[1] = self.sf_sim.state[i, 3]
                human.position[0] = self.sf_sim.state[i, 0]
                human.position[1] = self.sf_sim.state[i, 1]
                self.update_goals(human)
                self.sf_sim.state[i, 4] = human.goals[0][0]
                self.sf_sim.state[i, 5] = human.goals[0][1]
        else: raise ValueError("Motion model for umans not correctly set")
        ### Post-update changes
        if post_update:
            if self.parallel_traffic_humans_respawn:
                for iindex, human in enumerate(self.humans):
                    if np.linalg.norm(human.position - human.goals[0]) < 3:
                        ### Respawn logic
                        if self.consider_robot: human.position[0] = max(max([h.position[0] for h in self.humans] + [self.robot.position[0]]) + (max([h.radius + h.safety_space for h in self.humans] + [self.robot.radius + self.robot.safety_space]) * 2), self.respawn_bounds[0])
                        else: human.position[0] = max(max([h.position[0] for h in self.humans]) + (max([h.radius + h.safety_space for h in self.humans]) * 2), self.respawn_bounds[0])
                        if human.position[1] >= 0: human.position[1] = min(human.position[1], self.respawn_bounds[1])
                        else: human.position[1] = max(human.position[1], -self.respawn_bounds[1])
                        if self.orca: self.sim.setAgentPosition(self.agents[iindex], (human.position[0], human.position[1]))
                        if self.sf: self.sf_sim.state[iindex, 0:2] = human.position
                        human.set_goals([[human.goals[0][0], human.position[1]]])
                        if self.parallel:
                            self.states[iindex, 0:2] = np.copy(human.position)
                            self.states[iindex, 6:8] = np.array(human.goals[0], np.float64)
                            self.goals[iindex] = np.array(human.goals, np.float64)                      

    def compute_single_human_forces(self, agent_idx:int, human:HumanAgent, groups:dict, social_force=True):
        desired_direction = compute_desired_force(human)
        compute_obstacle_force(human)
        if social_force: compute_social_force(agent_idx, self.humans, self.robot, self.consider_robot)
        if not self.headed:
            compute_group_force(agent_idx, self.humans, desired_direction, groups)
            human.global_force = human.desired_force + human.obstacle_force + human.social_force + human.group_force
        else:
            compute_group_force(agent_idx, self.humans, desired_direction, groups)
            compute_torque_force(human)
            human.global_force[0] = np.dot(human.desired_force + human.obstacle_force + human.social_force, human.rotational_matrix[:,0]) + human.group_force[0]
            human.global_force[1] = human.ko * np.dot(human.obstacle_force + human.social_force, human.rotational_matrix[:,1]) - human.kd * human.body_velocity[1] + human.group_force[1]

    def compute_forces(self):
        groups = {}
        for i, human in enumerate(self.humans):
            # Update goals
            if self.update_targets: self.update_goals(human)
            # Update obstacles
            human.obstacles.clear()
            for wall in self.walls:
                obstacle, distance = wall.get_closest_point(human.position)
                human.obstacles.append(obstacle)
            # Update linear velocity and rotation matrix for Headed models
            if self.headed: self.headed_agent_update_linear_velocity(human)
            # Update groups
            if (human.group_id <0): continue
            if (not human.group_id in groups): groups[human.group_id] = Group()
            groups[human.group_id].append_agent(i)
            groups[human.group_id].center += human.position
        for key in groups: groups[key].compute_center()
        if self.all_equal_humans:
            compute_all_social_forces(self.type, self.humans, self.robot, self.consider_robot)
            for i, human in enumerate(self.humans): self.compute_single_human_forces(i, human, groups, social_force=False)
        else:
            for i, human in enumerate(self.humans): self.compute_single_human_forces(i, human, groups, social_force=True)

    def complete_rk45_simulation(self, t:float, dt:float, final_time:float):
        evaluation_times = np.arange(t,final_time,dt, dtype=np.float64)
        if self.headed: 
            current_state = np.reshape(self.get_human_states(include_goal=False, headed=True), (len(self.humans) * N_HEADED_STATES,))
            solution = solve_ivp(self.f_rk45_headed, (t, t+final_time), current_state, method='RK45', t_eval=evaluation_times)
            human_states = np.empty((len(evaluation_times),len(self.humans),N_HEADED_STATES), dtype=np.float64)
            for i in range(len(solution.y[0])):
                for j in range(len(self.humans)):
                    human_states[i,j,0] = solution.y[j*N_HEADED_STATES][i]
                    human_states[i,j,1] = solution.y[j*N_HEADED_STATES+1][i]
                    human_states[i,j,2] = solution.y[j*N_HEADED_STATES+2][i]
                    human_states[i,j,3] = solution.y[j*N_HEADED_STATES+3][i]
                    human_states[i,j,4] = solution.y[j*N_HEADED_STATES+4][i]
                    human_states[i,j,5] = solution.y[j*N_HEADED_STATES+5][i]
        else:
            current_state = np.reshape(self.get_human_states(include_goal=False, headed=False), (len(self.humans) * N_NOT_HEADED_STATES,))
            solution = solve_ivp(self.f_rk45_not_headed, (t, t+final_time), current_state, method='RK45', t_eval=evaluation_times)
            human_states = np.empty((len(evaluation_times),len(self.humans),N_NOT_HEADED_STATES), dtype=np.float64)
            for i in range(len(solution.y[0])):
                for j in range(len(self.humans)):
                    human_states[i,j,0] = solution.y[j*N_NOT_HEADED_STATES][i]
                    human_states[i,j,1] = solution.y[j*N_NOT_HEADED_STATES+1][i]
                    human_states[i,j,2] = solution.y[j*N_NOT_HEADED_STATES+2][i]
                    human_states[i,j,3] = solution.y[j*N_NOT_HEADED_STATES+3][i]
        return human_states

    def f_rk45_headed(self, t, y):
        for i in range(len(self.humans)): self.set_new_headed_state_from_rk45_solution(self.humans[i], y[(i*N_HEADED_STATES):(i*N_HEADED_STATES+N_HEADED_STATES)])
        self.compute_forces()
        ydot = np.empty((len(self.humans) * N_HEADED_STATES,), dtype=np.float64)
        for i in range(len(self.humans)):
            ydot[i*N_HEADED_STATES] = np.dot(self.humans[i].rotational_matrix[0,:], self.humans[i].body_velocity)
            ydot[i*N_HEADED_STATES+1] = np.dot(self.humans[i].rotational_matrix[1,:], self.humans[i].body_velocity)
            ydot[i*N_HEADED_STATES+2] = self.humans[i].angular_velocity
            ydot[i*N_HEADED_STATES+3] = self.humans[i].global_force[0] / self.humans[i].mass
            ydot[i*N_HEADED_STATES+4] = self.humans[i].global_force[1] / self.humans[i].mass
            ydot[i*N_HEADED_STATES+5] = self.humans[i].torque_force / self.humans[i].inertia
        return ydot
    
    def f_rk45_not_headed(self, t, y):
        for i in range(len(self.humans)): self.set_new_not_headed_state_from_rk45_solution(self.humans[i], y[(i*N_NOT_HEADED_STATES):(i*N_NOT_HEADED_STATES+N_NOT_HEADED_STATES)])
        self.compute_forces()
        ydot = np.empty((len(self.humans) * N_NOT_HEADED_STATES,), dtype=np.float64)
        if self.include_mass:
            for i in range(len(self.humans)):
                ydot[i*N_NOT_HEADED_STATES] = self.humans[i].linear_velocity[0]
                ydot[i*N_NOT_HEADED_STATES+1] = self.humans[i].linear_velocity[1]
                ydot[i*N_NOT_HEADED_STATES+2] = self.humans[i].global_force[0] / self.humans[i].mass
                ydot[i*N_NOT_HEADED_STATES+3] = self.humans[i].global_force[1] / self.humans[i].mass
        else:
            for i in range(len(self.humans)):
                ydot[i*N_NOT_HEADED_STATES] = self.humans[i].linear_velocity[0]
                ydot[i*N_NOT_HEADED_STATES+1] = self.humans[i].linear_velocity[1]
                ydot[i*N_NOT_HEADED_STATES+2] = self.humans[i].global_force[0]
                ydot[i*N_NOT_HEADED_STATES+3] = self.humans[i].global_force[1]
        return ydot
    
    ### METHODS ONLY FOR ROBOT

    def get_robot_state(self, include_goal=True, headed=False):
        if include_goal:
            if headed:
                # State: [x, y, yaw, BVx, BVy, Omega, Gx, Gy] - Pose (x,y,yaw), Body velocity (body_linear_x,body_linear_y,angular), and Goal (goal_x, goal_y)
                state = np.array([self.robot.position[0],self.robot.position[1],self.robot.yaw,self.robot.body_velocity[0],self.robot.body_velocity[1],self.robot.angular_velocity,self.robot.goals[0][0],self.robot.goals[0][1]], dtype=np.float64)
            else:
                # State: [x, y, yaw, Vx, Vy, Omega, Gx, Gy] - Pose (x,y,yaw), Velocity (linear_x,linear_y,angular), and Goal (goal_x, goal_y)
                state = np.array([self.robot.position[0],self.robot.position[1],self.robot.yaw,self.robot.linear_velocity[0],self.robot.linear_velocity[1],self.robot.angular_velocity,self.robot.goals[0][0],self.robot.goals[0][1]], dtype=np.float64)
        else:
            if headed:
                # State: [x, y, yaw, BVx, BVy, Omega] - Pose (x,y,yaw) and Velocity (body_linear_x,body_linear_y,angular)
                state = np.array([self.robot.position[0],self.robot.position[1],self.robot.yaw,self.robot.body_velocity[0],self.robot.body_velocity[1],self.robot.angular_velocity], dtype=np.float64)
            else:
                # State: [x, y, Vx, Vy] - Position (x,y) and Velocity (linear_x,linear_y)
                state = np.array([self.robot.position[0],self.robot.position[1],self.robot.linear_velocity[0],self.robot.linear_velocity[1]], dtype=np.float64)
        return state

    def set_robot_state(self, state:np.array):
        self.robot.position[0] = state[0]
        self.robot.position[1] = state[1]
        self.robot.yaw = state[2]
        if not self.robot.headed:
            self.robot.linear_velocity[0] = state[3]
            self.robot.linear_velocity[1] = state[4]
        else:
            self.robot.body_velocity[0] = state[3]
            self.robot.body_velocity[1] = state[4]
        self.robot.angular_velocity = state[5]
        self.rewind_goals(self.robot, [state[6],state[7]])
        if self.consider_robot and self.orca: self.set_state_orca(len(self.humans)) # Set robot state in Humans ORCA simulation
        if self.robot.orca: self.set_state_orca(len(self.humans), True) # Set robot state in Robot ORCA simulation

    def set_robot_motion_model(self, motion_model_title:str, runge_kutta:bool):
        """"
        Sets the policy which the robot will follow to move in the environment.

        params:
        - motion_model_title (str): title of the motion model to set. Should be one between:
        "sfm_roboticsupo","sfm_helbing","sfm_guo","sfm_moussaid","hsfm_farina","hsfm_guo",
        "hsfm_moussaid","hsfm_new","hsfm_new_guo","hsfm_new_moussaid","orca".
        - runge_kutta (bool): If true, integration is carried out with RK45, otherwise with Euler

        output: None
        """
        self.robot_runge_kutta = runge_kutta
        self.robot_motion_model_title = motion_model_title
        global compute_robot_desired_force, compute_robot_obstacle_force, compute_robot_social_force, compute_robot_torque_force
        if self.robot_motion_model_title == "sfm_helbing": from social_gym.src.forces import compute_desired_force as compute_robot_desired_force, compute_obstacle_force_helbing as compute_robot_obstacle_force, compute_social_force_helbing as compute_robot_social_force; self.robot.headed=False; self.robot_headed = False; self.robot_include_mass = True; self.robot_orca = False; self.robot.orca = False
        elif self.robot_motion_model_title == "sfm_guo": from social_gym.src.forces import compute_desired_force as compute_robot_desired_force, compute_obstacle_force_guo as compute_robot_obstacle_force, compute_social_force_guo as compute_robot_social_force; self.robot.headed=False; self.robot_headed = False; self.robot_include_mass = True; self.robot_orca = False; self.robot.orca = False
        elif self.robot_motion_model_title == "sfm_moussaid": from social_gym.src.forces import compute_desired_force as compute_robot_desired_force, compute_obstacle_force_helbing as compute_robot_obstacle_force, compute_social_force_moussaid as compute_robot_social_force; self.robot.headed=False; self.robot_headed = False; self.robot_include_mass = True; self.robot_orca = False; self.robot.orca = False
        elif self.robot_motion_model_title == "sfm_roboticsupo": from social_gym.src.forces import compute_desired_force_roboticsupo as compute_robot_desired_force, compute_obstacle_force_roboticsupo as compute_robot_obstacle_force, compute_social_force_roboticsupo as compute_robot_social_force; self.robot.headed=False; self.robot_headed = False; self.robot_include_mass = False; self.robot_orca = False; self.robot.orca = False
        elif self.robot_motion_model_title == "hsfm_farina": from social_gym.src.forces import compute_desired_force as compute_robot_desired_force, compute_obstacle_force_helbing as compute_robot_obstacle_force, compute_social_force_helbing as compute_robot_social_force, compute_torque_force_farina as compute_robot_torque_force; self.robot.headed=True; self.robot_headed = True; self.robot_include_mass = True; self.robot_orca = False; self.robot.orca = False
        elif self.robot_motion_model_title == "hsfm_guo": from social_gym.src.forces import compute_desired_force as compute_robot_desired_force, compute_obstacle_force_guo as compute_robot_obstacle_force, compute_social_force_guo as compute_robot_social_force, compute_torque_force_farina as compute_robot_torque_force; self.robot.headed=True; self.robot_headed = True; self.robot_include_mass = True; self.robot_orca = False; self.robot.orca = False
        elif self.robot_motion_model_title == "hsfm_moussaid": from social_gym.src.forces import compute_desired_force as compute_robot_desired_force, compute_obstacle_force_helbing as compute_robot_obstacle_force, compute_social_force_moussaid as compute_robot_social_force, compute_torque_force_farina as compute_robot_torque_force; self.robot.headed=True; self.robot_headed = True; self.robot_include_mass = True; self.robot_orca = False; self.robot.orca = False
        elif self.robot_motion_model_title == "hsfm_new": from social_gym.src.forces import compute_desired_force as compute_robot_desired_force, compute_obstacle_force_helbing as compute_robot_obstacle_force, compute_social_force_helbing as compute_robot_social_force, compute_torque_force_new as compute_robot_torque_force; self.robot.headed=True; self.robot_headed = True; self.robot_include_mass = True; self.robot_orca = False; self.robot.orca = False
        elif self.robot_motion_model_title == "hsfm_new_guo": from social_gym.src.forces import compute_desired_force as compute_robot_desired_force, compute_obstacle_force_guo as compute_robot_obstacle_force, compute_social_force_guo as compute_robot_social_force, compute_torque_force_new as compute_robot_torque_force; self.robot.headed=True; self.robot_headed = True; self.robot_include_mass = True; self.robot_orca = False; self.robot.orca = False
        elif self.robot_motion_model_title == "hsfm_new_moussaid": from social_gym.src.forces import compute_desired_force as compute_robot_desired_force, compute_obstacle_force_helbing as compute_robot_obstacle_force, compute_social_force_moussaid as compute_robot_social_force, compute_torque_force_new as compute_robot_torque_force; self.robot.headed=True; self.robot_headed = True; self.robot_include_mass = True; self.robot_orca = False; self.robot.orca = False
        elif self.robot_motion_model_title == "orca": 
            if runge_kutta: raise NotImplementedError
            self.robot_orca = True; self.robot.orca = True; self.robot.headed=False; self.robot_headed = False; self.include_mass = False
            self.robot_sim = rvo2.PyRVOSimulator(1/60, ORCA_DEFAULTS[0], ORCA_DEFAULTS[1], ORCA_DEFAULTS[2], ORCA_DEFAULTS[3], 0.3, 1) # dt is set at each update
            self.robot_sim_agents = []
            for i, agent in enumerate(self.humans):
                # Adding agents to the simulator: parameters ((poistion_x, position_y), agents_neighbor_dist, agents_max_neighbors, safety_time_horizon, safety_time_horizon_obstacles, agents_radius, agent_max_speed, (velocity_x, velocity_y))
                self.robot_sim_agents.append(self.robot_sim.addAgent((agent.position[0], agent.position[1]), ORCA_DEFAULTS[0], ORCA_DEFAULTS[1], ORCA_DEFAULTS[2], ORCA_DEFAULTS[3], agent.radius + 0.01, agent.desired_speed, (agent.linear_velocity[0], agent.linear_velocity[1])))
                self.robot_sim.setAgentPrefVelocity(i, (0,0))
            self.robot_sim_agents.append(self.robot_sim.addAgent((self.robot.position[0], self.robot.position[1]), ORCA_DEFAULTS[0], ORCA_DEFAULTS[1], ORCA_DEFAULTS[2], ORCA_DEFAULTS[3], self.robot.radius + 0.01, self.robot.desired_speed, (self.robot.linear_velocity[0], self.robot.linear_velocity[1])))
            self.update_goals_orca(len(self.humans), robot_sim=True, robot=True)
            for wall in self.walls: self.robot_sim.addObstacle(list(wall.vertices))
            self.robot_sim.processObstacles()
        else: raise Exception(f"The robot motion model '{self.robot_motion_model_title}' does not exist")
        if self.robot_motion_model_title != "orca": self.robot.set_parameters(self.robot_motion_model_title)

    def compute_robot_forces(self):
        """
        Computes the forces based on the selected robot Social Force Model (No ORCA).
        """
        # Update goals
        if self.update_targets: self.update_goals(self.robot) # Pay attention to this
        # Update obstacles
        self.robot.obstacles.clear()
        for wall in self.walls:
            obstacle, distance = wall.get_closest_point(self.robot.position)
            self.robot.obstacles.append(obstacle)
        # Update linear velocity and rotation matrix for Headed models
        if self.robot_headed: self.headed_agent_update_linear_velocity(self.robot)
        _ = compute_robot_desired_force(self.robot)
        compute_robot_obstacle_force(self.robot)
        compute_robot_social_force(len(self.humans), self.humans, self.robot, False)
        if not self.robot_headed: self.robot.global_force = self.robot.desired_force + self.robot.obstacle_force + self.robot.social_force
        else:
            compute_robot_torque_force(self.robot)
            self.robot.global_force[0] = np.dot(self.robot.desired_force + self.robot.obstacle_force + self.robot.social_force, self.robot.rotational_matrix[:,0])
            self.robot.global_force[1] = self.robot.ko * np.dot(self.robot.obstacle_force + self.robot.social_force, self.robot.rotational_matrix[:,1]) - self.robot.kd * self.robot.body_velocity[1]

    def update_robot(self, t:float, dt:float, just_velocities=False):
        """
        Makes a step to update the robot state based on the timestep given (dt) and the selected motion model.

        params:
        - t (float): current time
        - dt (float): time step

        output: None
        """
        if not self.robot_orca: ## SFM & HSFM (both Euler and RK45)
            if not self.robot_runge_kutta:
                self.compute_robot_forces()
                if not self.robot_headed: self.euler_not_headed_single_agent_update(self.robot, dt, self.robot_include_mass, just_velocities=just_velocities) # SFM Euler
                else: self.euler_headed_single_agent_update(self.robot, dt, just_velocities=just_velocities) # HSFM Euler
            else:
                if just_velocities: raise ValueError("Runge-kutta integration cannot be used if robot and environment sampling times are different")
                if not self.robot_headed: # SFM RK45
                    current_state = self.get_robot_state(include_goal=False, headed=False)
                    solution = solve_ivp(self.f_rk45_robot_not_headed, (t, t+dt), current_state, method='RK45')
                    self.set_new_not_headed_state_from_rk45_solution(self.robot, solution.y[:,-1])
                else: # HSFM RK45
                    current_state = self.get_robot_state(include_goal=False, headed=True)
                    solution = solve_ivp(self.f_rk45_robot_headed, (t, t+dt), current_state, method='RK45')
                    self.set_new_headed_state_from_rk45_solution(self.robot, solution.y[:,-1])
                    self.headed_agent_update_linear_velocity(self.robot) # We update the linear velocity here becaue CrowdNav uses linear velocity for Observable and Full states             
        else: ## ORCA (only Euler)
            self.robot_sim.setTimeStep(dt)
            for i in range(len(self.robot_sim_agents)):
                if i < len(self.humans): self.set_state_orca(i, robot_sim=True); # Human
            self.robot_sim.doStep()
            self.robot.linear_velocity[0] = self.robot_sim.getAgentVelocity(self.robot_sim_agents[len(self.humans)])[0]
            self.robot.linear_velocity[1] = self.robot_sim.getAgentVelocity(self.robot_sim_agents[len(self.humans)])[1]
            if not just_velocities:
                self.robot.position[0] = self.robot_sim.getAgentPosition(self.robot_sim_agents[len(self.humans)])[0]
                self.robot.position[1] = self.robot_sim.getAgentPosition(self.robot_sim_agents[len(self.humans)])[1]
            else:
                self.robot_sim.setAgentPosition(self.robot_sim_agents[len(self.humans)], tuple(self.robot.position))
            self.update_goals_orca(self.robot_sim_agents[len(self.humans)], robot_sim=True, robot=True)

    def update_robot_pose(self, dt:float):
        self.robot.position += self.robot.linear_velocity * dt
        self.robot.yaw += self.robot.angular_velocity * dt
        if hasattr(self, "robot_orca") and self.robot_orca: self.set_state_orca(len(self.humans), True)
        if self.orca and self.consider_robot: self.set_state_orca(len(self.humans), False)

    def f_rk45_robot_headed(self, t, y):
        self.set_new_headed_state_from_rk45_solution(self.robot, y)
        self.compute_robot_forces()
        ydot = np.empty((N_HEADED_STATES,), dtype=np.float64)
        ydot[0] = np.dot(self.robot.rotational_matrix[0,:], self.robot.body_velocity)
        ydot[1] = np.dot(self.robot.rotational_matrix[1,:], self.robot.body_velocity)
        ydot[2] = self.robot.angular_velocity
        ydot[3] = self.robot.global_force[0] / self.robot.mass
        ydot[4] = self.robot.global_force[1] / self.robot.mass
        ydot[5] = self.robot.torque_force / self.robot.inertia
        return ydot

    def f_rk45_robot_not_headed(self, t, y):
        self.set_new_not_headed_state_from_rk45_solution(self.robot, y)
        self.compute_robot_forces()
        ydot = np.empty((N_NOT_HEADED_STATES,), dtype=np.float64)
        if self.robot_include_mass:
            ydot[0] = self.robot.linear_velocity[0]
            ydot[1] = self.robot.linear_velocity[1]
            ydot[2] = self.robot.global_force[0] / self.robot.mass
            ydot[3] = self.robot.global_force[1] / self.robot.mass
        else:
            ydot[0] = self.robot.linear_velocity[0]
            ydot[1] = self.robot.linear_velocity[1]
            ydot[2] = self.robot.global_force[0]
            ydot[3] = self.robot.global_force[1]
        return ydot

    ### METHODS FOR ROBOT CROWDNAV POLICIES

    def get_next_human_observable_states(self, dt:float, theta_and_omega_visible=False):
        """
        This function returns next humans states (in the form [px, py, vx, vy] if theta_and_omega_visible=False or 
        in the form [x, y, yaw, Vx, Vy, Omega, Gx, Gy], otherwise) without actually updating it.
        Initially, it saves the human states, then makes an update and saves the next state.
        Finally, humans' state is set as the previous one.

        params:
        - dt (float): time step used for the update

        output:
        - next_human_observable_states (np.array): next humans observable states (n, 4), for each agent (px, py, vx, vy)
        """
        current_human_states = self.get_human_states(include_goal=True, headed=self.headed)
        self.update_humans(0, dt, post_update=False)
        if theta_and_omega_visible: next_human_observable_states = self.get_human_states(include_goal=True, headed=False)
        else: next_human_observable_states = self.get_human_states(include_goal=False, headed=False)
        self.set_human_states(current_human_states)
        return next_human_observable_states

    def get_next_robot_full_state(self, dt:float):
        """
        This function computes the next robot full state without actually modifying its state.

        params:
        - dt (float): time step used for the update

        output:
        - next_robot_full_state (np.array): next full state of the robot
        - next_robot_linear_velocity (np.array): next robot velocity
        """
        current_robot_state = self.get_robot_state(include_goal=True, headed=self.robot_headed)
        self.update_robot(0, dt)
        next_robot_full_state = self.get_robot_state(include_goal=True, headed=self.robot_headed)
        next_robot_linear_velocity = self.robot.linear_velocity.copy()
        self.set_robot_state(current_robot_state)
        return next_robot_full_state, next_robot_linear_velocity