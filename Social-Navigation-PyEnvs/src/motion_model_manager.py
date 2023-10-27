from src.utils import bound_angle
from src.human_agent import HumanAgent
from src.robot_agent import RobotAgent
from scipy.integrate import solve_ivp
import numpy as np

GOAL_RADIUS = 0.35

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
    def __init__(self, motion_model_title:str, consider_robot:bool, runge_kutta:bool):
        self.motion_model_title = motion_model_title
        self.consider_robot = consider_robot
        self.runge_kutta = runge_kutta
        global compute_desired_force
        global compute_obstacle_force
        global compute_social_force
        global compute_group_force
        if self.motion_model_title == "sfm_helbing": from src.motion_models.sfm_helbing import compute_desired_force, compute_obstacle_force, compute_social_force, compute_group_force; self.headed = False; self.include_mass = True
        elif self.motion_model_title == "sfm_guo": from src.motion_models.sfm_guo import compute_desired_force, compute_obstacle_force, compute_social_force, compute_group_force; self.headed = False; self.include_mass = True
        elif self.motion_model_title == "sfm_moussaid": from src.motion_models.sfm_moussaid import compute_desired_force, compute_obstacle_force, compute_social_force, compute_group_force; self.headed = False; self.include_mass = True
        elif self.motion_model_title == "sfm_roboticsupo": from src.motion_models.sfm_roboticsupo import compute_desired_force, compute_obstacle_force, compute_social_force, compute_group_force; self.headed = False; self.include_mass = False
        else: raise Exception(f"The human motion model '{self.motion_model_title}' does not exist")

    def update_goals(self, agent:HumanAgent):
        if ((agent.goals) and (np.linalg.norm(agent.goals[0] - agent.position) < GOAL_RADIUS)):
            goal = agent.goals[0]
            agent.goals.remove(goal)
            agent.goals.append(goal)

    def bound_linear_velocity(self, agent:HumanAgent):
        if (np.linalg.norm(agent.linear_velocity) > agent.desired_speed): agent.linear_velocity = (agent.linear_velocity / np.linalg.norm(agent.linear_velocity)) * agent.desired_speed

    def compute_forces(self, agents:list[HumanAgent], robot=RobotAgent):
        groups = {}
        for i in range(len(agents)):
            if (agents[i].group_id <0): continue
            if (not agents[i].group_id in groups): groups[agents[i].group_id] = Group()
            groups[agents[i].group_id].append_agent(i)
            groups[agents[i].group_id].center += agents[i].position
        for key in groups:
            groups[key].compute_center()
        for i in range(len(agents)):
            desired_direction = compute_desired_force(agents[i])
            compute_obstacle_force(agents[i])
            compute_social_force(i, agents, robot, self.consider_robot)
            compute_group_force(i, agents, desired_direction, groups)
            if not self.headed:
                agents[i].global_force = agents[i].desired_force + agents[i].obstacle_force + agents[i].social_force + agents[i].group_force
            else:
                pass
        
    def update_positions(self, agents:list[HumanAgent], t:float, dt:float):
        if not self.headed:
            if not self.runge_kutta:
                for agent in agents:
                    init_yaw = agent.yaw
                    if self.include_mass: agent.linear_velocity += (agent.global_force / agent.mass) * dt
                    else: agent.linear_velocity += agent.global_force * dt
                    self.bound_linear_velocity(agent)
                    agent.yaw = bound_angle(np.arctan2(agent.linear_velocity[1], agent.linear_velocity[0]))
                    agent.position += agent.linear_velocity * dt
                    agent.angular_velocity = (agent.yaw - init_yaw) / dt
                    self.update_goals(agent)
            else:
                for agent in agents:
                    init_yaw = agent.yaw
                    global MASS, LINEAR_VELOCITY, GLOBAL_FORCE
                    MASS = agent.mass
                    GLOBAL_FORCE = agent.global_force
                    ## First integration
                    if self.include_mass: solution = solve_ivp(f1_sfm_with_mass, (t, t+dt), agent.linear_velocity, method='RK45')
                    else: solution = solve_ivp(f1_sfm_no_mass, (t, t+dt), agent.linear_velocity, method='RK45')
                    agent.linear_velocity = np.array([solution.y[0][-1],solution.y[1][-1]],dtype=np.float64)
                    self.bound_linear_velocity(agent)
                    LINEAR_VELOCITY = agent.linear_velocity
                    agent.yaw = bound_angle(np.arctan2(agent.linear_velocity[1], agent.linear_velocity[0]))
                    ## Second integration
                    solution2 = solve_ivp(f2_sfm, (t, t+dt), agent.position, method='RK45')
                    agent.position = np.array([solution2.y[0][-1],solution2.y[1][-1]], dtype=np.float64)
                    agent.angular_velocity = (agent.yaw - init_yaw) / dt
                    self.update_goals(agent)
        else:
            if not self.runge_kutta:
                pass
            else:
                pass

def f1_sfm_with_mass(t, y):
    return GLOBAL_FORCE / MASS

def f1_sfm_no_mass(t, y):
    return GLOBAL_FORCE

def f2_sfm(t, y):
    return LINEAR_VELOCITY

def f1_hsfm(t, y):
    pass

def f2_hsfm(t, y):
    pass