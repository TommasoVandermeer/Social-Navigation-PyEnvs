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
        global compute_torque_force
        if self.motion_model_title == "sfm_helbing": from src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_helbing as compute_social_force, compute_group_force_dummy as compute_group_force; self.headed = False; self.include_mass = True
        elif self.motion_model_title == "sfm_guo": from src.forces import compute_desired_force, compute_obstacle_force_guo as compute_obstacle_force, compute_social_force_guo as compute_social_force, compute_group_force_dummy as compute_group_force; self.headed = False; self.include_mass = True
        elif self.motion_model_title == "sfm_moussaid": from src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_moussaid as compute_social_force, compute_group_force_dummy as compute_group_force; self.headed = False; self.include_mass = True
        elif self.motion_model_title == "sfm_roboticsupo": from src.forces import compute_desired_force_roboticsupo as compute_desired_force, compute_obstacle_force_roboticsupo as compute_obstacle_force, compute_social_force_roboticsupo as compute_social_force, compute_group_force_roboticsupo as compute_group_force; self.headed = False; self.include_mass = False
        elif self.motion_model_title == "hsfm_farina": from src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_helbing as compute_social_force, compute_torque_force_farina as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True
        elif self.motion_model_title == "hsfm_guo": from src.forces import compute_desired_force, compute_obstacle_force_guo as compute_obstacle_force, compute_social_force_guo as compute_social_force, compute_torque_force_farina as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True
        elif self.motion_model_title == "hsfm_moussaid": from src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_moussaid as compute_social_force, compute_torque_force_farina as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True
        elif self.motion_model_title == "hsfm_new": from src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_helbing as compute_social_force, compute_torque_force_new as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True
        elif self.motion_model_title == "hsfm_new_guo": from src.forces import compute_desired_force, compute_obstacle_force_guo as compute_obstacle_force, compute_social_force_guo as compute_social_force, compute_torque_force_new as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True
        elif self.motion_model_title == "hsfm_new_moussaid": from src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_moussaid as compute_social_force, compute_torque_force_new as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True
        else: raise Exception(f"The human motion model '{self.motion_model_title}' does not exist")

    def update_goals(self, agent:HumanAgent):
        if ((agent.goals) and (np.linalg.norm(agent.goals[0] - agent.position) < GOAL_RADIUS)):
            goal = agent.goals[0]
            agent.goals.remove(goal)
            agent.goals.append(goal)

    def bound_linear_velocity(self, agent:HumanAgent):
        if (np.linalg.norm(agent.linear_velocity) > agent.desired_speed): agent.linear_velocity = (agent.linear_velocity / np.linalg.norm(agent.linear_velocity)) * agent.desired_speed

    def bound_body_velocity(self, agent:HumanAgent):
        if (np.linalg.norm(agent.body_velocity) > agent.desired_speed): agent.body_velocity = (agent.body_velocity / np.linalg.norm(agent.body_velocity)) * agent.desired_speed

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
            if not self.headed:
                compute_group_force(i, agents, desired_direction, groups)
                agents[i].global_force = agents[i].desired_force + agents[i].obstacle_force + agents[i].social_force + agents[i].group_force
            else:
                agents[i].rotational_matrix = np.array([[np.cos(agents[i].yaw), -np.sin(agents[i].yaw)],[np.sin(agents[i].yaw), np.cos(agents[i].yaw)]], dtype=np.float64)
                compute_group_force(i, agents, desired_direction, groups)
                compute_torque_force(agents[i])
                agents[i].global_force[0] = np.dot(agents[i].desired_force + agents[i].obstacle_force + agents[i].social_force, agents[i].rotational_matrix[:,0]) + agents[i].group_force[0]
                agents[i].global_force[1] = agents[i].ko * np.dot(agents[i].obstacle_force + agents[i].social_force, agents[i].rotational_matrix[:,1]) - agents[i].kd * agents[i].body_velocity[1] + agents[i].group_force[1]
        
    def update_positions(self, agents:list[HumanAgent], t:float, dt:float):
        global MASS, LINEAR_VELOCITY, ANGULAR_VELOCITY, GLOBAL_FORCE, TORQUE_FORCE, INERTIA
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
                for agent in agents:
                    agent.body_velocity += (agent.global_force / agent.mass) * dt
                    agent.angular_velocity += (agent.torque_force / agent.inertia) * dt
                    self.bound_body_velocity(agent)
                    agent.linear_velocity = np.matmul(agent.rotational_matrix, agent.body_velocity)
                    agent.position += agent.linear_velocity * dt
                    agent.yaw = bound_angle(agent.yaw + agent.angular_velocity * dt)
                    self.update_goals(agent)
            else:
                for agent in agents:
                    MASS = agent.mass
                    GLOBAL_FORCE = agent.global_force
                    TORQUE_FORCE = agent.torque_force
                    INERTIA = agent.inertia
                    ## First integration
                    solution = solve_ivp(f1_hsfm, (t, t+dt), [agent.body_velocity[0],agent.body_velocity[1],agent.angular_velocity], method='RK45')
                    agent.body_velocity = np.array([solution.y[0][-1],solution.y[1][-1]], dtype=np.float64)
                    agent.angular_velocity = solution.y[2][-1]
                    self.bound_body_velocity(agent)
                    agent.linear_velocity = np.matmul(agent.rotational_matrix, agent.body_velocity)
                    LINEAR_VELOCITY = agent.linear_velocity
                    ANGULAR_VELOCITY = agent.angular_velocity
                    ## Second integration
                    solution2 = solve_ivp(f2_hsfm, (t, t+dt), [agent.position[0], agent.position[1], agent.yaw], method='RK45')
                    agent.position = np.array([solution2.y[0][-1],solution2.y[1][-1]], dtype=np.float64)
                    agent.yaw = bound_angle(solution2.y[2][-1])
                    self.update_goals(agent)

def f1_sfm_with_mass(t, y):
    return GLOBAL_FORCE / MASS

def f1_sfm_no_mass(t, y):
    return GLOBAL_FORCE

def f2_sfm(t, y):
    return LINEAR_VELOCITY

def f1_hsfm(t, y):
    return np.array([GLOBAL_FORCE[0]/MASS, GLOBAL_FORCE[1]/MASS, TORQUE_FORCE/INERTIA], dtype=np.float64)

def f2_hsfm(t, y):
    return np.array([LINEAR_VELOCITY[0], LINEAR_VELOCITY[1], ANGULAR_VELOCITY], dtype=np.float64)