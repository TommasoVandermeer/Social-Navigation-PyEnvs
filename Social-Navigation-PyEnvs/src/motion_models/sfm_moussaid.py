import math
import numpy as np
from src.human_agent import HumanAgent
from src.robot_agent import RobotAgent
from src.utils import bound_angle
from scipy.integrate import solve_ivp

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

def compute_desired_force(agent:HumanAgent):
    distance = np.linalg.norm(agent.goals[0] - agent.position)
    if ((agent.goals) and (distance > GOAL_RADIUS)):
        desired_direction = (agent.goals[0] - agent.position) / distance
        agent.desired_force = agent.mass * (desired_direction * agent.desired_speed - agent.linear_velocity) / agent.relaxation_time
    else: desired_direction = np.array([0.0,0.0], dtype=np.float64)
    return desired_direction

def compute_obstacle_force(agent:HumanAgent):
    agent.obstacle_force = np.array([0.0,0.0], dtype=np.float64)
    for obstacle in agent.obstacles:
        difference = agent.position - obstacle
        distance = np.linalg.norm(difference)
        n_iw = difference / distance
        t_iw = np.array([-n_iw[1],n_iw[0]], dtype=np.float64)
        delta_v_iw = - np.dot(agent.linear_velocity, t_iw)
        agent.obstacle_force += (agent.Aw * math.exp((agent.radius - distance) / agent.Bw) + agent.k1 * max(0,agent.radius - distance)) * n_iw - agent.k2 * max(0,agent.radius - distance) * delta_v_iw * t_iw
    if (agent.obstacles): agent.obstacle_force /= len(agent.obstacles)

def compute_social_force(index:int, agents:list[HumanAgent], robot:RobotAgent):
    target_agent = agents[index]
    target_agent.social_force = np.array([0.0,0.0], dtype=np.float64)
    entities = agents.copy()
    entities.append(robot)
    for i in range(len(entities)):
        if (i == index): continue
        difference = target_agent.position - entities[i].position
        distance = np.linalg.norm(difference)
        n_ij = difference / distance
        interaction_norm = np.linalg.norm(target_agent.agent_lambda * (target_agent.linear_velocity - entities[i].linear_velocity) - n_ij)
        i_ij = (target_agent.agent_lambda * (target_agent.linear_velocity - entities[i].linear_velocity) - n_ij) / interaction_norm
        theta_ij = bound_angle(math.atan2(n_ij[1],n_ij[0]) - math.atan2(i_ij[1],i_ij[0]) + math.pi)
        k_ij = np.sign(theta_ij)
        h_ij = np.array([-i_ij[1], i_ij[0]], dtype=np.float64)
        F_ij = target_agent.gamma * interaction_norm
        target_agent.social_force += - target_agent.Ei * math.exp(-distance/F_ij) * (math.exp((-target_agent.ns1 * F_ij * theta_ij)**2) * i_ij + k_ij * math.exp((-target_agent.ns * F_ij * theta_ij)**2) * h_ij)

def compute_social_force_no_robot(index:int, agents:list[HumanAgent]):
    target_agent = agents[index]
    target_agent.social_force = np.array([0.0,0.0], dtype=np.float64)
    entities = agents.copy()
    for i in range(len(entities)):
        if (i == index): continue
        difference = target_agent.position - entities[i].position
        distance = np.linalg.norm(difference)
        n_ij = difference / distance
        interaction_norm = np.linalg.norm(target_agent.agent_lambda * (target_agent.linear_velocity - entities[i].linear_velocity) - n_ij)
        i_ij = (target_agent.agent_lambda * (target_agent.linear_velocity - entities[i].linear_velocity) - n_ij) / interaction_norm
        theta_ij = bound_angle(math.atan2(n_ij[1],n_ij[0]) - math.atan2(i_ij[1],i_ij[0]) + math.pi)
        k_ij = np.sign(theta_ij)
        h_ij = np.array([-i_ij[1], i_ij[0]], dtype=np.float64)
        F_ij = target_agent.gamma * interaction_norm
        target_agent.social_force += - target_agent.Ei * math.exp(-distance/F_ij) * (math.exp((-target_agent.ns1 * F_ij * theta_ij)**2) * i_ij + k_ij * math.exp((-target_agent.ns * F_ij * theta_ij)**2) * h_ij)

def compute_forces(agents:list[HumanAgent], robot:RobotAgent):
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
        compute_social_force(i, agents, robot)
        agents[i].global_force = agents[i].desired_force + agents[i].obstacle_force + agents[i].social_force

def compute_forces_no_robot(agents:list[HumanAgent]):
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
        compute_social_force_no_robot(i, agents)
        agents[i].global_force = agents[i].desired_force + agents[i].obstacle_force + agents[i].social_force
    
def update_positions(agents:list[HumanAgent], dt:float):
    for agent in agents:
        init_yaw = agent.yaw
        agent.linear_velocity += (agent.global_force / agent.mass) * dt
        if (np.linalg.norm(agent.linear_velocity) > agent.desired_speed): agent.linear_velocity = (agent.linear_velocity / np.linalg.norm(agent.linear_velocity)) * agent.desired_speed
        agent.yaw = bound_angle(np.arctan2(agent.linear_velocity[1], agent.linear_velocity[0]))
        agent.position += agent.linear_velocity * dt
        agent.angular_velocity = (agent.yaw - init_yaw) / dt
        if ((agent.goals) and (np.linalg.norm(agent.goals[0] - agent.position) < GOAL_RADIUS)):
            goal = agent.goals[0]
            agent.goals.remove(goal)
            agent.goals.append(goal)

def update_positions_RK45(agents:list[HumanAgent], t:float, dt:float):
    for agent in agents:
        init_yaw = agent.yaw
        global MASS
        global LINEAR_VELOCITY
        global GLOBAL_FORCE
        MASS = agent.mass
        GLOBAL_FORCE = agent.global_force
        ## First integration
        solution = solve_ivp(f1, (t, t+dt), agent.linear_velocity, method='RK45')
        agent.linear_velocity = np.array([solution.y[0][-1],solution.y[1][-1]],dtype=np.float64)
        LINEAR_VELOCITY = agent.linear_velocity
        if (np.linalg.norm(agent.linear_velocity) > agent.desired_speed): agent.linear_velocity = (agent.linear_velocity / np.linalg.norm(agent.linear_velocity)) * agent.desired_speed
        agent.yaw = bound_angle(np.arctan2(agent.linear_velocity[1], agent.linear_velocity[0]))
        ## Second integration
        solution2 = solve_ivp(f2, (t, t+dt), agent.position, method='RK45')
        agent.position = np.array([solution2.y[0][-1],solution2.y[1][-1]], dtype=np.float64)
        agent.angular_velocity = (agent.yaw - init_yaw) / dt
        if ((agent.goals) and (np.linalg.norm(agent.goals[0] - agent.position) < GOAL_RADIUS)):
            goal = agent.goals[0]
            agent.goals.remove(goal)
            agent.goals.append(goal)

def f1(t, y):
    ydot = GLOBAL_FORCE / MASS
    return ydot

def f2(t, y):
    ydot = LINEAR_VELOCITY
    return ydot
