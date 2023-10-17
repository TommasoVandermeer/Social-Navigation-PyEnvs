import math
import numpy as np
from src.human_agent import HumanAgent
from src.robot_agent import RobotAgent
from src.utils import bound_angle

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
        if (i == index): i += 1; continue
        r_ij = target_agent.radius + entities[i].radius
        difference = target_agent.position - entities[i].position
        distance = np.linalg.norm(difference)
        n_ij = difference / distance
        t_ij = np.array([-n_ij[1],n_ij[0]], dtype=np.float64)
        delta_v_ij = np.dot(entities[i].linear_velocity - target_agent.linear_velocity, t_ij)
        target_agent.social_force += (target_agent.Ai * math.exp((r_ij - distance) / target_agent.Bi) + target_agent.k1 * max(0,r_ij - distance)) * n_ij + target_agent.k2  * max(0,r_ij - distance) * delta_v_ij * t_ij

def compute_social_force_no_robot(index:int, agents:list[HumanAgent]):
    target_agent = agents[index]
    target_agent.social_force = np.array([0.0,0.0], dtype=np.float64)
    entities = agents.copy()
    for i in range(len(entities)):
        if (i == index): i += 1; continue
        r_ij = target_agent.radius + entities[i].radius
        difference = target_agent.position - entities[i].position
        distance = np.linalg.norm(difference)
        n_ij = difference / distance
        t_ij = np.array([-n_ij[1],n_ij[0]], dtype=np.float64)
        delta_v_ij = np.dot(entities[i].linear_velocity - target_agent.linear_velocity, t_ij)
        target_agent.social_force += (target_agent.Ai * math.exp((r_ij - distance) / target_agent.Bi) + target_agent.k1 * max(0,r_ij - distance)) * n_ij + target_agent.k2  * max(0,r_ij - distance) * delta_v_ij * t_ij

def compute_forces(agents:list[HumanAgent], robot:RobotAgent):
    groups = {}
    i = 0
    for agent in agents:
        if (agent.group_id < 0): i += 1; continue
        if (not agent.group_id in groups): groups[agent.group_id] = Group()
        groups[agent.group_id].append_agent(i)
        groups[agent.group_id].center += agent.position
        i += 1
    for key in groups:
        groups[key].compute_center()
    for i in range(len(agents)):
        desired_direction = compute_desired_force(agents[i])
        compute_obstacle_force(agents[i])
        compute_social_force(i, agents, robot)
        agents[i].global_force = agents[i].desired_force + agents[i].obstacle_force + agents[i].social_force

def compute_forces_no_robot(agents:list[HumanAgent]):
    groups = {}
    i = 0
    for agent in agents:
        if (agent.group_id < 0): i += 1; continue
        if (not agent.group_id in groups): groups[agent.group_id] = Group()
        groups[agent.group_id].append_agent(i)
        groups[agent.group_id].center += agent.position
        i += 1
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

