import math
import numpy as np
from src.human_agent import HumanAgent
from src.robot_agent import RobotAgent
from src.utils import bound_angle
from scipy.integrate import solve_ivp
from src.motion_model_manager import GOAL_RADIUS, Group

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

def compute_social_force(index:int, agents:list[HumanAgent], robot:RobotAgent, consider_robot:bool):
    target_agent = agents[index]
    target_agent.social_force = np.array([0.0,0.0], dtype=np.float64)
    entities = agents.copy()
    if consider_robot: entities.append(robot)
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

def compute_group_force(index:int, agents:list[HumanAgent], desired_direction:np.array, groups:dict):
    pass