import math
import numpy as np
from src.human_agent import HumanAgent
from src.robot_agent import RobotAgent
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
        agent.obstacle_force += (agent.Aw * math.exp((agent.radius - distance) / agent.Bw)) * n_iw + (agent.Cw * math.exp((agent.radius - distance) / agent.Dw)) * t_iw
    if (agent.obstacles): agent.obstacle_force /= len(agent.obstacles)

def compute_social_force(index:int, agents:list[HumanAgent], robot:RobotAgent, consider_robot:bool):
    target_agent = agents[index]
    target_agent.social_force = np.array([0.0,0.0], dtype=np.float64)
    entities = agents.copy()
    if consider_robot: entities.append(robot)
    for i in range(len(entities)):
        if (i == index): continue
        r_ij = target_agent.radius + entities[i].radius
        difference = target_agent.position - entities[i].position
        distance = np.linalg.norm(difference)
        n_ij = difference / distance
        t_ij = np.array([-n_ij[1],n_ij[0]], dtype=np.float64)
        target_agent.social_force += (target_agent.Ai * math.exp((r_ij - distance) / target_agent.Bi)) * n_ij + (target_agent.Ci * math.exp((r_ij - distance) / target_agent.Di)) * t_ij

def compute_group_force(index:int, agents:list[HumanAgent], desired_direction:np.array, groups:dict):
    pass