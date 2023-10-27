import math
import numpy as np
from src.human_agent import HumanAgent
from src.robot_agent import RobotAgent
from src.motion_model_manager import GOAL_RADIUS, Group
from src.utils import bound_angle

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
        r_ij = target_agent.radius + entities[i].radius
        difference = target_agent.position - entities[i].position
        distance = np.linalg.norm(difference)
        n_ij = difference / distance
        t_ij = np.array([-n_ij[1],n_ij[0]], dtype=np.float64)
        delta_v_ij = np.dot(entities[i].linear_velocity - target_agent.linear_velocity, t_ij)
        target_agent.social_force += (target_agent.Ai * math.exp((r_ij - distance) / target_agent.Bi) + target_agent.k1 * max(0,r_ij - distance)) * n_ij + target_agent.k2  * max(0,r_ij - distance) * delta_v_ij * t_ij

def compute_torque_force(agent:HumanAgent):
    agent.k_theta = agent.inertia * agent.k_lambda * np.linalg.norm(agent.desired_force)
    agent.k_omega = agent.inertia * (1 + agent.alpha) * math.sqrt((agent.k_lambda * np.linalg.norm(agent.desired_force)) / agent.alpha)
    agent.torque_force = - agent.k_theta * bound_angle(agent.yaw - math.atan2(agent.desired_force[1],agent.desired_force[0])) - agent.k_omega * agent.angular_velocity

def compute_group_force(index:int, agents:list[HumanAgent], desired_direction:np.array, groups:dict):
    target_agent = agents[index]
    target_agent.rotational_matrix = np.array([[math.cos(target_agent.yaw), -math.sin(target_agent.yaw)],[math.sin(target_agent.yaw), math.cos(target_agent.yaw)]], dtype=np.float64)
    target_agent.group_force = np.array([0.0,0.0], dtype= np.float64)

