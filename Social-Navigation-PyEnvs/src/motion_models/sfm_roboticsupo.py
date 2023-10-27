import math
import numpy as np
from src.human_agent import HumanAgent
from src.robot_agent import RobotAgent
from src.utils import bound_angle
from src.motion_model_manager import GOAL_RADIUS, Group

def compute_desired_force(agent:HumanAgent):
    distance = np.linalg.norm(agent.goals[0] - agent.position)
    if ((agent.goals) and (distance > GOAL_RADIUS)):
        desired_direction = (agent.goals[0] - agent.position) / distance
        agent.desired_force = agent.goal_weight * (desired_direction * agent.desired_speed - agent.linear_velocity) / agent.relaxation_time
    else: desired_direction = np.array([0.0,0.0], dtype=np.float64)
    return desired_direction

def compute_obstacle_force(agent:HumanAgent):
    agent.obstacle_force = np.array([0.0,0.0], dtype=np.float64)
    for obstacle in agent.obstacles:
        min_diff = np.array(agent.position) - obstacle
        distance = np.linalg.norm(min_diff) - agent.radius
        agent.obstacle_force += agent.obstacle_weight * math.exp(-distance / agent.obstacle_sigma) * (min_diff / np.linalg.norm(min_diff))
    if (agent.obstacles): agent.obstacle_force /= len(agent.obstacles)

def compute_social_force(index:int, agents:list[HumanAgent], robot:RobotAgent, consider_robot:bool):
    target_agent = agents[index]
    target_agent.social_force = np.array([0.0,0.0], dtype=np.float64)
    entities = agents.copy()
    if consider_robot: entities.append(robot)
    for i in range(len(entities)):
        if (i == index): continue
        diff = np.array(entities[i].position) - np.array(target_agent.position)
        diff_direction = diff / np.linalg.norm(diff)
        vel_diff = target_agent.linear_velocity - entities[i].linear_velocity
        interaction_vector = target_agent.agent_lambda * vel_diff + diff_direction
        interaction_length = np.linalg.norm(interaction_vector)
        interaction_direction = interaction_vector / interaction_length
        theta = bound_angle(np.arctan2(diff_direction[1], diff_direction[0]) - np.arctan2(interaction_direction[1], interaction_direction[0]))
        B = target_agent.agent_gamma * interaction_length
        force_velocity_amount = -math.exp(-np.linalg.norm(diff) / B - (target_agent.agent_nPrime * B * theta) ** 2)
        force_angle_amount = np.sign(-theta) * math.exp(-np.linalg.norm(diff) / B - (target_agent.agent_n * B * theta) ** 2)
        force_velocity = force_velocity_amount * interaction_direction
        force_angle = force_angle_amount * np.array([-interaction_direction[1], interaction_direction[0]])
        target_agent.social_force += target_agent.social_weight * (force_velocity + force_angle)

def compute_group_force(index:int, agents:list[HumanAgent], desired_direction:np.array, groups:dict):
    target_agent = agents[index]
    target_agent.group_force = np.array([0.0,0.0])
    if ((not target_agent.group_id in groups) or (groups[target_agent.group_id].num_agents() < 2) or (target_agent.group_id == -1)): return
    ## Gaze force
    com = np.array(groups[target_agent.group_id].center)
    com = (1/(groups[target_agent.group_id].num_agents() -1)) * (groups[target_agent.group_id].num_agents() * com - np.array(target_agent.position))
    relative_com = com - np.array(target_agent.position)
    vision_angle = math.radians(90)
    element_product = np.dot(desired_direction, relative_com)
    com_angle = math.acos(element_product / (np.linalg.norm(desired_direction) * np.linalg.norm(relative_com)))
    if com_angle > vision_angle:
        desired_direction_squared = np.linalg.norm(desired_direction) ** 2
        desired_direction_distance = element_product / desired_direction_squared
        group_gaze_force = target_agent.group_gaze_weight * desired_direction_distance * desired_direction
    else: group_gaze_force = np.array([0.0,0.0])
    ## Coherence force
    com = np.array(groups[target_agent.group_id].center)
    relative_com = com - np.array(target_agent.position)
    distance = np.linalg.norm(relative_com)
    max_distance = (groups[target_agent.group_id].num_agents() -1) / 2
    softened_factor = target_agent.group_coh_weight * (math.tanh(distance - max_distance) + 1) / 2
    group_coherence_force = relative_com * softened_factor
    ## Repulsion force
    group_repulsion_force = np.array([0.0,0.0])
    for i in range(groups[target_agent.group_id].num_agents()):
        if index == groups[target_agent.group_id].group_agents[i]: continue
        diff = np.array(target_agent.position) - np.array(agents[i].position)
        if (np.linalg.norm(diff) < target_agent.radius + agents[i].radius): group_repulsion_force += diff
    group_repulsion_force *= target_agent.group_rep_weight
    target_agent.group_force = group_gaze_force + group_coherence_force + group_repulsion_force