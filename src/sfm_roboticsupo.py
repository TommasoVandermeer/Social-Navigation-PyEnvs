import pygame
import math
import numpy as np
from src.human_agent import HumanAgent
from src.robot_agent import RobotAgent
from src.utils import points_distance, vector_difference, normalized, bound_angle, vector_angle, points_unit_vector

GOAL_RADIUS = 0.3

class Group:
    def __init__(self):
        self.group_agents = []
        self.center = [0.0,0.0]

    def append_agent(self, agent:int):
        self.group_agents.append(agent)

    def compute_center(self):
        self.center[0] /= len(self.group_agents)
        self.center[1] /= len(self.group_agents)

    def num_agents(self):
        return len(self.group_agents)

def compute_desired_force(agent:HumanAgent):
    if ((agent.goals) and (points_distance(agent.goals[0], agent.position) > GOAL_RADIUS)):
        desired_direction = np.array(normalized(vector_difference(agent.position, agent.goals[0])))
        agent.desired_force = agent.goal_weight * (desired_direction * agent.desired_speed - agent.linear_velocity) / agent.relaxation_time
    return desired_direction

def compute_obstacle_force(agent:HumanAgent):
    agent.obstacle_force = np.array([0.0,0.0])
    for obstacle in agent.obstacles:
        min_diff = np.array(agent.position) - obstacle
        distance = np.linalg.norm(min_diff) - agent.radius
        agent.obstacle_force += agent.obstacle_weight * math.exp(-distance / agent.obstacle_sigma) * (min_diff / np.linalg.norm(min_diff))

def compute_social_force(index:int, agents:list[HumanAgent], robot:RobotAgent):
    target_agent = agents[index]
    target_agent.social_force = np.array([0.0,0.0])
    entities = agents.copy()
    entities.append(robot)
    i = 0
    for agent in entities:
        if (i == index): i += 1; continue
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
        i += 1

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

def compute_forces(agents:list[HumanAgent], robot:RobotAgent):
    groups = {}
    i = 0
    for agent in agents:
        if (agent.group_id < 0): i += 1; continue
        if (not agent.group_id in groups): groups[agent.group_id] = Group()
        groups[agent.group_id].append_agent(i)
        groups[agent.group_id].center[0] += agent.position[0]
        groups[agent.group_id].center[1] += agent.position[1]
        i += 1
    for key in groups:
        groups[key].compute_center()
    i = 0
    for agent in agents:
        desired_direction = compute_desired_force(agent)
        compute_obstacle_force(agent)
        compute_social_force(i, agents, robot)
        compute_group_force(i, agents, desired_direction, groups)
        agent.global_force = agent.desired_force + agent.obstacle_force + agent.social_force + agent.group_force
        i += 1
    
def update_positions(agents:list[HumanAgent], dt:float):
    for agent in agents:
        agent.linear_velocity += agent.global_force * dt
        if (np.linalg.norm(agent.linear_velocity) > agent.desired_speed): agent.linear_velocity = (agent.linear_velocity / np.linalg.norm(agent.linear_velocity)) * agent.desired_speed
        agent.yaw = bound_angle(np.arctan2(agent.linear_velocity[1], agent.linear_velocity[0]))
        agent.position += agent.linear_velocity * dt
        check_agents_collisions(agents)
        if ((agent.goals) and (points_distance(agent.goals[0], agent.position) < GOAL_RADIUS)):
            goal = agent.goals[0]
            agent.goals.remove(goal)
            agent.goals.append(goal)

def check_agents_collisions(agents:list[HumanAgent]):
        for i in range(len(agents)):
            for j in range(len(agents)):
                if (j == i) or (j < i): continue
                agent1_position = np.array(agents[i].position)
                agent2_position = np.array(agents[j].position)
                if (np.linalg.norm(agent1_position - agent2_position) < agents[i].radius + agents[j].radius):
                    direction = (agent1_position - agent2_position) / np.linalg.norm(agent1_position - agent2_position)
                    angle = np.arctan2(direction[1], direction[0])
                    mean_point = (agent1_position + agent2_position) / 2
                    agents[i].position[0] = mean_point[0] + math.cos(angle) * (agents[i].radius)
                    agents[i].position[1] = mean_point[1] + math.sin(angle) * (agents[i].radius)
                    agents[j].position[0] = mean_point[0] - math.cos(angle) * (agents[j].radius)
                    agents[j].position[1] = mean_point[1] - math.sin(angle) * (agents[j].radius)

