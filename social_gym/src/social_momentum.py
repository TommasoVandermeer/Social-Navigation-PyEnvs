import math
import numpy as np
from social_gym.src.agent import Agent
from social_gym.src.robot_agent import RobotAgent

REACTIVE_AGENTS_ANGLE_BOUNDS = math.pi
LAMBDA = 0.11

def filter_action_set_for_collisions(index:int, agents:list[Agent], robot:RobotAgent, action_set:list[np.array], consider_robot:bool, dt:float):
    if index <  len(agents): target_agent = agents[index]
    else: target_agent = robot
    entities = agents.copy()
    if consider_robot: entities.append(robot)
    filtered_action_set = []
    for action in action_set:
        target_agent_next_position = target_agent.position + action * dt
        no_collision = True
        for i, entity in enumerate(entities):
            if i == index: continue 
            next_entity_position = entity.position + entity.linear_velocity * dt
            distance = np.linalg.norm(next_entity_position - target_agent_next_position)
            threshold = target_agent.radius + entity.radius + target_agent.safety_space + entity.safety_space
            if distance < threshold: no_collision = False; break
            else: no_collision = True
        if no_collision: filtered_action_set.append(np.copy(action))
    return filtered_action_set

def update_reactive_agents(index:int, agents:list[Agent], robot:RobotAgent, consider_robot:bool):
    if index <  len(agents): target_agent = agents[index]
    else: target_agent = robot
    entities = agents.copy()
    if consider_robot: entities.append(robot)
    reactive_agents = []
    angle_bounds = REACTIVE_AGENTS_ANGLE_BOUNDS
    for i, entity in enumerate(entities):
        if i == index: continue
        difference = entity.position - target_agent.position
        target_agent_speed = np.linalg.norm(target_agent.linear_velocity)
        if target_agent_speed == 0: angle = 0 # If there is no motion, the angle is assumed to be 0 (every agent is reactive)
        else: angle = np.arccos(np.clip(np.dot(target_agent.linear_velocity,difference)/ (np.linalg.norm(target_agent.linear_velocity) * np.linalg.norm(difference)),-1,1))
        if angle <= angle_bounds: reactive_agents.append(entity)
    return reactive_agents

def optimize_momentum(target_agent:Agent, reactive_agents:list[Agent], actions:list[np.array], dt:float):
    l = LAMBDA
    # Compute agents weights
    weights = np.empty((len(reactive_agents),), dtype=np.float64)
    for i, agent in enumerate(reactive_agents): weights[i] = 1 / np.linalg.norm(agent.position - target_agent.position)
    # Normalize weights
    weights /= np.sum(weights)
    # Compute best action
    max_reward = -100000
    best_action = np.zeros((2,), dtype=np.float64)
    for action in actions:
        next_target_agent_position = target_agent.position + action * dt
        next_distance_to_goal = np.linalg.norm(target_agent.goals[0] - next_target_agent_position)
        reward = 1 / next_distance_to_goal # Efficiency function contribution
        momentum_reward = 0
        for i, agent in enumerate(reactive_agents):
            # Compute momentum
            center_of_mass = (target_agent.position + agent.position) / 2
            pr_c = target_agent.position - center_of_mass
            ph_c = agent.position - center_of_mass
            other_agent_momentum = np.cross(ph_c, agent.linear_velocity)
            current_momentum = np.cross(pr_c, target_agent.linear_velocity) + other_agent_momentum
            expected_pairwise_momentum = np.cross(pr_c, action) + other_agent_momentum
            # Compute momentum reward
            if current_momentum * expected_pairwise_momentum > 0: momentum_reward += weights[i] * expected_pairwise_momentum
            else: momentum_reward = 0; break
        reward += l * momentum_reward
        if reward > max_reward: 
            max_reward = reward
            best_action = np.copy(action)
    return best_action
