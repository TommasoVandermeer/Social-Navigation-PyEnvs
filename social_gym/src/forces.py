import math
import numpy as np
from social_gym.src.human_agent import HumanAgent
from social_gym.src.agent import Agent
from social_gym.src.robot_agent import RobotAgent
from social_gym.src.utils import bound_angle
from social_gym.src.motion_model_manager import GOAL_RADIUS, Group

def compute_desired_force(agent:HumanAgent):
    difference = agent.goals[0] - agent.position
    distance = np.linalg.norm(difference)
    if ((agent.goals) and (distance > GOAL_RADIUS)):
        desired_direction = difference / distance
        agent.desired_force = agent.mass * (desired_direction * agent.desired_speed - agent.linear_velocity) / agent.relaxation_time
    else: desired_direction = np.array([0.0,0.0], dtype=np.float64)
    return desired_direction

def compute_desired_force_roboticsupo(agent:HumanAgent):
    difference = agent.goals[0] - agent.position
    distance = np.linalg.norm(difference)
    if ((agent.goals) and (distance > GOAL_RADIUS)):
        desired_direction = difference / distance
        agent.desired_force = agent.goal_weight * (desired_direction * agent.desired_speed - agent.linear_velocity) / agent.relaxation_time
    else: desired_direction = np.array([0.0,0.0], dtype=np.float64)
    return desired_direction

def compute_obstacle_force_helbing(agent:HumanAgent):
    agent.obstacle_force = np.array([0.0,0.0], dtype=np.float64)
    for obstacle in agent.obstacles:
        difference = agent.position - obstacle
        distance = np.linalg.norm(difference)
        n_iw = difference / distance
        t_iw = np.array([-n_iw[1],n_iw[0]], dtype=np.float64)
        delta_v_iw = - np.dot(agent.linear_velocity, t_iw)
        real_distance = agent.radius - distance
        agent.obstacle_force += (agent.Aw * math.exp((real_distance) / agent.Bw) + agent.k1 * max(0,real_distance)) * n_iw - agent.k2 * max(0,real_distance) * delta_v_iw * t_iw
    if (agent.obstacles): agent.obstacle_force /= len(agent.obstacles)

def compute_obstacle_force_guo(agent:HumanAgent):
    agent.obstacle_force = np.array([0.0,0.0], dtype=np.float64)
    for obstacle in agent.obstacles:
        difference = agent.position - obstacle
        distance = np.linalg.norm(difference)
        n_iw = difference / distance
        t_iw = np.array([-n_iw[1],n_iw[0]], dtype=np.float64)
        real_distance = agent.radius - distance
        delta_v_iw = - np.dot(agent.linear_velocity, t_iw)
        ## GUO with compression and friction modified to account for the direction of the sliding force through delta_viw
        agent.obstacle_force += (agent.Aw * math.exp((real_distance) / agent.Bw) + agent.k1 * max(0,real_distance)) * n_iw + (-agent.Cw * math.exp((real_distance) / agent.Dw) - agent.k2 * max(0,real_distance)) * delta_v_iw * t_iw
        ## GUO standard with friction and compression
        # agent.obstacle_force += (agent.Aw * math.exp((real_distance) / agent.Bw) + agent.k1 * max(0,real_distance)) * n_iw + (agent.Cw * math.exp((real_distance) / agent.Dw) - agent.k2 * max(0,real_distance) * delta_v_iw) * t_iw
        ## GUO standard without friction and compression
        # agent.obstacle_force += (agent.Aw * math.exp((real_distance) / agent.Bw)) * n_iw + (agent.Cw * math.exp((real_distance) / agent.Dw)) * t_iw 

def compute_obstacle_force_roboticsupo(agent:HumanAgent):
    agent.obstacle_force = np.array([0.0,0.0], dtype=np.float64)
    for obstacle in agent.obstacles:
        min_diff = np.array(agent.position) - obstacle
        distance = np.linalg.norm(min_diff) - agent.radius
        agent.obstacle_force += agent.obstacle_weight * math.exp(-distance / agent.obstacle_sigma) * (min_diff / np.linalg.norm(min_diff))
    if (agent.obstacles): agent.obstacle_force /= len(agent.obstacles)

def compute_social_force_helbing(index:int, agents:list[HumanAgent], robot:RobotAgent, consider_robot:bool):
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
        real_distance = r_ij - distance
        target_agent.social_force += (target_agent.Ai * math.exp((real_distance) / target_agent.Bi) + target_agent.k1 * max(0,real_distance)) * n_ij + target_agent.k2  * max(0,real_distance) * delta_v_ij * t_ij

def compute_social_force_guo(index:int, agents:list[HumanAgent], robot:RobotAgent, consider_robot:bool):
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
        real_distance = r_ij - distance
        ## GUO with compression and friction
        delta_v_ij = np.dot(entities[i].linear_velocity - target_agent.linear_velocity, t_ij)
        target_agent.social_force += (target_agent.Ai * math.exp((real_distance) / target_agent.Bi) + target_agent.k1 * max(0,real_distance)) * n_ij + (target_agent.Ci * math.exp((real_distance) / target_agent.Di) + target_agent.k2  * max(0,real_distance) * delta_v_ij) * t_ij
        ## GUO without compression and friction
        # target_agent.social_force += (target_agent.Ai * math.exp((real_distance) / target_agent.Bi)) * n_ij + (target_agent.Ci * math.exp((real_distance) / target_agent.Di)) * t_ij

def compute_social_force_moussaid(index:int, agents:list[HumanAgent], robot:RobotAgent, consider_robot:bool):
    target_agent = agents[index]
    target_agent.social_force = np.array([0.0,0.0], dtype=np.float64)
    entities = agents.copy()
    if consider_robot: entities.append(robot)
    for i in range(len(entities)):
        if (i == index): continue
        difference = target_agent.position - entities[i].position
        distance = np.linalg.norm(difference)
        n_ij = difference / distance
        interaction_vector = target_agent.agent_lambda * (target_agent.linear_velocity - entities[i].linear_velocity) - n_ij
        interaction_norm = np.linalg.norm(interaction_vector)
        i_ij = (interaction_vector) / interaction_norm
        theta_ij = bound_angle(np.arctan2(n_ij[1],n_ij[0]) - np.arctan2(i_ij[1],i_ij[0]) + math.pi) #+ 0.00000001) # Add the bias to obtain a symethric behaviour (everyone has a preferred direction)
        k_ij = np.sign(theta_ij)
        h_ij = np.array([-i_ij[1], i_ij[0]], dtype=np.float64)
        F_ij = target_agent.gamma * interaction_norm
        r_ij = target_agent.radius + entities[i].radius
        real_distance = r_ij - distance
        ## MOUSSAID with compression and friction (in n_ij, t_ij)
        # t_ij = np.array([-n_ij[1],n_ij[0]], dtype=np.float64)
        # delta_v_ij_t_ij = np.dot(entities[i].linear_velocity - target_agent.linear_velocity, t_ij)
        # target_agent.social_force -= target_agent.Ei * math.exp(-distance/F_ij) * (math.exp(-(target_agent.ns1 * F_ij * theta_ij)**2) * i_ij + k_ij * math.exp(-(target_agent.ns * F_ij * theta_ij)**2) * h_ij) - target_agent.k1 * max(0,real_distance) * n_ij - target_agent.k2  * max(0,real_distance) * delta_v_ij_t_ij * t_ij
        ## MOUSSAID with compression and friction (in i_ij, h_ij)
        delta_v_ij_h_ij = np.dot(entities[i].linear_velocity - target_agent.linear_velocity, h_ij)
        target_agent.social_force -= target_agent.Ei * math.exp(-distance/F_ij) * (math.exp(-(target_agent.ns1 * F_ij * theta_ij)**2) * i_ij + k_ij * math.exp(-(target_agent.ns * F_ij * theta_ij)**2) * h_ij) + target_agent.k1 * max(0,real_distance) * i_ij + target_agent.k2  * max(0,real_distance) * delta_v_ij_h_ij * h_ij
        ## MOUSSAID no compression and friction
        # target_agent.social_force -= target_agent.Ei * math.exp(-distance/F_ij) * (math.exp(-(target_agent.ns1 * F_ij * theta_ij)**2) * i_ij + k_ij * math.exp(-(target_agent.ns * F_ij * theta_ij)**2) * h_ij)        

def compute_social_force_roboticsupo(index:int, agents:list[HumanAgent], robot:RobotAgent, consider_robot:bool):
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
        b = target_agent.agent_gamma * interaction_length
        force_velocity_amount = -math.exp(-np.linalg.norm(diff) / b - (target_agent.agent_nPrime * b * theta) ** 2)
        force_angle_amount = np.sign(-theta) * math.exp(-np.linalg.norm(diff) / b - (target_agent.agent_n * b * theta) ** 2)
        force_velocity = force_velocity_amount * interaction_direction
        force_angle = force_angle_amount * np.array([-interaction_direction[1], interaction_direction[0]])
        target_agent.social_force += target_agent.social_weight * (force_velocity + force_angle)

def compute_group_force_dummy(index:int, agents:list[HumanAgent], desired_direction:np.array, groups:dict):
    # target_agent = agents[index]
    # target_agent.group_force = np.array([0.0,0.0], dtype= np.float64)
    pass

def compute_group_force_roboticsupo(index:int, agents:list[HumanAgent], desired_direction:np.array, groups:dict):
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

def compute_torque_force_farina(agent:HumanAgent):
    desired_force_norm = np.linalg.norm(agent.desired_force)
    agent.k_theta = agent.inertia * agent.k_lambda * desired_force_norm
    agent.k_omega = agent.inertia * (1 + agent.alpha) * math.sqrt((agent.k_lambda * desired_force_norm) / agent.alpha)
    agent.torque_force = - agent.k_theta * bound_angle(agent.yaw - math.atan2(agent.desired_force[1],agent.desired_force[0])) - agent.k_omega * agent.angular_velocity

def compute_torque_force_new(agent:HumanAgent):
    forces_sum = agent.desired_force + agent.obstacle_force + agent.social_force
    forces_sum_norm = np.linalg.norm(forces_sum)
    agent.k_theta = agent.inertia * agent.k_lambda * forces_sum_norm
    agent.k_omega = agent.inertia * (1 + agent.alpha) * math.sqrt((agent.k_lambda * forces_sum_norm) / agent.alpha)
    agent.torque_force = - agent.k_theta * bound_angle(agent.yaw - math.atan2(forces_sum[1],forces_sum[0])) - agent.k_omega * agent.angular_velocity