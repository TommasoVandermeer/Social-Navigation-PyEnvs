import numpy as np
import math
from typing import Union
from social_gym.src.state import ObservableState, FullState, FullStateHeaded
from social_gym.src.utils import bound_angle

def compute_desired_force(params:dict, state:Union[FullState,FullStateHeaded]):
    goal_position = np.array([state.gx, state.gy], dtype=np.float64)
    agent_position = np.array([state.px, state.py], dtype=np.float64)
    if not state.headed: agent_velocity = np.array([state.vx, state.vy], dtype=np.float64)
    else: 
        agent_body_velocity = np.array([state.vx, state.vy], dtype=np.float64)
        rotational_matrix = np.array([[np.cos(state.theta), -np.sin(state.theta)],[np.sin(state.theta), np.cos(state.theta)]], dtype=np.float64)
        agent_velocity = np.matmul(rotational_matrix, agent_body_velocity)
    difference = goal_position - agent_position
    distance = np.linalg.norm(difference)
    if (distance > state.radius):
        desired_direction = difference / distance
        desired_force = params['mass'] * (desired_direction * state.v_pref - agent_velocity) / params['relaxation_time']
    else: desired_direction = np.array([0.0,0.0], dtype=np.float64); desired_force = np.array([0.0,0.0], dtype=np.float64)
    return desired_direction, desired_force

def compute_social_force_helbing(params:dict, state:Union[FullState,FullStateHeaded], agents_state:list[ObservableState]):
    social_force = np.array([0.0,0.0], dtype=np.float64)
    agent_position = np.array([state.px, state.py], dtype=np.float64)
    if not state.headed: agent_velocity = np.array([state.vx, state.vy], dtype=np.float64)
    else: 
        agent_body_velocity = np.array([state.vx, state.vy], dtype=np.float64)
        rotational_matrix = np.array([[np.cos(state.theta), -np.sin(state.theta)],[np.sin(state.theta), np.cos(state.theta)]], dtype=np.float64)
        agent_velocity = np.matmul(rotational_matrix, agent_body_velocity)
    for other_state in agents_state:
        other_agent_position = np.array([other_state.px, other_state.py], dtype=np.float64)
        other_agent_velocity = np.array([other_state.vx, other_state.vy], dtype=np.float64)
        difference = agent_position - other_agent_position
        distance = np.linalg.norm(difference)
        n_ij = difference / distance
        t_ij = np.array([-n_ij[1],n_ij[0]], dtype=np.float64)
        delta_v_ij = np.dot(other_agent_velocity - agent_velocity, t_ij)
        real_distance = state.radius + other_state.radius - distance
        social_force += (params['Ai'] * math.exp(real_distance / params['Bi']) + params['k1'] * max(0, real_distance)) * n_ij + params['k2'] * max(0, real_distance) * delta_v_ij * t_ij
    return social_force

def compute_social_force_guo(params:dict, state:Union[FullState,FullStateHeaded], agents_state:list[ObservableState]):
    social_force = np.array([0.0,0.0], dtype=np.float64)
    agent_position = np.array([state.px, state.py], dtype=np.float64)
    if not state.headed: agent_velocity = np.array([state.vx, state.vy], dtype=np.float64)
    else: 
        agent_body_velocity = np.array([state.vx, state.vy], dtype=np.float64)
        rotational_matrix = np.array([[np.cos(state.theta), -np.sin(state.theta)],[np.sin(state.theta), np.cos(state.theta)]], dtype=np.float64)
        agent_velocity = np.matmul(rotational_matrix, agent_body_velocity)
    for other_state in agents_state:
        other_agent_position = np.array([other_state.px, other_state.py], dtype=np.float64)
        other_agent_velocity = np.array([other_state.vx, other_state.vy], dtype=np.float64)
        difference = agent_position - other_agent_position
        distance = np.linalg.norm(difference)
        n_ij = difference / distance
        t_ij = np.array([-n_ij[1],n_ij[0]], dtype=np.float64)
        real_distance = state.radius + other_state.radius - distance
        ## GUO with compression and friction
        delta_v_ij = np.dot(other_agent_velocity - agent_velocity, t_ij)
        social_force += (params['Ai'] * math.exp(real_distance / params['Bi']) + params['k1'] * max(0, real_distance)) * n_ij + (params['Ci'] * math.exp(real_distance / params['Di']) + params['k2'] * max(0, real_distance) * delta_v_ij) * t_ij
        ## GUO without compression and friction
        # social_force += (params['Ai'] * math.exp(real_distance / params['Bi'])) * n_ij + (params['Ci'] * math.exp(real_distance / params['Di'])) * t_ij
    return social_force

def compute_social_force_moussaid(params:dict, state:Union[FullState,FullStateHeaded], agents_state:list[ObservableState]):
    social_force = np.array([0.0,0.0], dtype=np.float64)
    agent_position = np.array([state.px, state.py], dtype=np.float64)
    if not state.headed: agent_velocity = np.array([state.vx, state.vy], dtype=np.float64)
    else: 
        agent_body_velocity = np.array([state.vx, state.vy], dtype=np.float64)
        rotational_matrix = np.array([[np.cos(state.theta), -np.sin(state.theta)],[np.sin(state.theta), np.cos(state.theta)]], dtype=np.float64)
        agent_velocity = np.matmul(rotational_matrix, agent_body_velocity)
    for other_state in agents_state:
        other_agent_position = np.array([other_state.px, other_state.py], dtype=np.float64)
        other_agent_velocity = np.array([other_state.vx, other_state.vy], dtype=np.float64)
        difference = agent_position - other_agent_position
        distance = np.linalg.norm(difference)
        real_distance = state.radius + other_state.radius - distance
        n_ij = difference / distance
        interaction_vector = params['agent_lambda'] * (agent_velocity - other_agent_velocity) - n_ij
        interaction_norm = np.linalg.norm(interaction_vector)
        i_ij = (interaction_vector) / interaction_norm
        theta_ij = bound_angle(np.arctan2(n_ij[1],n_ij[0]) - np.arctan2(i_ij[1],i_ij[0]) + math.pi) #+ 0.00000001) # Add the bias to obtain a symethric behaviour (everyone has a preferred direction)
        k_ij = np.sign(theta_ij)
        h_ij = np.array([-i_ij[1], i_ij[0]], dtype=np.float64)
        F_ij = params['gamma'] * interaction_norm
        ## MOUSSAID with compression and friction (in n_ij, t_ij)
        # t_ij = np.array([-n_ij[1],n_ij[0]], dtype=np.float64)
        # delta_v_ij_t_ij = np.dot(other_agent_velocity - agent_velocity, t_ij)
        # social_force -= params['Ei'] * math.exp(-distance/F_ij) * (math.exp(-(params['ns1'] * F_ij * theta_ij)**2) * i_ij + k_ij * math.exp(-(params['ns'] * F_ij * theta_ij)**2) * h_ij) - params['k1'] * max(0,real_distance) * n_ij - params['k2']  * max(0,real_distance) * delta_v_ij_t_ij * t_ij
        ## MOUSSAID with compression and friction (in i_ij, h_ij)
        delta_v_ij_h_ij = np.dot(other_agent_velocity - agent_velocity, h_ij)
        social_force -= params['Ei'] * math.exp(-distance/F_ij) * (math.exp(-(params['ns1'] * F_ij * theta_ij)**2) * i_ij + k_ij * math.exp(-(params['ns'] * F_ij * theta_ij)**2) * h_ij) + params['k1'] * max(0,real_distance) * i_ij + params['k2']  * max(0,real_distance) * delta_v_ij_h_ij * h_ij
        ## MOUSSAID no compression and friction
        # social_force -= params['Ei'] * math.exp(-distance/F_ij) * (math.exp(-(params['ns1'] * F_ij * theta_ij)**2) * i_ij + k_ij * math.exp(-(params['ns'] * F_ij * theta_ij)**2) * h_ij)
    return social_force

def compute_torque_force(params:dict, state:FullStateHeaded, inertia:float, driving_force:np.array):
    driving_force_norm = np.linalg.norm(driving_force)
    k_theta = inertia * params['k_lambda'] * driving_force_norm
    k_omega = inertia * (1 + params['alpha']) * math.sqrt((params['k_lambda'] * driving_force_norm) / params['alpha'])
    torque_force = - k_theta * bound_angle(state.theta - math.atan2(driving_force[1], driving_force[0])) - k_omega * state.w
    return torque_force