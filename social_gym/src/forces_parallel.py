import math
import numpy as np
from social_gym.src.utils import bound_angle, two_dim_norm, two_dim_dot_product
from numba import njit, prange

@njit(nogil=True)
def compute_desired_force_parallel(agent_state:np.ndarray, agent_params:np.ndarray):
    """
    Computes the attractive force of the agent's goal by means of the Social Force Model.

    args:
    - agent_state (numpy.ndarray) in the form [px,py,theta,vx,vy,bvx,bvy,omega,r,m,gx,gy,vd]
    - agent_params (numpy.ndarray) in the form [relax_t,Ai,Aw,Bi,Bw,Ci,Cw,Di,Dw,Ei,k1,k2,a_lambda,gamma,ns,ns1,ko,kd,alpha,k_lambda]

    returns:
    - desired_force (numpy.ndarray) in the form [dfx, dfy]
    """
    difference = agent_state[10:12] - agent_state[0:2]
    distance = two_dim_norm(difference)
    if distance > agent_state[8]:
        desired_direction = difference / distance
        desired_force = agent_state[9] * (desired_direction * agent_state[12] - agent_state[3:5]) / agent_params[0]
    else: desired_force = np.array([0.0,0.0], dtype=np.float64)
    return desired_force

@njit(nogil=True, parallel=True)
def compute_social_force_parallel(type:int, idx:int, agents_state:np.ndarray, agent_params:np.ndarray):
    """
    Computes the repulsive force exherted by other pedestrians to agent idx by means of the Social Force Model.

    args:
    - type (int) => 0: Helbing, 1: Guo, 2: Moussaid
    - idx (int): index of the agent for which the social_force between each interaction is computed
    - agents_state (numpy.ndarray): state of each agent where each state is in the form [px,py,theta,vx,vy,bvx,bvy,omega,r,m,gx,gy,vd]
    - agent_params (numpy.ndarray): in the form [relax_t,Ai,Aw,Bi,Bw,Ci,Cw,Di,Dw,Ei,k1,k2,a_lambda,gamma,ns,ns1,ko,kd,alpha,k_lambda]

    returns:
    - social_force (numpy.ndarray): in the form [sfx, sfy] for each pair of agents (idx,j), j = 0,1,...,n_humans
    """
    n_humans = len(agents_state)
    state = agents_state[idx]
    social_force = np.array([0.0,0.0])
    for j in prange(n_humans):
        if idx==j: continue
        # TODO: Add safety_space to rij (and so the the states of each agent)
        rij = state[8] + agents_state[j][8]
        difference = state[0:2] - agents_state[j][0:2]
        distance = two_dim_norm(difference)
        nij = difference / distance
        real_distance = rij - distance
        if type != 2:
            tij = np.array([-nij[1],nij[0]])
            delta_vij = two_dim_dot_product(agents_state[j][3:5] - state[3:5],tij)
        if type == 0: social_force += (agent_params[1] * math.exp((real_distance) / agent_params[3]) + agent_params[10] * max(0,real_distance)) * nij + agent_params[11]  * max(0,real_distance) * delta_vij * tij
        elif type == 1: social_force += (agent_params[1] * math.exp((real_distance) / agent_params[3]) + agent_params[10] * max(0,real_distance)) * nij + (agent_params[5] * math.exp((real_distance) / agent_params[7]) + agent_params[11]  * max(0,real_distance) * delta_vij) * tij
        elif type == 2:
            velocity_difference = state[3:5] - agents_state[j][3:5]
            interaction_vector = agent_params[12] * (velocity_difference) - nij
            interaction_norm = np.linalg.norm(interaction_vector)
            iij = (interaction_vector) / interaction_norm
            theta_ij = bound_angle(np.arctan2(nij[1],nij[0]) - np.arctan2(iij[1],iij[0]) + math.pi)
            kij = np.sign(theta_ij)
            hij = np.array([-iij[1], iij[0]], dtype=np.float64)
            Fij = agent_params[13] * interaction_norm
            delta_vij = two_dim_dot_product(-velocity_difference, hij)
            social_force -= agent_params[9] * math.exp(-distance/Fij) * (math.exp(-(agent_params[15]*Fij*theta_ij)**2) * iij + kij * math.exp(-(agent_params[14]*Fij*theta_ij)**2)*hij) + agent_params[10] * max(0,real_distance) * iij + agent_params[11] * max(0,real_distance) * delta_vij * hij
        else: raise ValueError(f"Type {type} does not exist for social force computation")
    return social_force

@njit(nogil=True, parallel=True)
def compute_obstacle_force_parallel(type:int, agent_state:np.ndarray, obstacles:np.ndarray, agent_params:np.ndarray):
    """
    Computes the repulsive force exherted by obstacles by means of the Social Force Model.

    args:
    - type (int) => 0: Helbing, 1: Guo
    - agent_state (numpy.ndarray): in the form [px,py,theta,vx,vy,bvx,bvy,omega,r,m,gx,gy,vd]
    - obstacles (numpy.ndarray): array of obstacles where each one is in the form [opx, opy]
    - agent_params (numpy.ndarray): in the form [relax_t,Ai,Aw,Bi,Bw,Ci,Cw,Di,Dw,Ei,k1,k2,a_lambda,gamma,ns,ns1,ko,kd,alpha,k_lambda]

    returns:
    - obstacle_force (numpy.ndarray): in the form [ofx, ofy] 
    """
    obstacle_force = np.array([0.0,0.0], np.float64)
    n_obstacles = len(obstacles)
    for i in prange(n_obstacles):
        # TODO: Add safety_space to real_distance (and so the the states of each agent)
        difference = agent_state[0:2] - obstacles[i]
        distance = two_dim_norm(difference)
        niw = difference / distance
        tiw = np.array([-niw[1],niw[0]], dtype=np.float64)
        delta_viw = - two_dim_dot_product(agent_state[3:5], tiw)
        real_distance = agent_state[8] - distance
        if type == 0: obstacle_force += (agent_params[2] * math.exp((real_distance) / agent_params[4]) + agent_params[10] * max(0,real_distance)) * niw - agent_params[11] * max(0,real_distance) * delta_viw * tiw
        elif type == 1: obstacle_force += (agent_params[2] * math.exp((real_distance) / agent_params[4]) + agent_params[10] * max(0,real_distance)) * niw + (-agent_params[6] * math.exp((real_distance) / agent_params[8]) - agent_params[11] * max(0,real_distance)) * delta_viw * tiw
        else: raise ValueError(f"Type {type} does not exist for obstacle force computation")
    obstacle_force /= n_obstacles
    return obstacle_force

@njit(nogil=True)
def compute_torque_force_parallel(agent_state:np.ndarray, driving_force:np.ndarray, agent_params:np.ndarray):
    """
    Computes the torque force by means of the Headed Social Force Model.

    args:
    - agent_state (numpy.ndarray): in the form [px,py,theta,vx,vy,bvx,bvy,omega,r,m,gx,gy,vd]
    - driving_force (numpy.ndarray): force driving the pedestrian's heading (For Farina HSFM it is the desired force)
    - agent_params (numpy.ndarray): in the form [relax_t,Ai,Aw,Bi,Bw,Ci,Cw,Di,Dw,Ei,k1,k2,a_lambda,gamma,ns,ns1,ko,kd,alpha,k_lambda]

    returns:
    - torque_force (float)
    """
    driving_force_norm = two_dim_norm(driving_force)
    inertia = 0.5 * agent_state[9] * agent_state[8] * agent_state[8]
    k_theta = inertia * agent_params[19] * driving_force_norm
    k_omega = inertia * (1 + agent_params[18]) * math.sqrt((agent_params[19] * driving_force_norm) / agent_params[18])
    torque_force = - k_theta * bound_angle(agent_state[2] - math.atan2(driving_force[1],driving_force[0])) - k_omega * agent_state[7]
    return torque_force