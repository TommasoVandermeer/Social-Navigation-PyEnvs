import math
import numpy as np
from numba import njit, prange
from social_gym.src.utils import jitted_bound_angle, two_dim_norm, two_dim_dot_product, two_by_two_matrix_mul_two_dim_array, bound_two_dim_array_norm

# TODO: Add robot moving using the SFM
# TODO: Add safety space
# TODO: Check obstacle force computation, consider saving pairwise contribution in array and sum them outside the parallel loop (right now implementation is wrong)

@njit(nogil=True)
def compute_rotational_matrix_parallel(agent_state:np.ndarray):
    """
    Computes the rotational matrix of the agent.

    args:
    - agent_state (numpy.ndarray): in the form [px,py,theta,vx,vy,bvx,bvy,omega,r,m,gx,gy,vd]

    returns:
    - rotational_matrix (numpy.ndarray): in the form [[r00,r01],[r10,r11]]
    """
    return np.array([[math.cos(agent_state[2]), -math.sin(agent_state[2])],[math.sin(agent_state[2]), math.cos(agent_state[2])]], np.float64)

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
    desired_force = np.array([0.0,0.0], dtype=np.float64)
    difference = agent_state[10:12] - agent_state[0:2]
    distance = two_dim_norm(difference)
    if distance > agent_state[8]:
        desired_direction = difference / distance
        desired_force = agent_state[9] * ((desired_direction * agent_state[12]) - agent_state[3:5]) / agent_params[0]
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
    - social_force (numpy.ndarray): in the form [sfx, sfy] for agent idx
    """
    n_humans = len(agents_state)
    state = agents_state[idx]
    social_force = np.zeros((n_humans,2), np.float64)
    for j in prange(n_humans):
        if idx==j: continue
        # TODO: Add safety_space to rij (and so the the states of each agent)
        rij = state[8] + agents_state[j][8]
        difference = state[0:2] - agents_state[j][0:2]
        distance = two_dim_norm(difference)
        nij = difference / distance
        real_distance = rij - distance
        if type < 2:
            tij = np.array([-nij[1],nij[0]])
            delta_vij = two_dim_dot_product(agents_state[j][3:5] - state[3:5],tij)
            if type == 0: social_force[j] = (agent_params[1] * math.exp((real_distance) / agent_params[3]) + agent_params[10] * max(0.0,real_distance)) * nij + agent_params[11]  * max(0.0,real_distance) * delta_vij * tij
            elif type == 1: social_force[j] = (agent_params[1] * math.exp((real_distance) / agent_params[3]) + agent_params[10] * max(0.0,real_distance)) * nij + (agent_params[5] * math.exp((real_distance) / agent_params[7]) + agent_params[11]  * max(0.0,real_distance) * delta_vij) * tij
        elif type == 2:
            velocity_difference = state[3:5] - agents_state[j][3:5]
            interaction_vector = agent_params[12] * (velocity_difference) - nij
            interaction_norm = two_dim_norm(interaction_vector)
            iij = (interaction_vector) / interaction_norm
            theta_ij = jitted_bound_angle(np.arctan2(nij[1],nij[0]) - np.arctan2(iij[1],iij[0]) + math.pi)
            kij = np.sign(theta_ij)
            hij = np.array([-iij[1], iij[0]], dtype=np.float64)
            Fij = agent_params[13] * interaction_norm
            delta_vij = two_dim_dot_product(-velocity_difference, hij)
            # TODO: Check if the formula is correct
            social_force[j] = -(agent_params[9] * math.exp(-distance/Fij) * (math.exp(-(agent_params[15]*Fij*theta_ij)**2) * iij + kij * math.exp(-(agent_params[14]*Fij*theta_ij)**2)*hij) + agent_params[10] * max(0.0,real_distance) * iij + agent_params[11] * max(0.0,real_distance) * delta_vij * hij)
    return np.sum(social_force, axis=0)

@njit(nogil=True, parallel=True)
def compute_all_social_force_parallel(type:int, agents_state:np.ndarray, agent_params:np.ndarray):
    """
    Computes the repulsive force exherted by other pedestrians to each other agent by means of the Social Force Model.
    WARNING: This function assumes that all agents parameters are the same.

    args:
    - type (int) => 0: Helbing, 1: Guo, 2: Moussaid
    - agents_state (numpy.ndarray): state of each agent where each state is in the form [px,py,theta,vx,vy,bvx,bvy,omega,r,m,gx,gy,vd]
    - agent_params (numpy.ndarray): in the form [relax_t,Ai,Aw,Bi,Bw,Ci,Cw,Di,Dw,Ei,k1,k2,a_lambda,gamma,ns,ns1,ko,kd,alpha,k_lambda]

    returns:
    - social_force (numpy.ndarray): for each agent in the form [sfx, sfy]
    """
    n_agents = len(agents_state)
    social_force = np.zeros((n_agents,n_agents,2), np.float64)
    for idx in prange(n_agents*n_agents):
        i = idx // n_agents
        j = idx % n_agents
        if i >= j: continue
        else:
            # TODO: Add safety_space to rij (and so the the states of each agent)
            pairwise_social_force = np.zeros((2,), np.float64)
            rij = agents_state[i][8] + agents_state[j][8]
            difference = agents_state[i][0:2] - agents_state[j][0:2]
            distance = two_dim_norm(difference)
            nij = difference / distance
            real_distance = rij - distance
            if type < 2:
                tij = np.array([-nij[1],nij[0]])
                delta_vij = two_dim_dot_product(agents_state[j][3:5] - agents_state[i][3:5],tij)
                if type == 0: pairwise_social_force = (agent_params[1] * math.exp((real_distance) / agent_params[3]) + agent_params[10] * max(0.0,real_distance)) * nij + agent_params[11]  * max(0.0,real_distance) * delta_vij * tij 
                elif type == 1: pairwise_social_force = (agent_params[1] * math.exp((real_distance) / agent_params[3]) + agent_params[10] * max(0.0,real_distance)) * nij + (agent_params[5] * math.exp((real_distance) / agent_params[7]) + agent_params[11]  * max(0.0,real_distance) * delta_vij) * tij
            elif type == 2:
                velocity_difference = agents_state[i][3:5] - agents_state[j][3:5]
                interaction_vector = agent_params[12] * (velocity_difference) - nij
                interaction_norm = two_dim_norm(interaction_vector)
                iij = (interaction_vector) / interaction_norm
                theta_ij = jitted_bound_angle(np.arctan2(nij[1],nij[0]) - np.arctan2(iij[1],iij[0]) + math.pi)
                kij = np.sign(theta_ij)
                hij = np.array([-iij[1], iij[0]], dtype=np.float64)
                Fij = agent_params[13] * interaction_norm
                delta_vij = two_dim_dot_product(-velocity_difference, hij)
                # TODO: Check if the formula is correct
                pairwise_social_force = -(agent_params[9] * math.exp(-distance/Fij) * (math.exp(-(agent_params[15]*Fij*theta_ij)**2) * iij + kij * math.exp(-(agent_params[14]*Fij*theta_ij)**2)*hij) + agent_params[10] * max(0.0,real_distance) * iij + agent_params[11] * max(0.0,real_distance) * delta_vij * hij)
            social_force[i,j] = pairwise_social_force
            social_force[j,i] = -pairwise_social_force
    return np.sum(social_force, axis=1)    

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
        if type == 0: obstacle_force += (agent_params[2] * math.exp((real_distance) / agent_params[4]) + agent_params[10] * max(0.0,real_distance)) * niw - agent_params[11] * max(0.0,real_distance) * delta_viw * tiw
        elif type == 1: obstacle_force += (agent_params[2] * math.exp((real_distance) / agent_params[4]) + agent_params[10] * max(0.0,real_distance)) * niw + (-agent_params[6] * math.exp((real_distance) / agent_params[8]) - agent_params[11] * max(0.0,real_distance)) * delta_viw * tiw
    obstacle_force /= n_obstacles
    return obstacle_force

@njit(nogil=True)
def compute_torque_force_parallel(agent_state:np.ndarray, inertia:np.float64, driving_force:np.ndarray, agent_params:np.ndarray):
    """
    Computes the torque force by means of the Headed Social Force Model.

    args:
    - agent_state (numpy.ndarray): in the form [px,py,theta,vx,vy,bvx,bvy,omega,r,m,gx,gy,vd]
    - inertia (float): inertia of the agent
    - driving_force (numpy.ndarray): force driving the pedestrian's heading (For Farina HSFM it is the desired force)
    - agent_params (numpy.ndarray): in the form [relax_t,Ai,Aw,Bi,Bw,Ci,Cw,Di,Dw,Ei,k1,k2,a_lambda,gamma,ns,ns1,ko,kd,alpha,k_lambda]

    returns:
    - torque_force (float)
    """
    driving_force_norm = two_dim_norm(driving_force)
    k_theta = inertia * agent_params[19] * driving_force_norm
    k_omega = inertia * (1 + agent_params[18]) * math.sqrt((agent_params[19] * driving_force_norm) / agent_params[18])
    torque_force = - k_theta * jitted_bound_angle(agent_state[2] - math.atan2(driving_force[1],driving_force[0])) - k_omega * agent_state[7]
    return torque_force

@njit(nogil=True, parallel=True)
def update_humans_parallel(type:int, agents_state:np.ndarray, goals:np.ndarray, obstacles:np.ndarray, agents_params:np.ndarray, dt:float, all_params_equal=False, last_is_robot=False): 
    """
    Makes a step (for humans) of dt length by means of the Headed / Social Force Model (depends on which type is passed).

    args:
    - type (int):
        -- 0: SFM Helbing
        -- 1: SFM Guo
        -- 2: SFM Moussaid
        -- 3: HSFM Farina
        -- 4: HSFM Guo
        -- 5: HSFM Moussaid
        -- 6: HSFM New
        -- 7: HSFM New Guo
        -- 8: HSFM New Moussaid
    - agents_state (numpy.ndarray): state of each agent where each state is in the form [px,py,theta,vx,vy,bvx,bvy,omega,r,m,gx,gy,vd]
    - goals (numpy.ndarray): goals for each agent where each element is in the form [g1,g2,g3,...] and each one is in the form [gpx, gpy]
    - obstacles (numpy.ndarray): array of all the obstacles [o1,o2,...] in the environment, each oi is an array [si1,si2,...] containing the obstacle segments,
      each segment sij is 2d array containing vertices [vij1, vij2] and each vertex vijk is a 2d vector [vijkx,vijky]
    - agent_params (numpy.ndarray): params of each agent where each one is in the form [relax_t,Ai,Aw,Bi,Bw,Ci,Cw,Di,Dw,Ei,k1,k2,a_lambda,gamma,ns,ns1,ko,kd,alpha,k_lambda]

    returns:
    - updated_state (numpy.ndarray): state of each agent where each state is in the form [px,py,theta,vx,vy,bvx,bvy,omega,r,m,gx,gy,vd]
    """
    if type < 0 or type > 8: raise ValueError(f"Type {type} does not exist for this implementation")
    n_agents = len(agents_state)
    if last_is_robot: n_agents -= 1
    updated_state = np.copy(agents_state)
    soc_force_type = type % 3
    obs_force_type = 0 if type in [0,2,3,5,6,8] else 1
    headed = type // 3
    ## Compute forces
    global_forces = np.zeros((n_agents,2), np.float64)
    if all_params_equal: social_forces = compute_all_social_force_parallel(soc_force_type, agents_state, agents_params[0])
    if headed > 0: 
        torque_forces = np.zeros((n_agents,), np.float64)
        inertias = np.zeros((n_agents,), np.float64)
    for i in prange(n_agents):
        ## Update goal
        if two_dim_norm(goals[i][0] - agents_state[i,0:2]) <= agents_state[i,8]:
            if np.isnan(goals[i]).any(): first_nan_idx = np.argwhere(np.isnan(goals[i]))[0][0]
            else: first_nan_idx = len(goals[i])
            reached_goal = np.copy(goals[i][0])
            for gidx in range(first_nan_idx):
                if gidx < first_nan_idx - 1: goals[i][gidx,:] = goals[i][gidx+1,:]
                else: goals[i][gidx,:] = reached_goal
            agents_state[i,10:12] = goals[i][0]
            updated_state[i,10:12] = goals[i][0]
        ## Update obstacles
        if obstacles is not None:
            n_obstacles = len(obstacles)
            n_segments = len(obstacles[0])
            obstacles_closest_points = np.zeros((n_obstacles,2), np.float64)
            distances = np.zeros((n_obstacles, n_segments,), np.float64)
            closest_points = np.zeros((n_obstacles, n_segments,2), np.float64)
            for j in prange(n_obstacles * n_segments):
                oidx = j // n_segments
                sidx = j % n_segments
                if np.isnan(obstacles[oidx,sidx,0,0]):
                    distances[oidx,sidx] = np.iinfo(np.int64).max
                else:
                    t = (two_dim_dot_product(agents_state[i,0:2] - obstacles[oidx][sidx][0], obstacles[oidx][sidx][1] - obstacles[oidx][sidx][0])) / (two_dim_norm(obstacles[oidx][sidx][1] - obstacles[oidx][sidx][0]) ** 2)
                    t_star = min(max(0, t), 1)
                    closest_points[oidx,sidx] = obstacles[oidx][sidx][0] + t_star * (obstacles[oidx][sidx][1] - obstacles[oidx][sidx][0])
                    distances[oidx,sidx] = two_dim_norm(closest_points[oidx,sidx] - agents_state[i,0:2])
            for j in range(n_obstacles): obstacles_closest_points[j] = closest_points[j,np.argmin(distances[j])]
        ## Compute rotational matrix and current linear velocity
        if headed > 0: 
            rotational_matrix = compute_rotational_matrix_parallel(agents_state[i])
            agents_state[i,3:5] = two_by_two_matrix_mul_two_dim_array(rotational_matrix, agents_state[i,5:7])
        desired_force = compute_desired_force_parallel(agents_state[i], agents_params[i])
        if obstacles is not None: obstacle_force = compute_obstacle_force_parallel(obs_force_type, agents_state[i], obstacles_closest_points, agents_params[i])
        else: obstacle_force = np.zeros((2,), np.float64)
        if all_params_equal: social_force = social_forces[i]
        else: social_force = compute_social_force_parallel(soc_force_type, i, agents_state, agents_params[i])
        input_force = desired_force + obstacle_force + social_force
        ## Compute final forces
        if headed == 0:
            global_forces[i] = input_force
        else:
            inertias[i] = 0.5 * agents_state[i,9] * agents_state[i,8] * agents_state[i,8]
            if headed == 1: torque_forces[i] = compute_torque_force_parallel(agents_state[i], inertias[i], desired_force, agents_params[i])
            elif headed == 2: torque_forces[i] = compute_torque_force_parallel(agents_state[i], inertias[i], input_force, agents_params[i])
            global_forces[i,0] = two_dim_dot_product(input_force, rotational_matrix[:,0])
            global_forces[i,1] = agents_params[i,16] * two_dim_dot_product(obstacle_force + social_force, rotational_matrix[:,1]) - agents_params[i,17] * agents_state[i,6]
        ## Update state
        updated_state[i,0:2] += updated_state[i,3:5] * dt # Position
        if headed > 0:
            updated_state[i,2] = jitted_bound_angle(updated_state[i,2] + updated_state[i,7] * dt) # Theta
            updated_state[i,5:7] += (global_forces[i] / updated_state[i,9]) * dt # Body velocity
            updated_state[i,5:7] = bound_two_dim_array_norm(updated_state[i,5:7], updated_state[i,12]) # Bound body velocity
            updated_state[i,7] += (torque_forces[i] / inertias[i]) * dt # Angular velocity
            rotational_matrix = np.array([[math.cos(updated_state[i,2]), -math.sin(updated_state[i,2])],[math.sin(updated_state[i,2]), math.cos(updated_state[i,2])]], np.float64)
            updated_state[i,3:5] = two_by_two_matrix_mul_two_dim_array(rotational_matrix, updated_state[i,5:7])
        else:
            updated_state[i,3:5] += (global_forces[i] / updated_state[i,9]) * dt # Linear velocity
            updated_state[i,3:5] = bound_two_dim_array_norm(updated_state[i,3:5], updated_state[i,12]) # Bound linear velocity
    return updated_state

@njit(nogil=True)
def update_robot_parallel(type:int, agents_state:np.ndarray, goals:np.ndarray, obstacles:np.ndarray, robot_params:np.ndarray, dt:float, all_params_equal=False):
    """
    Makes a step (for the robot) of dt length by means of the Headed / Social Force Model (depends on which type is passed).

    args:
    - type (int):
        -- 0: SFM Helbing
        -- 1: SFM Guo
        -- 2: SFM Moussaid
        -- 3: HSFM Farina
        -- 4: HSFM Guo
        -- 5: HSFM Moussaid
        -- 6: HSFM New
        -- 7: HSFM New Guo
        -- 8: HSFM New Moussaid
    - agents_state (numpy.ndarray): state of each agent where each state is in the form [px,py,theta,vx,vy,bvx,bvy,omega,r,m,gx,gy,vd]
    - goals (numpy.ndarray): goals of the robot in the form [g1,g2,g3,...] and each one is in the form [gpx, gpy]
    - obstacles (numpy.ndarray): array of all the obstacles [o1,o2,...] in the environment, each oi is an array [si1,si2,...] containing the obstacle segments,
      each segment sij is 2d array containing vertices [vij1, vij2] and each vertex vijk is a 2d vector [vijk_x,vijk_y]
    - robot_params (numpy.ndarray): params of the robot in the form [relax_t,Ai,Aw,Bi,Bw,Ci,Cw,Di,Dw,Ei,k1,k2,a_lambda,gamma,ns,ns1,ko,kd,alpha,k_lambda]

    returns:
    - updated_state (numpy.ndarray): updated state of the robot in the form [px,py,theta,vx,vy,bvx,bvy,omega,r,m,gx,gy,vd]
    """
    if type < 0 or type > 8: raise ValueError(f"Type {type} does not exist for this implementation")
    updated_state = np.copy(agents_state)
    return updated_state