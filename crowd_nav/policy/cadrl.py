import torch
import torch.nn as nn
import numpy as np
import itertools
import logging
from crowd_nav.policy_no_train.policy import Policy
from crowd_nav.utils.action import ActionRot, ActionXY
from crowd_nav.utils.state import ObservableState, FullState
from numba import njit, prange
from social_gym.src.utils import two_dim_norm, jitted_point_to_segment_distance
import math

@njit(nogil=True)
def transform_state_to_agent_centric(state:np.ndarray):
    # Input state is in the form: [px,py,vx,vy,r,gx,gy,vd,theta,px1,py1,vx1,vy1,r1] (14)
    # Output state is in the form: [dg,v_pref,theta,radius,vx,vy,px1,py1,vx1,vy1,radius1,da,radius_sum] (13)
    transformed_state = np.empty((13,), np.float64)
    rot = math.atan2(state[6] - state[1], state[5] - state[0]) 
    transformed_state[0] = two_dim_norm(state[5:7] - state[0:2]) # dg
    transformed_state[1] = state[7] # vpref
    transformed_state[2] = 0.0 # theta
    transformed_state[3] = state[4] # radius
    transformed_state[4] = state[2] * math.cos(rot) + state[3] * math.sin(rot) # vx
    transformed_state[5] = state[3] * math.cos(rot) - state[2] * math.sin(rot) # vy
    transformed_state[6] = (state[9] - state[0]) * math.cos(rot) + (state[10] - state[1]) * math.sin(rot) # px1
    transformed_state[7] = (state[10] - state[1]) * math.cos(rot) - (state[9] - state[0]) * math.sin(rot) # py1 
    transformed_state[8] = state[11] * math.cos(rot) + state[12] * math.sin(rot) # vx1
    transformed_state[9] = state[12] * math.cos(rot) - state[11] * math.sin(rot) # vy1
    transformed_state[10] = state[13] # radius1
    transformed_state[11] = two_dim_norm(state[9:11] - state[0:2]) # da
    transformed_state[12] = state[4] + state[13] # radius_sum
    return transformed_state

@njit(nogil=True, parallel=True)
def compute_rotated_states_and_reward(action_space:np.ndarray, next_humans_state:np.ndarray, current_robot_state:np.ndarray, dt:np.float64):
    ### Humans states are in the form: [px,py,vx,vy,r]
    ### Robot state is in the form: [px,py,vx,vy,r,gx,gy,vd]
    n_actions = len(action_space)
    n_humans = len(next_humans_state)
    rewards = np.empty((n_actions,), np.float64)
    rotated_states = np.empty((n_actions, n_humans, 13), np.float64)
    for ii in prange(n_actions):
        ## Compute next robot position
        next_robot_position = current_robot_state[0:2] + action_space[ii] * dt
        # TODO: Use social_nav_sim method for computing reward
        ## Collision detection
        dmin = np.iinfo(np.int64).max
        collision = False
        for j in range(n_humans):
            difference = next_humans_state[j][0:2] - next_robot_position
            velocity_difference = next_humans_state[j][2:4] - action_space[ii]
            e = difference + velocity_difference * dt
            distance = jitted_point_to_segment_distance(difference[0], difference[1], e[0], e[1], 0, 0) - next_humans_state[j][4] - current_robot_state[4]
            # distance = two_dim_norm(next_humans_state[j][0:2] - next_robot_position) - next_humans_state[j][4] - current_robot_state[4]
            if distance < 0: collision = True; break
            elif (distance >= 0) and (distance < dmin): dmin = distance
        ## Check if robot reached goal
        distance_to_goal = two_dim_norm(next_robot_position - current_robot_state[5:7])
        reached_goal = distance_to_goal < current_robot_state[4]
        ## Compute reward
        if collision: rewards[ii] = -0.25
        elif reached_goal: rewards[ii] = 1
        elif dmin < 0.2 and not collision: rewards[ii] = (dmin - 0.2) * 0.5 * dt
        else: rewards[ii] = 0
        ## Compute rotated state
        next_robot_state = np.array([*next_robot_position, *action_space[ii], *current_robot_state[4:8], 0.0], np.float64) # WARNING: 0.0 is theta of the robot, needs to be changed if use unicycle
        for j in prange(n_humans):
            # Transform state to be compatible with the Value Network input
            state = np.concatenate((next_robot_state, next_humans_state[j]))
            rotated_states[ii,j] = transform_state_to_agent_centric(state)
    return rotated_states, rewards

@njit(nogil=True, parallel=True)
def compute_action_value(rewards:np.ndarray, value_network_min_outputs:np.ndarray, dt:np.float64, gamma:np.float64, vpref:np.float64):
    n_actions = len(rewards)
    action_values = np.empty((n_actions,), np.float64)
    for ii in prange(n_actions): action_values[ii] = rewards[ii] + pow(gamma, dt * vpref) * value_network_min_outputs[ii]
    return action_values

def mlp(input_dim, mlp_dims, last_relu=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
        if i != len(mlp_dims) - 2 or last_relu: layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    return net

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, mlp_dims):
        super().__init__()
        self.value_network = mlp(input_dim, mlp_dims)

    def forward(self, state):
        value = self.value_network(state)
        return value

class CADRL(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'CADRL'
        self.trainable = True
        self.multiagent_training = None
        self.kinematics = None
        self.epsilon = None
        self.gamma = None
        self.sampling = None
        self.speed_samples = None
        self.rotation_samples = None
        self.query_env = None
        self.action_space = None
        self.speeds = None
        self.rotations = None
        self.action_values = None
        self.with_om = None
        self.cell_num = None
        self.cell_size = None
        self.om_channel_size = None
        self.parallelize = False
        self.self_state_dim = 6
        self.human_state_dim = 7
        self.joint_state_dim = self.self_state_dim + self.human_state_dim

    def configure(self, config):
        self.set_common_parameters(config)
        mlp_dims = [int(x) for x in config.get('cadrl', 'mlp_dims').split(', ')]
        self.model = ValueNetwork(self.joint_state_dim, mlp_dims)
        self.multiagent_training = config.getboolean('cadrl', 'multiagent_training')
        logging.debug('Policy: CADRL without occupancy map')

    def set_common_parameters(self, config):
        self.gamma = config.getfloat('rl', 'gamma')
        self.kinematics = config.get('action_space', 'kinematics')
        self.sampling = config.get('action_space', 'sampling')
        self.speed_samples = config.getint('action_space', 'speed_samples')
        self.rotation_samples = config.getint('action_space', 'rotation_samples')
        self.query_env = config.getboolean('action_space', 'query_env')
        self.cell_num = config.getint('om', 'cell_num')
        self.cell_size = config.getfloat('om', 'cell_size')
        self.om_channel_size = config.getint('om', 'om_channel_size')

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        holonomic = True if self.kinematics == 'holonomic' else False
        speeds = [(np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref for i in range(self.speed_samples)]
        if holonomic:
            rotations = np.linspace(0, 2 * np.pi, self.rotation_samples, endpoint=False)
        else:
            rotations = np.linspace(-np.pi / 4, np.pi / 4, self.rotation_samples)

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        self.action_space_ndarray = np.empty((self.speed_samples * self.rotation_samples + 1,2), np.float64)
        self.action_space_ndarray[0] = np.array([0, 0], np.float64)
        ii = 1
        for rotation, speed in itertools.product(rotations, speeds):
            # Array for parallelization
            self.action_space_ndarray[ii] = np.array([speed * np.cos(rotation), speed * np.sin(rotation)], np.float64)
            # Standard array of actions
            if holonomic: action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else: action_space.append(ActionRot(speed, rotation))
            ii += 1

        self.speeds = speeds
        self.rotations = rotations
        self.action_space = action_space

    def propagate(self, state, action):
        if isinstance(state, ObservableState):
            # propagate state of humans
            next_px = state.px + action.vx * self.time_step
            next_py = state.py + action.vy * self.time_step
            next_state = ObservableState(next_px, next_py, action.vx, action.vy, state.radius)
        elif isinstance(state, FullState):
            # propagate state of current agent
            # perform action without rotation
            if self.kinematics == 'holonomic':
                next_px = state.px + action.vx * self.time_step
                next_py = state.py + action.vy * self.time_step
                next_state = FullState(next_px, next_py, action.vx, action.vy, state.radius,
                                       state.gx, state.gy, state.v_pref, state.theta)
            else:
                next_theta = state.theta + action.r
                next_vx = action.v * np.cos(next_theta)
                next_vy = action.v * np.sin(next_theta)
                next_px = state.px + next_vx * self.time_step
                next_py = state.py + next_vy * self.time_step
                next_state = FullState(next_px, next_py, next_vx, next_vy, state.radius, state.gx, state.gy,
                                       state.v_pref, next_theta)
        else:
            raise ValueError('Type error')

        return next_state

    def predict(self, state):
        """
        Input state is the joint state of robot concatenated by the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed
        """
        if self.phase is None or self.device is None: raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None: raise AttributeError('Epsilon attribute has to be set in training phase')
        if self.reach_destination(state): 
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None: self.build_action_space(state.self_state.v_pref)
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon: max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            if self.parallelize:
                r = state.self_state
                h = state.human_states
                current_robot_state = np.copy(np.array([r.px,r.py,r.vx,r.vy,r.radius,r.gx,r.gy,r.v_pref,r.theta], np.float64))
                next_humans_state = np.copy(np.array([[hi.px,hi.py,hi.vx,hi.vy,hi.radius] for hi in h], np.float64))
                ## Compute next human state querying env (not assuming constant velocity)
                next_humans_pos_and_vel = self.env.motion_model_manager.get_next_human_observable_states(self.time_step)  
                for i, hs in enumerate(next_humans_state): hs[0:4] = next_humans_pos_and_vel[i]
                ## Compute Value Network input and rewards
                rotated_states, rewards = compute_rotated_states_and_reward(self.action_space_ndarray, next_humans_state, current_robot_state, self.time_step)
                ## Compute Value Network output - BOTTLENECK
                value_network_min_outputs = np.zeros((len(rewards),), np.float64) 
                for ii in range(len(rewards)):
                    batch_next_states = torch.Tensor(rotated_states[ii]).to(self.device).reshape((len(rotated_states[ii]),13))
                    # batch_next_states = torch.cat([torch.Tensor([rotated_state]).to(self.device) for rotated_state in rotated_states[ii]], dim=0)
                    outputs = self.model(batch_next_states)  
                    min_output, _ = torch.min(outputs, 0)
                    value_network_min_outputs[ii] = min_output.data.item()
                ## Compute action value 
                action_values = compute_action_value(rewards, value_network_min_outputs, self.time_step, self.gamma, state.self_state.v_pref)
                max_action = ActionXY(*self.action_space_ndarray[np.argmax(action_values)])
            else:
                self.action_values = list()
                max_min_value = float('-inf')
                max_action = None
                for action in self.action_space:
                    next_self_state = self.propagate(state.self_state, action)
                    ob, reward = self.env.onestep_lookahead(action, self.time_step)
                    batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device) for next_human_state in ob], dim=0)
                    # VALUE UPDATE
                    outputs = self.model(self.rotate(batch_next_states))
                    min_output, min_index = torch.min(outputs, 0)
                    min_value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * min_output.data.item()
                    self.action_values.append(min_value)
                    if min_value > max_min_value:
                        max_min_value = min_value
                        max_action = action
        if self.phase == 'train': self.last_state = self.transform(state)
        return max_action

    def transform(self, state):
        """
        Take the state passed from agent and transform it to tensor for batch training

        :param state:
        :return: tensor of shape (len(state), )
        """
        assert len(state.human_states) == 1
        state = torch.Tensor(state.self_state + state.human_states[0]).to(self.device)
        state = self.rotate(state.unsqueeze(0)).squeeze(dim=0)
        return state

    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)

        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        if self.kinematics == 'unicycle':
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            # set theta to be zero since it's not used
            theta = torch.zeros_like(v_pref)
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
        return new_state
