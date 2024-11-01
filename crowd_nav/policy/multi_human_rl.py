import torch
import numpy as np
from crowd_nav.utils.action import ActionRot, ActionXY
from crowd_nav.policy.cadrl import CADRL, compute_rotated_states_and_reward, compute_action_value, propagate_humans_state_with_constant_velocity_model

from social_gym.src.utils import PRECISION

class MultiHumanRL(CADRL):
    def __init__(self):
        super().__init__()

    def predict(self, state):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        occupancy_maps = None
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            if self.parallelize:
                r = state.self_state
                h = state.human_states
                current_robot_state = np.copy(np.array([r.px,r.py,r.vx,r.vy,r.radius,r.gx,r.gy,r.v_pref,r.theta], PRECISION))
                if self.with_theta_and_omega_visible: current_humans_state = np.copy(np.array([[hi.px,hi.py,hi.vx,hi.vy,hi.radius,hi.theta,hi.omega] for hi in h], PRECISION))
                else: current_humans_state = np.copy(np.array([[hi.px,hi.py,hi.vx,hi.vy,hi.radius] for hi in h], PRECISION))
                ## Compute next human state querying env (not assuming constant velocity)
                if self.query_env: 
                    if self.with_theta_and_omega_visible: next_humans_state = self.env.motion_model_manager.get_next_human_observable_states(self.time_step, theta_and_omega_visible=True)[:,:6]
                    else: next_humans_state = self.env.motion_model_manager.get_next_human_observable_states(self.time_step)
                ## Compute next human state assuming constant velocity
                else: next_humans_state = propagate_humans_state_with_constant_velocity_model(current_humans_state, self.time_step, theta_and_omega_visible=self.with_theta_and_omega_visible)
                ## Compute Value Network input and rewards
                rotated_states, rewards = compute_rotated_states_and_reward(self.action_space_ndarray, next_humans_state, current_humans_state, current_robot_state, self.time_step, theta_and_omega_visible=self.with_theta_and_omega_visible)
                # In LSTM-RL humans have to be sorted by decreasing distance to robot
                if self.name == "LSTM-RL": 
                    indices = np.flip(np.argsort(rotated_states[:,:,11], axis=1), axis=1)
                    rotated_states = rotated_states[np.arange(rotated_states.shape[0])[:, None], indices, :]
                ## Compute Value Network output - BOTTLENECK
                value_network_outputs = np.zeros((len(rewards),), PRECISION) 
                for ii in range(len(rewards)):
                    batch_next_states = torch.Tensor(rotated_states[ii]).to(self.device).reshape((len(rotated_states[ii]),13+2*int(self.with_theta_and_omega_visible))).unsqueeze(0)
                    # batch_next_states = torch.cat([torch.Tensor([rotated_state]).to(self.device) for rotated_state in rotated_states[ii]], dim=0)
                    ## Compute occupancy map
                    if self.with_om:
                        if occupancy_maps is None: occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                        batch_next_states = torch.cat([batch_next_states, occupancy_maps.to(self.device)], dim=2)
                    value_network_outputs[ii] = self.model(batch_next_states).data.item()
                ## Compute action value 
                action_values = compute_action_value(rewards, value_network_outputs, self.time_step, self.gamma, state.self_state.v_pref)
                max_action = ActionXY(*self.action_space_ndarray[np.argmax(action_values)])
            else:
                self.action_values = list()
                max_value = float('-inf')
                max_action = None
                for action in self.action_space:
                    next_self_state = self.propagate(state.self_state, action)
                    next_human_states, reward = self.env.onestep_lookahead(action, self.time_step)
                    # In LSTM-RL humans have to be sorted by decreasing distance to robot
                    if self.name == "LSTM-RL": next_human_states = sorted(next_human_states, key=lambda x: np.linalg.norm((next_self_state.px - x.px, next_self_state.py - x.py)), reverse=True)
                    batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device) for next_human_state in next_human_states], dim=0)
                    rotated_batch_input = self.rotate(batch_next_states, theta_and_omega_visible=self.with_theta_and_omega_visible).unsqueeze(0)
                    if self.with_om:
                        if occupancy_maps is None:
                            occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                        rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps.to(self.device)], dim=2)
                    # VALUE UPDATE
                    next_state_value = self.model(rotated_batch_input).data.item()
                    value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
                    self.action_values.append(value)
                    if value > max_value:
                        max_value = value
                        max_action = action
                if max_action is None: raise ValueError('Value network is not well trained. ')
        if self.phase == 'train': self.last_state = self.transform(state)
        return max_action

    def compute_reward(self, nav, humans):
        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(humans):
            dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            if dist < 0:
                collision = True
                break
            if dist < dmin:
                dmin = dist

        # check if reaching the goal
        reaching_goal = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy)) < nav.radius
        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1
        elif dmin < 0.2:
            reward = (dmin - 0.2) * 0.5 * self.time_step
        else:
            reward = 0

        return reward

    def transform(self, state):
        """
        Take the state passed from agent and transform it to the input of value network

        :param state:
        :return: tensor of shape (# of humans, len(state))
        """
        state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.device) for human_state in state.human_states], dim=0)
        if self.with_om:
            occupancy_maps = self.build_occupancy_maps(state.human_states)
            state_tensor = torch.cat([self.rotate(state_tensor, theta_and_omega_visible=self.with_theta_and_omega_visible), occupancy_maps.to(self.device)], dim=1)
        else:
            state_tensor = self.rotate(state_tensor, theta_and_omega_visible=self.with_theta_and_omega_visible)
        return state_tensor

    def input_dim(self):
        return self.joint_state_dim + (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)

    def build_occupancy_maps(self, human_states):
        """

        :param human_states:
        :return: tensor of shape (# human - 1, self.cell_num ** 2)
        """
        occupancy_maps = []
        for human in human_states:
            other_humans = np.concatenate([np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                                         for other_human in human_states if other_human != human], axis=0)
            other_px = other_humans[:, 0] - human.px
            other_py = other_humans[:, 1] - human.py
            # new x-axis is in the direction of human's velocity
            human_velocity_angle = np.arctan2(human.vy, human.vx)
            other_human_orientation = np.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            # compute indices of humans in the grid
            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                # calculate relative velocity for other agents
                other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed
                dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num ** 2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[3 * int(index)].append(1)
                            dm[3 * int(index) + 1].append(other_vx[i])
                            dm[3 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for i, cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()

