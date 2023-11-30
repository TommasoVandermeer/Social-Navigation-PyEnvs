import numpy as np
from social_gym.policy.forces import compute_desired_force, compute_social_force_guo as compute_social_force, compute_torque_force
from social_gym.policy.policy import Policy
from social_gym.src.action import ActionXYW

class HSFMNewGuo(Policy):
    def __init__(self):
        """
        The Headed Social Force Model defined by Farina with a new modification affecting the torqie force and
        a modification proposed by Guo.
        """
        super().__init__()
        self.name = 'hsfm_new_guo'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic3'
        self.params = {'relaxation_time': 0.5,
                       'Ai': 2000.0,
                       'Aw': 2000.0, # For obstacles, not used
                       'Bi': 0.08,
                       'Bw': 0.08, # For obstacles, not used
                       'Ci': 120.0,
                       'Cw': 120.0, # For obstacles, not used
                       'Di': 0.6,
                       'Dw': 0.6, # For obstacles, not used
                       'k1': 120000.0,
                       'k2': 240000.0,
                       'ko': 1.0,
                       'kd': 500.0,
                       'alpha': 3.0,
                       'k_lambda': 0.3,
                       'mass': 80}

    def configure(self, config):
        return
    
    def set_phase(self, phase):
        return
    
    def predict(self, state):
        """
        Predict new velocity on the basis of the current state. The HSFM Forces are computed 
        and then integrated with Euler to find the new velocity.
        """
        # Goals should be updated outside this loop beacuse the goal position is embedded in the state passed as an argument
        # Obstacles are not taken into account
        # Group forces are not taken into account
        self_state = state.self_state
        other_states = state.human_states
        inertia =  0.5 * self.params['mass'] * self_state.radius * self_state.radius
        ## Compute forces
        _, desired_force = compute_desired_force(self.params, self_state)
        social_force = compute_social_force(self.params, self_state, other_states)
        driving_force = desired_force + social_force
        torque_force = compute_torque_force(self.params, self_state, inertia, driving_force)
        rotational_matrix = np.array([[np.cos(self_state.theta), -np.sin(self_state.theta)],[np.sin(self_state.theta), np.cos(self_state.theta)]], dtype=np.float64)
        global_force = np.empty((2,), dtype=np.float64)
        global_force[0] = np.dot(desired_force + social_force, rotational_matrix[:,0])
        global_force[1] = self.params['ko'] * np.dot(social_force, rotational_matrix[:,1]) - self.params['kd'] * self_state.vy
        ## Compute action
        new_body_velocity = np.array([self_state.vx, self_state.vy], dtype=np.float64) + (global_force / self.params['mass']) * self.time_step
        if (np.linalg.norm(new_body_velocity) > self_state.v_pref): new_body_velocity = (new_body_velocity / np.linalg.norm(new_body_velocity)) * self_state.v_pref
        new_angular_velocity = self_state.w + ((torque_force / inertia) * self.time_step)
        action = ActionXYW(new_body_velocity[0],new_body_velocity[1],new_angular_velocity)
        ## Saving last state
        self.last_state = state
        return action