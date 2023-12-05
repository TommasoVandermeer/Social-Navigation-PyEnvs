import numpy as np
from social_gym.policy.forces import compute_desired_force, compute_social_force_guo as compute_social_force, compute_torque_force
from social_gym.policy.hsfm_farina import HSFMFarina
from social_gym.src.action import ActionXYW

class HSFMNewGuo(HSFMFarina):
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

    def compute_forces(self, state, other_states):
        _, desired_force = compute_desired_force(self.params, state)
        social_force = compute_social_force(self.params, state, other_states)
        driving_force = desired_force + social_force
        torque_force = compute_torque_force(self.params, state, self.inertia, driving_force)
        rotational_matrix = np.array([[np.cos(state.theta), -np.sin(state.theta)],[np.sin(state.theta), np.cos(state.theta)]], dtype=np.float64)
        global_force = np.empty((2,), dtype=np.float64)
        global_force[0] = np.dot(desired_force + social_force, rotational_matrix[:,0])
        global_force[1] = self.params['ko'] * np.dot(social_force, rotational_matrix[:,1]) - self.params['kd'] * state.vy
        return global_force, torque_force, rotational_matrix