import numpy as np
from crowd_nav.policy_no_train.forces import compute_desired_force, compute_social_force_moussaid as compute_social_force, compute_torque_force
from crowd_nav.policy_no_train.hsfm_farina import HSFMFarina

class HSFMMoussaid(HSFMFarina):
    def __init__(self):
        """
        The Headed Social Force Model defined by Farina with a modification proposed by Moussaid.    
        """
        super().__init__()
        self.name = 'hsfm_moussaid'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic3'
        self.params = {'relaxation_time': 0.5,
                       'Ei': 360,
                       'agent_lambda': 2.0,
                       'gamma': 0.35,
                       'ns': 2.0,
                       'ns1': 3.0,
                       'Aw': 2000.0, # For obstacles, not used
                       'Bw': 0.08, # For obstacles, not used
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
        torque_force = compute_torque_force(self.params, state, self.inertia, desired_force)
        rotational_matrix = np.array([[np.cos(state.theta), -np.sin(state.theta)],[np.sin(state.theta), np.cos(state.theta)]], dtype=np.float64)
        global_force = np.empty((2,), dtype=np.float64)
        global_force[0] = np.dot(desired_force + social_force, rotational_matrix[:,0])
        global_force[1] = self.params['ko'] * np.dot(social_force, rotational_matrix[:,1]) - self.params['kd'] * state.vy
        return global_force, torque_force, rotational_matrix