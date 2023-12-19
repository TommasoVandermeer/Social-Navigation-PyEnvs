import numpy as np
from crowd_nav.policy_no_train.forces import compute_desired_force, compute_social_force_guo as compute_social_force
from crowd_nav.policy_no_train.sfm_helbing import SFMHelbing

class SFMGuo(SFMHelbing):
    def __init__(self):
        """
        The Social Force Model defined by Helbing with a modification proposed by Guo.    
        """
        super().__init__()
        self.name = 'sfm_guo'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
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
                       'mass': 80}
    
    def configure(self, config):
        return
    
    def set_phase(self, phase):
        return

    def compute_forces(self, state, other_states):
        _, desired_force = compute_desired_force(self.params, state)
        social_force = compute_social_force(self.params, state, other_states)
        global_force = desired_force + social_force
        return global_force