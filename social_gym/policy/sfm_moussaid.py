import numpy as np
from social_gym.policy.forces import compute_desired_force, compute_social_force_moussaid as compute_social_force
from social_gym.policy.policy import Policy
from social_gym.src.action import ActionXY

class SFMMoussaid(Policy):
    def __init__(self):
        """
        The Social Force Model defined by Helbing with a modification proposed by Moussaid.    
        """
        super().__init__()
        self.name = 'sfm_moussaid'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
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
                       'mass': 80}
    
    def configure(self, config):
        return
    
    def set_phase(self, phase):
        return

    def predict(self, state):
        """
        Predict new velocity on the basis of the current state. The SFM Forces are computed 
        and then integrated with Euler to find the new velocity.
        """
        # Goals should be updated outside this loop beacuse the goal position is embedded in the state passed as an argument
        # Obstacles are not taken into account
        # Group forces are not taken into account
        self_state = state.self_state
        other_states = state.human_states
        ## Compute forces
        _, desired_force = compute_desired_force(self.params, self_state)
        social_force = compute_social_force(self.params, self_state, other_states)
        global_force = desired_force + social_force
        ## Compute action
        new_velocity = np.array([self_state.vx, self_state.vy], dtype=np.float64) + (global_force / self.params['mass']) * self.time_step
        if (np.linalg.norm(new_velocity) > self_state.v_pref): new_velocity = (new_velocity / np.linalg.norm(new_velocity)) * self_state.v_pref
        action = ActionXY(new_velocity[0],new_velocity[1])
        ## Saving last state
        self.last_state = state
        return action