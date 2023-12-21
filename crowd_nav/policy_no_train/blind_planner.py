import numpy as np
from crowd_nav.policy_no_train.policy import Policy
from crowd_nav.utils.action import ActionXY

class BlindPlanner(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'bp'
        self.trainable = False
        self.kinematics = 'holonomic'
        self.multiagent_training = True

    def configure(self, config):
        assert True

    def predict(self, state):
        self_state = state.self_state
        theta = np.arctan2(self_state.gy-self_state.py, self_state.gx-self_state.px)
        vx = np.cos(theta) * self_state.v_pref
        vy = np.sin(theta) * self_state.v_pref
        action = ActionXY(vx, vy)

        return action