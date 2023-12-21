import numpy as np
from crowd_nav.policy_no_train.policy import Policy
from crowd_nav.utils.action import ActionXY

DISTANCE_THRESHOLD = 0.2

class SimpleSocialPlanner(Policy):
    def __init__(self):
        super().__init__()
        self.name = 'ssp'
        self.trainable = False
        self.kinematics = 'holonomic'
        self.multiagent_training = True

    def configure(self, config):
        assert True

    def predict(self, state):
        self_state = state.self_state
        self_position = np.array([self_state.px, self_state.py], dtype=np.float64)
        human_states = state.human_states
        over_threshold = False
        for human_state in human_states:
            human_position = np.array([human_state.px, human_state.py], dtype=np.float64)
            distance = np.linalg.norm(human_position - self_position) - human_state.radius - self_state.radius
            if distance <= DISTANCE_THRESHOLD: over_threshold = True; break
        if over_threshold: action = ActionXY(0,0)
        else:
            theta = np.arctan2(self_state.gy-self_state.py, self_state.gx-self_state.px)
            vx = np.cos(theta) * self_state.v_pref
            vy = np.sin(theta) * self_state.v_pref
            action = ActionXY(vx, vy)
        return action