class FullState(object):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta
        self.headed = False

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta]])

class ObservableState(object):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])

class JointState(object):
    def __init__(self, self_state, human_states):
        # assert isinstance(self_state, (FullState, FullStateHeaded))
        assert isinstance(self_state, FullState)
        for human_state in human_states: assert isinstance(human_state, ObservableState)
        self.self_state = self_state
        self.human_states = human_states

class FullStateHeaded(FullState): # Added for HSFM
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta, w):
        super().__init__(px, py, vx, vy, radius, gx, gy, v_pref, theta) # velocity refers to BODY_VELOCITY
        self.w = w # Angular velocity
        self.headed = True

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta, self.w)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta, self.w]])
