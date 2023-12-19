import numpy as np
from crowd_nav.policy_no_train.forces import compute_desired_force, compute_social_force_helbing as compute_social_force, compute_torque_force
from crowd_nav.policy_no_train.policy import Policy
from crowd_nav.utils.action import ActionXYW, NewHeadedState
from scipy.integrate import solve_ivp
from crowd_nav.utils.state import FullStateHeaded

RUNGE_KUTTA_STEP_TIME_LIMIT = 0.05

class HSFMFarina(Policy):
    def __init__(self):
        """
        The Headed Social Force Model defined by Farina.    
        """
        super().__init__()
        self.name = 'hsfm_farina'
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic3'
        self.params = {'relaxation_time': 0.5,
                       'Ai': 2000.0,
                       'Aw': 2000.0, # For obstacles, not used
                       'Bi': 0.08,
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
        self.current_state = state.self_state # Save current state for RK45 integration
        self.current_other_states = state.human_states # Save current other state for RK45 integration
        self.inertia =  0.5 * self.params['mass'] * self_state.radius * self_state.radius
        if self.time_step <= RUNGE_KUTTA_STEP_TIME_LIMIT: ## EULER
            ## Compute forces
            global_force, torque_force, _ = self.compute_forces(self_state, other_states)
            ## Compute action
            new_body_velocity = np.array([self_state.vx, self_state.vy], dtype=np.float64) + (global_force / self.params['mass']) * self.time_step
            if (np.linalg.norm(new_body_velocity) > self_state.v_pref): new_body_velocity = (new_body_velocity / np.linalg.norm(new_body_velocity)) * self_state.v_pref
            new_angular_velocity = self_state.w + ((torque_force / self.inertia) * self.time_step)
            action = ActionXYW(new_body_velocity[0],new_body_velocity[1],new_angular_velocity)
        else: ## RUNGE-KUTTA-45
            current_y = np.array([self_state.px, self_state.py, self_state.theta, self_state.vx, self_state.vy, self_state.w], dtype=np.float64)
            solution = solve_ivp(self.f_rk45, (0, self.time_step), current_y, method='RK45')
            action = NewHeadedState(solution.y[0][-1],solution.y[1][-1],solution.y[2][-1],solution.y[3][-1],solution.y[4][-1],solution.y[5][-1])
        ## Saving last state
        self.last_state = state
        return action
    
    def compute_forces(self, state, other_states):
        _, desired_force = compute_desired_force(self.params, state)
        social_force = compute_social_force(self.params, state, other_states)
        torque_force = compute_torque_force(self.params, state, self.inertia, desired_force)
        rotational_matrix = np.array([[np.cos(state.theta), -np.sin(state.theta)],[np.sin(state.theta), np.cos(state.theta)]], dtype=np.float64)
        global_force = np.empty((2,), dtype=np.float64)
        global_force[0] = np.dot(desired_force + social_force, rotational_matrix[:,0])
        global_force[1] = self.params['ko'] * np.dot(social_force, rotational_matrix[:,1]) - self.params['kd'] * state.vy
        return global_force, torque_force, rotational_matrix

    def f_rk45(self, t, y):
        # y: [px, py, theta, bvx, bvy, w]
        # state: [px, py, bvx, bvy, radius, gx, gy, v_pref, theta, w]
        self_state = FullStateHeaded(y[0],y[1],y[3],y[4],self.current_state.radius,self.current_state.gx,self.current_state.gy,self.current_state.v_pref,y[2],y[5])
        global_force, torque_force, rotational_matrix = self.compute_forces(self_state, self.current_other_states)
        ydot = np.empty((6,), dtype=np.float64)
        ydot[0] = np.dot(rotational_matrix[0,:], np.array([self_state.vx,self_state.vy], dtype=np.float64))
        ydot[1] = np.dot(rotational_matrix[1,:], np.array([self_state.vx,self_state.vy], dtype=np.float64))
        ydot[2] = self_state.w
        ydot[3] = global_force[0] / self.params['mass']
        ydot[4] = global_force[1] / self.params['mass']
        ydot[5] = torque_force / self.inertia
        return ydot