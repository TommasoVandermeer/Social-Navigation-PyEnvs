from social_gym.src.utils import bound_angle
from social_gym.src.human_agent import HumanAgent
from social_gym.src.robot_agent import RobotAgent
from social_gym.src.action import ActionXY, ActionRot
from scipy.integrate import solve_ivp
import numpy as np

GOAL_RADIUS = 0.35
N_GENERAL_STATES = 8
N_HEADED_STATES = 6
N_NOT_HEADED_STATES = 4

class Group:
    def __init__(self):
        self.group_agents = []
        self.center = np.array([0.0,0.0],dtype=np.float64)

    def append_agent(self, agent:int):
        self.group_agents.append(agent)

    def compute_center(self):
        self.center /= len(self.group_agents)

    def num_agents(self):
        return len(self.group_agents)

class MotionModelManager:
    def __init__(self, motion_model_title:str, consider_robot:bool, runge_kutta:bool, humans:list[HumanAgent], robot:RobotAgent, walls:list):
        self.consider_robot = consider_robot
        self.runge_kutta = runge_kutta
        self.humans = humans
        self.robot = robot
        self.walls = walls
        self.set_motion_model(motion_model_title)

    def set_motion_model(self, motion_model_title:str):
        self.motion_model_title = motion_model_title
        global compute_desired_force, compute_obstacle_force, compute_social_force, compute_group_force, compute_torque_force
        if self.motion_model_title == "sfm_helbing": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_helbing as compute_social_force, compute_group_force_dummy as compute_group_force; self.headed = False; self.include_mass = True
        elif self.motion_model_title == "sfm_guo": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_guo as compute_obstacle_force, compute_social_force_guo as compute_social_force, compute_group_force_dummy as compute_group_force; self.headed = False; self.include_mass = True
        elif self.motion_model_title == "sfm_moussaid": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_moussaid as compute_social_force, compute_group_force_dummy as compute_group_force; self.headed = False; self.include_mass = True
        elif self.motion_model_title == "sfm_roboticsupo": from social_gym.src.forces import compute_desired_force_roboticsupo as compute_desired_force, compute_obstacle_force_roboticsupo as compute_obstacle_force, compute_social_force_roboticsupo as compute_social_force, compute_group_force_roboticsupo as compute_group_force; self.headed = False; self.include_mass = False
        elif self.motion_model_title == "hsfm_farina": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_helbing as compute_social_force, compute_torque_force_farina as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True
        elif self.motion_model_title == "hsfm_guo": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_guo as compute_obstacle_force, compute_social_force_guo as compute_social_force, compute_torque_force_farina as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True
        elif self.motion_model_title == "hsfm_moussaid": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_moussaid as compute_social_force, compute_torque_force_farina as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True
        elif self.motion_model_title == "hsfm_new": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_helbing as compute_social_force, compute_torque_force_new as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True
        elif self.motion_model_title == "hsfm_new_guo": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_guo as compute_obstacle_force, compute_social_force_guo as compute_social_force, compute_torque_force_new as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True
        elif self.motion_model_title == "hsfm_new_moussaid": from social_gym.src.forces import compute_desired_force, compute_obstacle_force_helbing as compute_obstacle_force, compute_social_force_moussaid as compute_social_force, compute_torque_force_new as compute_torque_force, compute_group_force_dummy as compute_group_force; self.headed = True; self.include_mass = True
        else: raise Exception(f"The human motion model '{self.motion_model_title}' does not exist")
        for human in self.humans: human.set_parameters(motion_model_title)

    def get_human_states(self, include_goal=True, headed=False):
        if include_goal:
            if headed:
                # State: [x, y, yaw, BVx, BVy, Omega, Gx, Gy] - Pose (x,y,yaw), Body velocity (body_linear_x,body_linear_y,angular), and Goal (goal_x, goal_y)
                state = np.empty([len(self.humans),N_GENERAL_STATES],dtype=np.float64)
                for i in range(len(self.humans)):
                    human_state = np.array([self.humans[i].position[0],self.humans[i].position[1],self.humans[i].yaw,self.humans[i].body_velocity[0],self.humans[i].body_velocity[1],self.humans[i].angular_velocity,self.humans[i].goals[0][0],self.humans[i].goals[0][1]], dtype=np.float64)
                    state[i] = human_state
            else:
                # State: [x, y, yaw, Vx, Vy, Omega, Gx, Gy] - Pose (x,y,yaw), Velocity (linear_x,linear_y,angular), and Goal (goal_x, goal_y)
                state = np.empty([len(self.humans),N_GENERAL_STATES],dtype=np.float64)
                for i in range(len(self.humans)):
                    human_state = np.array([self.humans[i].position[0],self.humans[i].position[1],self.humans[i].yaw,self.humans[i].linear_velocity[0],self.humans[i].linear_velocity[1],self.humans[i].angular_velocity,self.humans[i].goals[0][0],self.humans[i].goals[0][1]], dtype=np.float64)
                    state[i] = human_state
        else:
            if headed:
                # State: [x, y, yaw, BVx, BVy, Omega] - Pose (x,y,yaw) and Velocity (body_linear_x,body_linear_y,angular)
                state = np.empty([len(self.humans),N_HEADED_STATES],dtype=np.float64)
                for i in range(len(self.humans)):
                    human_state = np.array([self.humans[i].position[0],self.humans[i].position[1],self.humans[i].yaw,self.humans[i].body_velocity[0],self.humans[i].body_velocity[1],self.humans[i].angular_velocity], dtype=np.float64)
                    state[i] = human_state
            else:
                # State: [x, y, Vx, Vy] - Position (x,y) and Velocity (linear_x,linear_y)
                state = np.empty([len(self.humans),N_NOT_HEADED_STATES],dtype=np.float64)
                for i in range(len(self.humans)):
                    human_state = np.array([self.humans[i].position[0],self.humans[i].position[1],self.humans[i].linear_velocity[0],self.humans[i].linear_velocity[1]], dtype=np.float64)
                    state[i] = human_state
        return state

    def set_human_states(self, state:np.array, just_visual=False):
        if not just_visual:
            # State must be of the form: [x, y, yaw, Vx, Vy, Omega, Gx, Gy] or [x, y, yaw, BVx, BVy, Omega, Gx, Gy]
            for i in range(len(self.humans)):
                self.humans[i].position[0] = state[i,0]
                self.humans[i].position[1] = state[i,1]
                self.humans[i].yaw = state[i,2]
                if self.headed:
                    self.humans[i].body_velocity[0] = state[i,3]
                    self.humans[i].body_velocity[1] = state[i,4]
                else:
                    self.humans[i].linear_velocity[0] = state[i,3]
                    self.humans[i].linear_velocity[1] = state[i,4]
                self.humans[i].angular_velocity = state[i,5]
                self.rewind_goals(self.humans[i], [state[i,6],state[i,7]])
        else:
            # We only care about position and yaw [x, y, yaw], state can be of any form
            for i in range(len(self.humans)):
                self.humans[i].position[0] = state[i,0]
                self.humans[i].position[1] = state[i,1]
                self.humans[i].yaw = state[i,2]
                self.humans[i].update()

    def rewind_goals(self, agent:HumanAgent, goal:list):
        if ((agent.goals) and (agent.goals[0] != goal)):
            while agent.goals[0] != goal:
                goal_back = agent.goals[0]
                agent.goals.remove(goal_back)
                agent.goals.append(goal_back)
    
    def update_goals(self, agent:HumanAgent):
        if ((agent.goals) and (np.linalg.norm(agent.goals[0] - agent.position) < GOAL_RADIUS)):
            goal = agent.goals[0]
            agent.goals.remove(goal)
            agent.goals.append(goal)

    def bound_velocity(self, velocity:np.array, desired_speed:float):
        if (np.linalg.norm(velocity) > desired_speed): velocity = (velocity / np.linalg.norm(velocity)) * desired_speed
        return velocity

    def update(self, t:float, dt:float):
        if not self.runge_kutta: self.update_positions_euler(dt)
        else: self.update_positions_rk45(t,dt)

    def compute_rotational_matrix(self, agent:HumanAgent):
        agent.rotational_matrix = np.array([[np.cos(agent.yaw), -np.sin(agent.yaw)],[np.sin(agent.yaw), np.cos(agent.yaw)]], dtype=np.float64)

    def compute_forces(self):
        groups = {}
        for i in range(len(self.humans)):
            # Update goals
            self.update_goals(self.humans[i])
            # Update obstacles
            self.humans[i].obstacles.clear()
            for wall in self.walls:
                obstacle, distance = wall.get_closest_point(self.humans[i].position)
                self.humans[i].obstacles.append(obstacle)
            # Update linear velocity and rotation matrix for Headed models
            if self.headed:
                self.compute_rotational_matrix(self.humans[i])
                self.humans[i].linear_velocity = np.matmul(self.humans[i].rotational_matrix, self.humans[i].body_velocity)
            # Update groups
            if (self.humans[i].group_id <0): continue
            if (not self.humans[i].group_id in groups): groups[self.humans[i].group_id] = Group()
            groups[self.humans[i].group_id].append_agent(i)
            groups[self.humans[i].group_id].center += self.humans[i].position
        for key in groups:
            groups[key].compute_center()
        for i in range(len(self.humans)):
            desired_direction = compute_desired_force(self.humans[i])
            compute_obstacle_force(self.humans[i])
            compute_social_force(i, self.humans, self.robot, self.consider_robot)
            if not self.headed:
                compute_group_force(i, self.humans, desired_direction, groups)
                self.humans[i].global_force = self.humans[i].desired_force + self.humans[i].obstacle_force + self.humans[i].social_force + self.humans[i].group_force
            else:
                compute_group_force(i, self.humans, desired_direction, groups)
                compute_torque_force(self.humans[i])
                self.humans[i].global_force[0] = np.dot(self.humans[i].desired_force + self.humans[i].obstacle_force + self.humans[i].social_force, self.humans[i].rotational_matrix[:,0]) + self.humans[i].group_force[0]
                self.humans[i].global_force[1] = self.humans[i].ko * np.dot(self.humans[i].obstacle_force + self.humans[i].social_force, self.humans[i].rotational_matrix[:,1]) - self.humans[i].kd * self.humans[i].body_velocity[1] + self.humans[i].group_force[1]
        
    def update_positions_euler(self, dt:float):
        self.compute_forces()
        if not self.headed:
            for agent in self.humans:
                init_yaw = agent.yaw
                if self.include_mass: agent.linear_velocity += (agent.global_force / agent.mass) * dt
                else: agent.linear_velocity += agent.global_force * dt
                agent.linear_velocity = self.bound_velocity(agent.linear_velocity, agent.desired_speed)
                agent.yaw = bound_angle(np.arctan2(agent.linear_velocity[1], agent.linear_velocity[0]))
                agent.position += agent.linear_velocity * dt
                agent.angular_velocity = (agent.yaw - init_yaw) / dt
        else:
            for agent in self.humans:
                agent.body_velocity += (agent.global_force / agent.mass) * dt
                agent.angular_velocity += (agent.torque_force / agent.inertia) * dt
                agent.body_velocity = self.bound_velocity(agent.body_velocity, agent.desired_speed)
                agent.position += np.matmul(agent.rotational_matrix, agent.body_velocity) * dt
                agent.yaw = bound_angle(agent.yaw + agent.angular_velocity * dt)

    def update_positions_rk45(self, t:float, dt:float):
        if self.headed: 
            current_state = np.reshape(self.get_human_states(include_goal=False, headed=True), (len(self.humans) * N_HEADED_STATES,))
            solution = solve_ivp(self.f_rk45_headed, (t, t+dt), current_state, method='RK45')
            for i in range(len(self.humans)):
                self.humans[i].position[0] = solution.y[i*N_HEADED_STATES][-1]
                self.humans[i].position[1] = solution.y[i*N_HEADED_STATES+1][-1]
                self.humans[i].yaw = bound_angle(solution.y[i*N_HEADED_STATES+2][-1])
                self.humans[i].body_velocity[0] = solution.y[i*N_HEADED_STATES+3][-1]
                self.humans[i].body_velocity[1] = solution.y[i*N_HEADED_STATES+4][-1]
                self.humans[i].body_velocity = self.bound_velocity(self.humans[i].body_velocity, self.humans[i].desired_speed)
                self.humans[i].angular_velocity = solution.y[i*N_HEADED_STATES+5][-1]
                self.humans[i].linear_velocity = np.matmul(self.humans[i].rotational_matrix, self.humans[i].body_velocity)
        else: 
            current_state = np.reshape(self.get_human_states(include_goal=False, headed=False), (len(self.humans) * N_NOT_HEADED_STATES,))
            solution = solve_ivp(self.f_rk45_not_headed, (t, t+dt), current_state, method='RK45')
            for i in range(len(self.humans)):
                self.humans[i].position[0] = solution.y[i*N_NOT_HEADED_STATES][-1]
                self.humans[i].position[1] = solution.y[i*N_NOT_HEADED_STATES+1][-1]
                self.humans[i].linear_velocity[0] = solution.y[i*N_NOT_HEADED_STATES+2][-1]
                self.humans[i].linear_velocity[1] = solution.y[i*N_NOT_HEADED_STATES+3][-1]
                self.humans[i].linear_velocity = self.bound_velocity(self.humans[i].linear_velocity, self.humans[i].desired_speed)

    def complete_rk45_simulation(self, t:float, dt:float, final_time:float):
        evaluation_times = np.arange(t,final_time,dt, dtype=np.float64)
        if self.headed: 
            current_state = np.reshape(self.get_human_states(include_goal=False, headed=True), (len(self.humans) * N_HEADED_STATES,))
            solution = solve_ivp(self.f_rk45_headed, (t, t+final_time), current_state, method='RK45', t_eval=evaluation_times)
            human_states = np.empty((len(evaluation_times),len(self.humans),N_HEADED_STATES), dtype=np.float64)
            for i in range(len(solution.y[0])):
                for j in range(len(self.humans)):
                    human_states[i,j,0] = solution.y[j*N_HEADED_STATES][i]
                    human_states[i,j,1] = solution.y[j*N_HEADED_STATES+1][i]
                    human_states[i,j,2] = solution.y[j*N_HEADED_STATES+2][i]
                    human_states[i,j,3] = solution.y[j*N_HEADED_STATES+3][i]
                    human_states[i,j,4] = solution.y[j*N_HEADED_STATES+4][i]
                    human_states[i,j,5] = solution.y[j*N_HEADED_STATES+5][i]
        else:
            current_state = np.reshape(self.get_human_states(include_goal=False, headed=False), (len(self.humans) * N_NOT_HEADED_STATES,))
            solution = solve_ivp(self.f_rk45_not_headed, (t, t+final_time), current_state, method='RK45', t_eval=evaluation_times)
            human_states = np.empty((len(evaluation_times),len(self.humans),N_NOT_HEADED_STATES), dtype=np.float64)
            for i in range(len(solution.y[0])):
                for j in range(len(self.humans)):
                    human_states[i,j,0] = solution.y[j*N_NOT_HEADED_STATES][i]
                    human_states[i,j,1] = solution.y[j*N_NOT_HEADED_STATES+1][i]
                    human_states[i,j,2] = solution.y[j*N_NOT_HEADED_STATES+2][i]
                    human_states[i,j,3] = solution.y[j*N_NOT_HEADED_STATES+3][i]
        return human_states

    def f_rk45_headed(self, t, y):
        for i in range(len(self.humans)):
            self.humans[i].position[0] = y[i*N_HEADED_STATES]
            self.humans[i].position[1] = y[i*N_HEADED_STATES+1]
            self.humans[i].yaw = bound_angle(y[i*N_HEADED_STATES+2])
            self.humans[i].body_velocity[0] = y[i*N_HEADED_STATES+3]
            self.humans[i].body_velocity[1] = y[i*N_HEADED_STATES+4]
            self.humans[i].body_velocity = self.bound_velocity(self.humans[i].body_velocity, self.humans[i].desired_speed)
            self.humans[i].angular_velocity = y[i*N_HEADED_STATES+5]
        self.compute_forces()
        ydot = np.empty((len(self.humans) * N_HEADED_STATES,), dtype=np.float64)
        for i in range(len(self.humans)):
            ydot[i*N_HEADED_STATES] = np.dot(self.humans[i].rotational_matrix[0,:], self.humans[i].body_velocity)
            ydot[i*N_HEADED_STATES+1] = np.dot(self.humans[i].rotational_matrix[1,:], self.humans[i].body_velocity)
            ydot[i*N_HEADED_STATES+2] = self.humans[i].angular_velocity
            ydot[i*N_HEADED_STATES+3] = self.humans[i].global_force[0] / self.humans[i].mass
            ydot[i*N_HEADED_STATES+4] = self.humans[i].global_force[1] / self.humans[i].mass
            ydot[i*N_HEADED_STATES+5] = self.humans[i].torque_force / self.humans[i].inertia
        return ydot
    
    def f_rk45_not_headed(self, t, y):
        for i in range(len(self.humans)):
            self.humans[i].position[0] = y[i*N_NOT_HEADED_STATES]
            self.humans[i].position[1] = y[i*N_NOT_HEADED_STATES+1]
            self.humans[i].linear_velocity[0] = y[i*N_NOT_HEADED_STATES+2]
            self.humans[i].linear_velocity[1] = y[i*N_NOT_HEADED_STATES+3]
            self.humans[i].linear_velocity = self.bound_velocity(self.humans[i].linear_velocity, self.humans[i].desired_speed)
        self.compute_forces()
        ydot = np.empty((len(self.humans) * N_NOT_HEADED_STATES,), dtype=np.float64)
        if self.include_mass:
            for i in range(len(self.humans)):
                ydot[i*N_NOT_HEADED_STATES] = self.humans[i].linear_velocity[0]
                ydot[i*N_NOT_HEADED_STATES+1] = self.humans[i].linear_velocity[1]
                ydot[i*N_NOT_HEADED_STATES+2] = self.humans[i].global_force[0] / self.humans[i].mass
                ydot[i*N_NOT_HEADED_STATES+3] = self.humans[i].global_force[1] / self.humans[i].mass
        else:
            for i in range(len(self.humans)):
                ydot[i*N_NOT_HEADED_STATES] = self.humans[i].linear_velocity[0]
                ydot[i*N_NOT_HEADED_STATES+1] = self.humans[i].linear_velocity[1]
                ydot[i*N_NOT_HEADED_STATES+2] = self.humans[i].global_force[0]
                ydot[i*N_NOT_HEADED_STATES+3] = self.humans[i].global_force[1]
        return ydot
    
    def predict_actions(self, dt:float):
        self.compute_forces()
        actions = []
        if not self.headed:
            for agent in self.humans:
                linear_velocity = agent.linear_velocity.copy()
                if self.include_mass: linear_velocity += (agent.global_force / agent.mass) * dt
                else: linear_velocity += agent.global_force * dt
                linear_velocity = self.bound_velocity(linear_velocity, agent.desired_speed)
                actions.append(ActionXY(linear_velocity[0], linear_velocity[1]))
        else:
            raise NotImplementedError
        return actions