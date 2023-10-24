import math
import numpy as np
from src.human_agent import HumanAgent
from src.robot_agent import RobotAgent
from src.utils import bound_angle
from scipy.integrate import solve_ivp

GOAL_RADIUS = 0.3

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

def compute_desired_force(agent:HumanAgent):
    distance = np.linalg.norm(agent.goals[0] - agent.position)
    if ((agent.goals) and (distance > GOAL_RADIUS)):
        desired_direction = (agent.goals[0] - agent.position) / distance
        agent.desired_force = agent.goal_weight * (desired_direction * agent.desired_speed - agent.linear_velocity) / agent.relaxation_time
    else: desired_direction = np.array([0.0,0.0], dtype=np.float64)
    return desired_direction

def compute_obstacle_force(agent:HumanAgent):
    agent.obstacle_force = np.array([0.0,0.0], dtype=np.float64)
    for obstacle in agent.obstacles:
        min_diff = np.array(agent.position) - obstacle
        distance = np.linalg.norm(min_diff) - agent.radius
        agent.obstacle_force += agent.obstacle_weight * math.exp(-distance / agent.obstacle_sigma) * (min_diff / np.linalg.norm(min_diff))
    if (agent.obstacles): agent.obstacle_force /= len(agent.obstacles)

def compute_social_force(index:int, agents:list[HumanAgent], robot:RobotAgent):
    target_agent = agents[index]
    target_agent.social_force = np.array([0.0,0.0], dtype=np.float64)
    entities = agents.copy()
    entities.append(robot)
    for i in range(len(entities)):
        if (i == index): i += 1; continue
        diff = np.array(entities[i].position) - np.array(target_agent.position)
        diff_direction = diff / np.linalg.norm(diff)
        vel_diff = target_agent.linear_velocity - entities[i].linear_velocity
        interaction_vector = target_agent.agent_lambda * vel_diff + diff_direction
        interaction_length = np.linalg.norm(interaction_vector)
        interaction_direction = interaction_vector / interaction_length
        theta = bound_angle(np.arctan2(diff_direction[1], diff_direction[0]) - np.arctan2(interaction_direction[1], interaction_direction[0]))
        B = target_agent.agent_gamma * interaction_length
        force_velocity_amount = -math.exp(-np.linalg.norm(diff) / B - (target_agent.agent_nPrime * B * theta) ** 2)
        force_angle_amount = np.sign(-theta) * math.exp(-np.linalg.norm(diff) / B - (target_agent.agent_n * B * theta) ** 2)
        force_velocity = force_velocity_amount * interaction_direction
        force_angle = force_angle_amount * np.array([-interaction_direction[1], interaction_direction[0]])
        target_agent.social_force += target_agent.social_weight * (force_velocity + force_angle)

def compute_social_force_no_robot(index:int, agents:list[HumanAgent]):
    target_agent = agents[index]
    target_agent.social_force = np.array([0.0,0.0], dtype=np.float64)
    entities = agents.copy()
    for i in range(len(entities)):
        if (i == index): i += 1; continue
        diff = np.array(entities[i].position) - np.array(target_agent.position)
        diff_direction = diff / np.linalg.norm(diff)
        vel_diff = target_agent.linear_velocity - entities[i].linear_velocity
        interaction_vector = target_agent.agent_lambda * vel_diff + diff_direction
        interaction_length = np.linalg.norm(interaction_vector)
        interaction_direction = interaction_vector / interaction_length
        theta = bound_angle(np.arctan2(diff_direction[1], diff_direction[0]) - np.arctan2(interaction_direction[1], interaction_direction[0]))
        B = target_agent.agent_gamma * interaction_length
        force_velocity_amount = -math.exp(-np.linalg.norm(diff) / B - (target_agent.agent_nPrime * B * theta) ** 2)
        force_angle_amount = np.sign(-theta) * math.exp(-np.linalg.norm(diff) / B - (target_agent.agent_n * B * theta) ** 2)
        force_velocity = force_velocity_amount * interaction_direction
        force_angle = force_angle_amount * np.array([-interaction_direction[1], interaction_direction[0]])
        target_agent.social_force += target_agent.social_weight * (force_velocity + force_angle)

def compute_group_force(index:int, agents:list[HumanAgent], desired_direction:np.array, groups:dict):
    target_agent = agents[index]
    target_agent.group_force = np.array([0.0,0.0])
    if ((not target_agent.group_id in groups) or (groups[target_agent.group_id].num_agents() < 2) or (target_agent.group_id == -1)): return
    ## Gaze force
    com = np.array(groups[target_agent.group_id].center)
    com = (1/(groups[target_agent.group_id].num_agents() -1)) * (groups[target_agent.group_id].num_agents() * com - np.array(target_agent.position))
    relative_com = com - np.array(target_agent.position)
    vision_angle = math.radians(90)
    element_product = np.dot(desired_direction, relative_com)
    com_angle = math.acos(element_product / (np.linalg.norm(desired_direction) * np.linalg.norm(relative_com)))
    if com_angle > vision_angle:
        desired_direction_squared = np.linalg.norm(desired_direction) ** 2
        desired_direction_distance = element_product / desired_direction_squared
        group_gaze_force = target_agent.group_gaze_weight * desired_direction_distance * desired_direction
    else: group_gaze_force = np.array([0.0,0.0])
    ## Coherence force
    com = np.array(groups[target_agent.group_id].center)
    relative_com = com - np.array(target_agent.position)
    distance = np.linalg.norm(relative_com)
    max_distance = (groups[target_agent.group_id].num_agents() -1) / 2
    softened_factor = target_agent.group_coh_weight * (math.tanh(distance - max_distance) + 1) / 2
    group_coherence_force = relative_com * softened_factor
    ## Repulsion force
    group_repulsion_force = np.array([0.0,0.0])
    for i in range(groups[target_agent.group_id].num_agents()):
        if index == groups[target_agent.group_id].group_agents[i]: continue
        diff = np.array(target_agent.position) - np.array(agents[i].position)
        if (np.linalg.norm(diff) < target_agent.radius + agents[i].radius): group_repulsion_force += diff
    group_repulsion_force *= target_agent.group_rep_weight
    target_agent.group_force = group_gaze_force + group_coherence_force + group_repulsion_force

def compute_forces(agents:list[HumanAgent], robot:RobotAgent):
    groups = {}
    i = 0
    for agent in agents:
        if (agent.group_id < 0): i += 1; continue
        if (not agent.group_id in groups): groups[agent.group_id] = Group()
        groups[agent.group_id].append_agent(i)
        groups[agent.group_id].center += agent.position
        i += 1
    for key in groups:
        groups[key].compute_center()
    for i in range(len(agents)):
        desired_direction = compute_desired_force(agents[i])
        compute_obstacle_force(agents[i])
        compute_social_force(i, agents, robot)
        compute_group_force(i, agents, desired_direction, groups)
        agents[i].global_force = agents[i].desired_force + agents[i].obstacle_force + agents[i].social_force + agents[i].group_force

def compute_forces_no_robot(agents:list[HumanAgent]):
    groups = {}
    i = 0
    for agent in agents:
        if (agent.group_id < 0): i += 1; continue
        if (not agent.group_id in groups): groups[agent.group_id] = Group()
        groups[agent.group_id].append_agent(i)
        groups[agent.group_id].center += agent.position
        i += 1
    for key in groups:
        groups[key].compute_center()
    for i in range(len(agents)):
        desired_direction = compute_desired_force(agents[i])
        compute_obstacle_force(agents[i])
        compute_social_force_no_robot(i, agents)
        compute_group_force(i, agents, desired_direction, groups)
        agents[i].global_force = agents[i].desired_force + agents[i].obstacle_force + agents[i].social_force + agents[i].group_force
    
def update_positions(agents:list[HumanAgent], dt:float):
    for agent in agents:
        init_yaw = agent.yaw
        agent.linear_velocity += agent.global_force * dt
        if (np.linalg.norm(agent.linear_velocity) > agent.desired_speed): agent.linear_velocity = (agent.linear_velocity / np.linalg.norm(agent.linear_velocity)) * agent.desired_speed
        agent.yaw = bound_angle(np.arctan2(agent.linear_velocity[1], agent.linear_velocity[0]))
        agent.position += agent.linear_velocity * dt
        agent.angular_velocity = (agent.yaw - init_yaw) / dt
        check_agents_collisions(agents)
        if ((agent.goals) and (np.linalg.norm(agent.goals[0] - agent.position) < GOAL_RADIUS)):
            goal = agent.goals[0]
            agent.goals.remove(goal)
            agent.goals.append(goal)

def update_positions_RK45(agents:list[HumanAgent], t:float, dt:float):
    for agent in agents:
        init_yaw = agent.yaw
        global LINEAR_VELOCITY
        global GLOBAL_FORCE
        GLOBAL_FORCE = agent.global_force
        ## First integration
        solution = solve_ivp(f1, (t, t+dt), agent.linear_velocity, method='RK45')
        agent.linear_velocity = np.array([solution.y[0][-1],solution.y[1][-1]],dtype=np.float64)
        LINEAR_VELOCITY = agent.linear_velocity
        if (np.linalg.norm(agent.linear_velocity) > agent.desired_speed): agent.linear_velocity = (agent.linear_velocity / np.linalg.norm(agent.linear_velocity)) * agent.desired_speed
        agent.yaw = bound_angle(np.arctan2(agent.linear_velocity[1], agent.linear_velocity[0]))
        ## Second integration
        solution2 = solve_ivp(f2, (t, t+dt), agent.position, method='RK45')
        agent.position = np.array([solution2.y[0][-1],solution2.y[1][-1]], dtype=np.float64)
        agent.angular_velocity = (agent.yaw - init_yaw) / dt
        if ((agent.goals) and (np.linalg.norm(agent.goals[0] - agent.position) < GOAL_RADIUS)):
            goal = agent.goals[0]
            agent.goals.remove(goal)
            agent.goals.append(goal)

def f1(t, y):
    ydot = GLOBAL_FORCE
    return ydot

def f2(t, y):
    ydot = LINEAR_VELOCITY
    return ydot

def check_agents_collisions(agents:list[HumanAgent]):
        for i in range(len(agents)):
            for j in range(len(agents)):
                if (j == i) or (j < i): continue
                if (np.linalg.norm(agents[i].position - agents[j].position) < agents[i].radius + agents[j].radius):
                    direction = (agents[i].position - agents[j].position) / np.linalg.norm(agents[i].position - agents[j].position)
                    collision_point = agents[j].position + direction * agents[j].radius
                    agents[i].position = collision_point + direction * agents[i].radius
                    agents[j].position = collision_point - direction * agents[j].radius


