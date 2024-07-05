import os
import math
import numpy as np
import time
from social_gym.social_nav_sim import SocialNavSim
import matplotlib.pyplot as plt

HUMANS_AND_ROBOT_EQUIVALENCE_TEST_PARALLELIZE_ROBOT = True
HUMANS_AND_ROBOT_EQUIVALENCE_TEST_PARALLELIZE_HUMANS = True
HUMANS_AND_ROBOT_EQUIVALENCE_TEST_PARALLELIZE_ROBOT_AND_HUMANS = True
ONLY_HUMANS_EQUIVALENCE_TEST_PARALLELIZE_HUMANS = True

QUERY_ENV = True
SAFETY_SPACE = False

### When robot is in the simulation and queries the env in onestep lookahead and humans are HSFM-driven, the parallel and unparallel versions are not equivalent

SIMULATION_SECONDS = 10
HUMANS_POLICY = "hsfm_new_guo"
N_HUMANS = 5
RANDOM_SEED = 0
TIME_STEP = 1/100
ROBOT_TIMESTEP = 1/4
ROBOT_POLICY = "cadrl" # "sarl", "cadrl", or "lstm_rl"
ROBOT_MODEL_DIR = "robot_models/trained_on_hybrid_scenario/cadrl_on_hsfm_new_guo"

STATES_DEFS = {0:"px", 1:"py", 2:"theta", 3:"vx", 4:"vy", 5:"omega", 6:"gx", 7:"gx"}

def assert_equivalence(parallel_human_states:np.ndarray, parallel_robot_states:np.ndarray, unparallel_human_states:np.ndarray, unparallel_robot_states:np.ndarray):
    parallel_human_states = np.round(parallel_human_states, decimals=7)
    unparallel_human_states = np.round(unparallel_human_states, decimals=7)
    humans_equal = np.array_equal(parallel_human_states, unparallel_human_states)
    print("ALL HUMAN STATES ARE EQUAL:", humans_equal)
    if not humans_equal:
        not_equal_index = np.where(parallel_human_states != unparallel_human_states)[0][0]
        check_next_state = True
        while check_next_state:
            if not_equal_index >= len(parallel_human_states): print("No more states to check\n"); break
            print("STATE INDEX (not equal)", not_equal_index)
            not_equal_human_index = np.where(parallel_human_states[not_equal_index] != unparallel_human_states[not_equal_index])[0][0]
            check_next_human = True
            while check_next_human:
                if not_equal_human_index >= len(parallel_human_states[not_equal_index]): print("No more humans to check\n"); break
                print("Human index:", not_equal_human_index)
                not_equal_element_indexes = np.where(parallel_human_states[not_equal_index][not_equal_human_index] != unparallel_human_states[not_equal_index][not_equal_human_index])[0]
                print("Not equal element indexes:", [value for key, value in STATES_DEFS.items() if key in not_equal_element_indexes])
                check_next_human = input("Do you want to check the next human? (y/n): ")
                if check_next_human.lower() == "y": not_equal_human_index += 1
                elif check_next_human.lower() == "n": check_next_human = False
                else: break
            check_next_state = input("Do you want to check the next state? (y/n): ")
            if check_next_state.lower() == "y": not_equal_index += 1
            elif check_next_state.lower() == "n": check_next_state = False
            else:break
    if parallel_robot_states is not None and unparallel_robot_states is not None:
        parallel_robot_states = np.round(parallel_robot_states, decimals=7)
        unparallel_robot_states = np.round(unparallel_robot_states, decimals=7)
        robots_equal = np.array_equal(parallel_robot_states, unparallel_robot_states)
        print("ALL ROBOT STATES ARE EQUAL:", robots_equal)
        if not robots_equal:
            not_equal_index = np.where(parallel_robot_states != unparallel_robot_states)[0][0]
            check_next_state = True
            while check_next_state:
                if not_equal_index >= len(parallel_robot_states): print("No more states to check\n"); break
                print("STATE INDEX (not equal)", not_equal_index)
                not_equal_element_indexes = np.where(parallel_robot_states[not_equal_index] != unparallel_robot_states[not_equal_index])[0]
                print("Not equal element indexes:", [value for key, value in STATES_DEFS.items() if key in not_equal_element_indexes])
                check_next_state = input("Do you want to check the next state? (y/n): ")
                if check_next_state.lower() == "y": not_equal_index += 1
                elif check_next_state.lower() == "n": check_next_state = False
                else:break

def define_simulators(parallel_robot:bool, parallel_humans:bool, insert_robot=True):
    ### Parallel simulator
    np.random.seed(RANDOM_SEED)
    parallel_social_nav = SocialNavSim(config_data = {"insert_robot": insert_robot, "human_policy": HUMANS_POLICY, "headless": True,
                                         "runge_kutta": False, "robot_visible": insert_robot, "robot_radius": 0.3,
                                         "circle_radius": 7, "n_actors": N_HUMANS, "randomize_human_positions": True, "randomize_human_attributes": False},
                          scenario="circular_crossing", parallelize_robot = parallel_robot, parallelize_humans = parallel_humans)    
    parallel_social_nav.set_time_step(TIME_STEP)
    parallel_social_nav.set_robot_time_step(ROBOT_TIMESTEP)
    parallel_social_nav.set_robot_policy(policy_name=ROBOT_POLICY, crowdnav_policy=True, model_dir=os.path.join(os.path.dirname(__file__),ROBOT_MODEL_DIR), il=False)
    parallel_social_nav.robot.policy.query_env = QUERY_ENV
    if SAFETY_SPACE: parallel_social_nav.motion_model_manager.set_safety_space(0.15)
    ### Unparallel simulator
    np.random.seed(RANDOM_SEED)
    unparallel_social_nav = SocialNavSim(config_data = {"insert_robot": insert_robot, "human_policy": HUMANS_POLICY, "headless": True,
                                         "runge_kutta": False, "robot_visible": insert_robot, "robot_radius": 0.3,
                                         "circle_radius": 7, "n_actors": N_HUMANS, "randomize_human_positions": True, "randomize_human_attributes": False},
                          scenario="circular_crossing", parallelize_robot = False, parallelize_humans = False)
    unparallel_social_nav.set_time_step(TIME_STEP)
    unparallel_social_nav.set_robot_time_step(ROBOT_TIMESTEP)
    unparallel_social_nav.set_robot_policy(policy_name=ROBOT_POLICY, crowdnav_policy=True, model_dir=os.path.join(os.path.dirname(__file__),ROBOT_MODEL_DIR), il=False)
    unparallel_social_nav.robot.policy.query_env = QUERY_ENV
    if SAFETY_SPACE: unparallel_social_nav.motion_model_manager.set_safety_space(0.15)
    return parallel_social_nav, unparallel_social_nav

def run_simulators(parallel_social_nav:SocialNavSim, unparallel_social_nav:SocialNavSim, insert_robot=True):
    if insert_robot:
        parallel_human_states, parallel_robot_states = parallel_social_nav.run_k_steps(int(SIMULATION_SECONDS/TIME_STEP), save_states_time_step=TIME_STEP)
        unparallel_human_states, unparallel_robot_states = unparallel_social_nav.run_k_steps(int(SIMULATION_SECONDS/TIME_STEP), save_states_time_step=TIME_STEP)
        return parallel_human_states, parallel_robot_states, unparallel_human_states, unparallel_robot_states
    else:
        parallel_human_states = parallel_social_nav.run_k_steps(int(SIMULATION_SECONDS/TIME_STEP), save_states_time_step=TIME_STEP)
        unparallel_human_states  = unparallel_social_nav.run_k_steps(int(SIMULATION_SECONDS/TIME_STEP), save_states_time_step=TIME_STEP)
        return parallel_human_states, unparallel_human_states

if HUMANS_AND_ROBOT_EQUIVALENCE_TEST_PARALLELIZE_ROBOT:
    ### Define simulators
    parallel_social_nav, unparallel_social_nav = define_simulators(parallel_robot=True, parallel_humans=False)
    ### Run simulators
    parallel_human_states, parallel_robot_states, unparallel_human_states, unparallel_robot_states = run_simulators(parallel_social_nav, unparallel_social_nav)
    ### Assert equivalence
    print(f"\nEQUIVALENCE TEST PARALLELIZE ROBOT - QUERY ENV: {QUERY_ENV} - SAFETY SPACE: {SAFETY_SPACE}")
    assert_equivalence(parallel_human_states, parallel_robot_states, unparallel_human_states, unparallel_robot_states)

if HUMANS_AND_ROBOT_EQUIVALENCE_TEST_PARALLELIZE_HUMANS:
    ### Define simulators
    parallel_social_nav, unparallel_social_nav = define_simulators(parallel_robot=False, parallel_humans=True)
    ### Run simulators
    parallel_human_states, parallel_robot_states, unparallel_human_states, unparallel_robot_states = run_simulators(parallel_social_nav, unparallel_social_nav)
    ### Assert equivalence
    print(f"\nEQUIVALENCE TEST PARALLELIZE HUMANS - QUERY ENV: {QUERY_ENV} - SAFETY SPACE: {SAFETY_SPACE}")
    assert_equivalence(parallel_human_states, parallel_robot_states, unparallel_human_states, unparallel_robot_states)

if HUMANS_AND_ROBOT_EQUIVALENCE_TEST_PARALLELIZE_ROBOT_AND_HUMANS:
    ### Define simulators
    parallel_social_nav, unparallel_social_nav = define_simulators(parallel_robot=True, parallel_humans=True)
    ### Run simulators
    parallel_human_states, parallel_robot_states, unparallel_human_states, unparallel_robot_states = run_simulators(parallel_social_nav, unparallel_social_nav)
    ### Assert equivalence
    print(f"\nEQUIVALENCE TEST PARALLELIZE ROBOT AND HUMANS - QUERY ENV: {QUERY_ENV} - SAFETY SPACE: {SAFETY_SPACE}")
    assert_equivalence(parallel_human_states, parallel_robot_states, unparallel_human_states, unparallel_robot_states)

if ONLY_HUMANS_EQUIVALENCE_TEST_PARALLELIZE_HUMANS:
    ### Define simulators
    parallel_social_nav, unparallel_social_nav = define_simulators(parallel_robot=False, parallel_humans=True, insert_robot=False)
    ### Run simulators
    parallel_human_states, unparallel_human_states = run_simulators(parallel_social_nav, unparallel_social_nav, insert_robot=False)
    ### Assert equivalence
    print(f"\nEQUIVALENCE TEST ONLY HUMANS PARALLELIZE HUMANS - QUERY ENV: {QUERY_ENV} - SAFETY SPACE: {SAFETY_SPACE}")
    assert_equivalence(parallel_human_states, None, unparallel_human_states, None)