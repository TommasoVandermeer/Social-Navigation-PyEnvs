import numpy as np
import os
import sys
import logging
import pickle
from social_gym.social_nav_sim import SocialNavSim

### GLOBAL VARIABLES TO BE SET TO RUN THE TEST # [5,7,14,21,28,35]
N_HUMANS = np.array([5,7,14,21,28,35], dtype=int) # With 42 and 49 humans and a radius of 7 meters the code is unable to find a random initial configuration and remains stucked
CIRCLE_RADIUS = 7
TRIALS = 100
TIME_PER_EPISODE = 50
HEADLESS = True
FULLY_COOPERATIVE = True # If true, robot is visible by humans
TIME_STEP = 0.01 # Time step for humans update
ROBOT_TIME_STEP = 0.25 # Time step for robot update
SEED_OFFSET = 1000
ROBOT_RADIUS = 0.3
RUNGE_KUTTA = False
SINGLE_TEST = False # If false, multiple test with different robot and human policies are executed in one time
SAVE_STATES = True # If true, agents (humans and robot) states are saved in an output file
## SINGLE TEST VARIABLES
HUMAN_POLICY = "orca"
ROBOT_POLICY = "bp"
ROBOT_MODEL_DIR = "robot_models/cadrl_on_orca" # Used only if testing a trainable policy
## MULTIPLE TESTS VARIABLES
# ROBOT_POLICIES_TO_BE_TESTED = ["ssp", "bp", "cadrl", "cadrl", "cadrl", "sarl", "sarl", "sarl", "lstm_rl", "lstm_rl", "lstm_rl"]
# ROBOT_MODEL_DIRS_TO_BE_TESTED = ["-", "-", "robot_models/cadrl_on_orca", "robot_models/cadrl_on_sfm_guo", "robot_models/cadrl_on_hsfm_new_guo",
#                                  "robot_models/sarl_on_orca", "robot_models/sarl_on_sfm_guo", "robot_models/sarl_on_hsfm_new_guo",
#                                  "robot_models/lstm_rl_on_orca", "robot_models/lstm_rl_on_sfm_guo", "robot_models/lstm_rl_on_hsfm_new_guo"]
# HUMAN_POLICIES_TO_BE_TESTED = ["orca", "sfm_guo", "hsfm_new_guo"]
ROBOT_POLICIES_TO_BE_TESTED = ["cadrl", "sarl", "sarl", "sarl", "lstm_rl", "lstm_rl", "lstm_rl"]
ROBOT_MODEL_DIRS_TO_BE_TESTED = ["robot_models/cadrl_on_hsfm_new_guo",
                                 "robot_models/sarl_on_orca", "robot_models/sarl_on_sfm_guo", "robot_models/sarl_on_hsfm_new_guo",
                                 "robot_models/lstm_rl_on_orca", "robot_models/lstm_rl_on_sfm_guo", "robot_models/lstm_rl_on_hsfm_new_guo"]
HUMAN_POLICIES_TO_BE_TESTED = ["orca", "sfm_guo", "hsfm_new_guo"]
OUTPUT_FILE_NAME = "multiple_tests.log"
### VARIABLES USED FOR IMPLEMENTATION PURPOSES, DO NOT CHANGE THESE
TRAINABLE_POLICIES = ["cadrl", "sarl", "lstm_rl"]
ROBOT_POLICIES = ["sfm_roboticsupo","sfm_helbing","sfm_guo","sfm_moussaid","hsfm_farina","hsfm_guo",
                 "hsfm_moussaid","hsfm_new","hsfm_new_guo","hsfm_new_moussaid","orca","ssp","bp","cadrl",
                 "sarl","lstm_rl"]
HUMAN_POLICIES = ["sfm_roboticsupo","sfm_helbing","sfm_guo","sfm_moussaid","hsfm_farina","hsfm_guo",
                 "hsfm_moussaid","hsfm_new","hsfm_new_guo","hsfm_new_moussaid","orca"]
LOGGING_BASE_DIR = "tests"
RESULTS_BASE_DIR = "results"
if ROBOT_POLICY in TRAINABLE_POLICIES: LOGGING_FILE_NAME = ROBOT_MODEL_DIR[13:] + "_on_" + HUMAN_POLICY + ".log"
else: LOGGING_FILE_NAME = ROBOT_POLICY + "_on_" + HUMAN_POLICY + ".log"

def single_human_robot_policy_test(human_policy:str, robot_policy:str, robot_model_dir=None):
    if robot_policy in TRAINABLE_POLICIES: robot_policy_title = robot_model_dir[13:]
    else: robot_policy_title = robot_policy
    all_tests = {}
    for i, n_agents in enumerate(N_HUMANS):
        test_time_to_goal = []
        test_success = []
        test_collisions = []
        test_truncated = []
        if SAVE_STATES:
            # TODO: Save human radiuses (also on multiple tests)
            test_specifics = {"humans": n_agents, "circle_radius": CIRCLE_RADIUS, "trials": TRIALS,
                                "max_episode_time": TIME_PER_EPISODE, "fully_cooperative": FULLY_COOPERATIVE,
                                "time_step": TIME_STEP, "robot_time_step": ROBOT_TIME_STEP, "seed_offset": SEED_OFFSET, "robot_radius": ROBOT_RADIUS,
                                "runge_kutta": RUNGE_KUTTA, "human_policy": human_policy, "robot_policy": robot_policy,
                                "robot_model_dir": robot_model_dir}
            test_results = {"specifics": test_specifics, "results": []}
        for trial in range(TRIALS):
            if trial % 20 == 0: logging.info(f"Start trial {trial+1} w/ {N_HUMANS[i]} humans")
            np.random.seed(SEED_OFFSET + trial)
            simulator = SocialNavSim([CIRCLE_RADIUS,n_agents,True,human_policy,HEADLESS,RUNGE_KUTTA,True,False,FULLY_COOPERATIVE], "circular_crossing")
            if trial == 0 and SAVE_STATES: test_specifics["humans_radiuses"] = [human.radius for human in simulator.humans] # Save human radiuses
            simulator.set_time_step(TIME_STEP)
            simulator.set_robot_time_step(ROBOT_TIME_STEP)
            robot_policy_index = ROBOT_POLICIES.index(robot_policy)
            if robot_policy_index < 10: simulator.set_robot_policy(policy_name=robot_policy, crowdnav_policy=False, runge_kutta=True)
            elif robot_policy_index == 10: simulator.set_robot_policy(policy_name=robot_policy, crowdnav_policy=False)
            elif robot_policy_index >= 11 and robot_policy_index < 13: simulator.set_robot_policy(policy_name=robot_policy, crowdnav_policy=True)
            else: simulator.set_robot_policy(policy_name=robot_policy, crowdnav_policy=True, model_dir=os.path.join(os.path.dirname(__file__),robot_model_dir), il=False)
            simulator.robot.set_radius_and_update_graphics(ROBOT_RADIUS)
            ## RUN FOR MAX TIME PER EPISODE
            human_states, robot_states, trial_collisions, trial_time_to_goal, trial_success, trial_truncated = simulator.run_k_steps(int(TIME_PER_EPISODE/TIME_STEP), additional_info=True, stop_when_collision_or_goal=True, save_states_time_step=ROBOT_TIME_STEP)
            ## SAVE STATES
            if SAVE_STATES: 
                trial_results = {"trial": trial, "human_states": human_states, "robot_states": robot_states, 
                                 "collision": trial_collisions, "success": trial_success, "truncated": trial_truncated,
                                 "time_to_goal": trial_time_to_goal}
                test_results["results"].append(trial_results)
            ## SAVE METRICS
            test_time_to_goal.append(trial_time_to_goal)
            test_success.append(trial_success)
            test_collisions.append(trial_collisions)
            test_truncated.append(trial_truncated)
        logging.info(f"ROBOT POLICY: {robot_policy_title} - HUMAN POLICY: {human_policy} - HUMANS: {n_agents}")
        logging.info(f"Success rate for test w/ {N_HUMANS[i]} humans: {sum(test_success)/TRIALS}")
        if len(tuple(filter(None, test_time_to_goal))) == 0: logging.info(f"Average time to goal for test w/ {N_HUMANS[i]} humans: None")
        else: logging.info(f"Average time to goal for test w/ {N_HUMANS[i]} humans: {sum(filter(None, test_time_to_goal))/len(tuple(filter(None, test_time_to_goal)))}")
        logging.info(f"Collisions for test w/ {N_HUMANS[i]} humans: {sum(test_collisions)}")
        logging.info(f"Truncated episodes for test w/ {N_HUMANS[i]} humans: {sum(test_truncated)}")
        if SAVE_STATES: all_tests[f"{n_agents}_humans"] = test_results
    return all_tests

### SINGLE TEST
if SINGLE_TEST:
    # Configue Logging
    base_dir = os.path.join(os.path.dirname(__file__),LOGGING_BASE_DIR)
    results_dir = os.path.join(base_dir,RESULTS_BASE_DIR)
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    if os.path.exists(os.path.join(base_dir,LOGGING_FILE_NAME)): 
        key = input('Output directory already exists! Overwrite it? (y/n)')
        if key == 'y': pass
        else: sys.exit()
    file_handler = logging.FileHandler(os.path.join(base_dir,LOGGING_FILE_NAME), mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    if HUMAN_POLICY not in HUMAN_POLICIES: raise(NotImplementedError)
    all_tests = single_human_robot_policy_test(HUMAN_POLICY, ROBOT_POLICY, ROBOT_MODEL_DIR)
    # Save test results in output file
    if SAVE_STATES:
        if ROBOT_POLICY in TRAINABLE_POLICIES: robot_policy_title = ROBOT_MODEL_DIR[13:]
        else: robot_policy_title = ROBOT_POLICY
        with open(os.path.join(results_dir,f'{robot_policy_title}_on_{HUMAN_POLICY}.pkl'), "wb") as f: pickle.dump(all_tests, f); f.close()
### MULTIPLE TESTS
else:
    # Configure logging
    base_dir = os.path.join(os.path.dirname(__file__),LOGGING_BASE_DIR)
    results_dir = os.path.join(base_dir,RESULTS_BASE_DIR)
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    file_handler = logging.FileHandler(os.path.join(base_dir,OUTPUT_FILE_NAME), mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler],format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    for k, robot_policy in enumerate(ROBOT_POLICIES_TO_BE_TESTED):
        if robot_policy in TRAINABLE_POLICIES: robot_policy_title = ROBOT_MODEL_DIRS_TO_BE_TESTED[k][13:]
        else: robot_policy_title = robot_policy
        for human_policy in HUMAN_POLICIES_TO_BE_TESTED:
            # TO BE REMOVED
            if robot_policy == "cadrl" and (human_policy == "orca" or human_policy == "sfm_guo"): continue
            # TO BE REMOVED END
            if human_policy == "orca": rk45 = False
            else: rk45 = RUNGE_KUTTA
            # Initialize metrics
            if human_policy not in HUMAN_POLICIES: raise(NotImplementedError)
            # Save test results in output file
            all_tests = single_human_robot_policy_test(human_policy, robot_policy, ROBOT_MODEL_DIRS_TO_BE_TESTED[k])
            if SAVE_STATES:
                with open(os.path.join(results_dir,f'{robot_policy_title}_on_{human_policy}.pkl'), "wb") as f: pickle.dump(all_tests, f); f.close()