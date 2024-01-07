import numpy as np
import os
import sys
import logging
from social_gym.social_nav_sim import SocialNavSim

### GLOBAL VARIABLES TO BE SET TO RUN THE TEST
N_HUMANS = np.array([5,7,14,21,28,35], dtype=int) # With 42 and 49 humans and a radius of 7 meters the code is unable to find a random initial configuration and remains stucked
CIRCLE_RADIUS = 7
TRIALS = 100
TIME_PER_EPISODE = 50
HEADLESS = True
FULLY_COOPERATIVE = True # If true, robot is visible by humans
TIME_STEP = 0.25
SEED_OFFSET = 1000
ROBOT_RADIUS = 0.3
ORCA_SAFETY_SPACE = 0.03
SINGLE_TEST = False # If false, multiple test with different robot and human policies are executed in one time
## SINGLE TEST VARIABLES
RUNGE_KUTTA = False
HUMAN_POLICY = "sfm_guo"
ROBOT_POLICY = "bp"
ROBOT_MODEL_DIR = "robot_models/cadrl_on_orca" # Used only if testing a trainable policy
## MULTIPLE TESTS VARIABLES
ROBOT_POLICIES_TO_BE_TESTED = ["bp", "ssp", "cadrl", "cadrl", "cadrl", "sarl", "sarl", "sarl", "lstm_rl", "lstm_rl", "lstm_rl"]
ROBOT_MODEL_DIRS_TO_BE_TESTED = ["-", "-", "robot_models/cadrl_on_orca", "robot_models/cadrl_on_sfm_guo", "robot_models/cadrl_on_hsfm_new_guo",
                                 "robot_models/sarl_on_orca", "robot_models/sarl_on_sfm_guo", "robot_models/sarl_on_hsfm_new_guo",
                                 "robot_models/lstm_rl_on_orca", "robot_models/lstm_rl_on_sfm_guo", "robot_models/lstm_rl_on_hsfm_new_guo"]
HUMAN_POLICIES_TO_BE_TESTED = ["orca", "sfm_guo", "hsfm_new_guo"]
### VARIABLES USED FOR IMPLEMENTATION PURPOSES, DO NOT CHANGE THESE
TRAINABLE_POLICIES = ["cadrl", "sarl", "lstm_rl"]
ROBOT_POLICIES = ["sfm_roboticsupo","sfm_helbing","sfm_guo","sfm_moussaid","hsfm_farina","hsfm_guo",
                 "hsfm_moussaid","hsfm_new","hsfm_new_guo","hsfm_new_moussaid","orca","ssp","bp","cadrl",
                 "sarl","lstm_rl"]
HUMAN_POLICIES = ["sfm_roboticsupo","sfm_helbing","sfm_guo","sfm_moussaid","hsfm_farina","hsfm_guo",
                 "hsfm_moussaid","hsfm_new","hsfm_new_guo","hsfm_new_moussaid","orca"]
LOGGING_BASE_DIR = "tests"
if ROBOT_POLICY in TRAINABLE_POLICIES: LOGGING_FILE_NAME = ROBOT_MODEL_DIR[13:] + "_on_" + HUMAN_POLICY + ".log"
else: LOGGING_FILE_NAME = ROBOT_POLICY + "_on_" + HUMAN_POLICY + ".log"

### SINGLE TEST
if SINGLE_TEST:
    # Configue Logging
    base_dir = os.path.join(os.path.dirname(__file__),LOGGING_BASE_DIR)
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    if os.path.exists(os.path.join(base_dir,LOGGING_FILE_NAME)): 
        key = input('Output directory already exists! Overwrite it? (y/n)')
        if key == 'y': pass
        else: sys.exit()
    file_handler = logging.FileHandler(os.path.join(base_dir,LOGGING_FILE_NAME), mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    if HUMAN_POLICY not in HUMAN_POLICIES: raise(NotImplementedError)
    # Test
    for i, n_agents in enumerate(N_HUMANS):
        test_time_to_goal = []
        test_success = []
        test_collisions = np.empty((TRIALS,), dtype=int)
        for trial in range(TRIALS):
            logging.info(f"Start trial {trial} w/ {N_HUMANS[i]} humans")
            np.random.seed(SEED_OFFSET + trial)
            simulator = SocialNavSim([7,n_agents,True,HUMAN_POLICY,HEADLESS,RUNGE_KUTTA,True,False,FULLY_COOPERATIVE], "circular_crossing")
            simulator.set_time_step(TIME_STEP)
            if HUMAN_POLICY == "orca": simulator.motion_model_manager.set_safety_space(ORCA_SAFETY_SPACE) # This is because otherwise collisions are always present
            robot_policy_index = ROBOT_POLICIES.index(ROBOT_POLICY)
            if robot_policy_index < 10: simulator.set_robot_policy(policy_name=ROBOT_POLICY, crowdnav_policy=False, runge_kutta=True)
            elif robot_policy_index == 10: simulator.set_robot_policy(policy_name=ROBOT_POLICY, crowdnav_policy=False)
            elif robot_policy_index >= 11 and robot_policy_index < 13: simulator.set_robot_policy(policy_name=ROBOT_POLICY, crowdnav_policy=True)
            else: simulator.set_robot_policy(policy_name=ROBOT_POLICY, crowdnav_policy=True, model_dir=os.path.join(os.path.dirname(__file__),ROBOT_MODEL_DIR), il=False)
            simulator.robot.set_radius_and_update_graphics(ROBOT_RADIUS)
            ## RUN FOR MAX TIME PER EPISODE
            human_states, robot_poses, trial_collisions, trial_time_to_goal, trial_success = simulator.run_k_steps(int(TIME_PER_EPISODE/TIME_STEP), additional_info=True, stop_when_collision_or_gl=True)
            ## SAVE METRICS
            test_time_to_goal.append(trial_time_to_goal)
            test_success.append(trial_success)
            test_collisions[trial] = trial_collisions
        logging.info(f"Success rate for test w/ {N_HUMANS[i]} humans: {sum(test_success)/TRIALS}")
        logging.info(f"Average time to goal for test w/ {N_HUMANS[i]} humans: {sum(filter(None, test_time_to_goal))/TRIALS}")
        logging.info(f"Collisions for test w/ {N_HUMANS[i]} humans: {np.sum(test_collisions)}")
### MULTIPLE TESTS
else:
    base_dir = os.path.join(os.path.dirname(__file__),LOGGING_BASE_DIR)
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    for k, robot_policy in enumerate(ROBOT_POLICIES_TO_BE_TESTED):
        for human_policy in HUMAN_POLICIES_TO_BE_TESTED:
            if human_policy == "orca": rk45 = False
            else: rk45 = True
            # Configure logging
            if robot_policy in TRAINABLE_POLICIES: output_file_name = ROBOT_MODEL_DIRS_TO_BE_TESTED[k][13:] + "_on_" + human_policy + ".log"
            else: output_file_name = robot_policy + "_on_" + human_policy + ".log"
            file_handler = logging.FileHandler(os.path.join(base_dir,output_file_name), mode='w')
            stdout_handler = logging.StreamHandler(sys.stdout)
            logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler],format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
            # Initialize metrics
            if human_policy not in HUMAN_POLICIES: raise(NotImplementedError)
            # Test
            for i, n_agents in enumerate(N_HUMANS):
                test_time_to_goal = []
                test_success = []
                test_collisions = np.empty((TRIALS,), dtype=int)
                for trial in range(TRIALS):
                    logging.info(f"Start trial {trial} w/ {N_HUMANS[i]} humans")
                    np.random.seed(SEED_OFFSET + trial)
                    simulator = SocialNavSim([7,n_agents,True,human_policy,HEADLESS,rk45,True,False,FULLY_COOPERATIVE], "circular_crossing")
                    simulator.set_time_step(TIME_STEP)
                    if human_policy == "orca": simulator.motion_model_manager.set_safety_space(ORCA_SAFETY_SPACE) # This is because otherwise collisions are always present
                    robot_policy_index = ROBOT_POLICIES.index(robot_policy)
                    if robot_policy_index < 10: simulator.set_robot_policy(policy_name=robot_policy, crowdnav_policy=False, runge_kutta=True)
                    elif robot_policy_index == 10: simulator.set_robot_policy(policy_name=robot_policy, crowdnav_policy=False)
                    elif robot_policy_index >= 11 and robot_policy_index < 13: simulator.set_robot_policy(policy_name=robot_policy, crowdnav_policy=True)
                    else: simulator.set_robot_policy(policy_name=robot_policy, crowdnav_policy=True, model_dir=os.path.join(os.path.dirname(__file__),ROBOT_MODEL_DIRS_TO_BE_TESTED[k]), il=False)
                    simulator.robot.set_radius_and_update_graphics(ROBOT_RADIUS)
                    ## RUN FOR MAX TIME PER EPISODE
                    human_states, robot_poses, trial_collisions, trial_time_to_goal, trial_success = simulator.run_k_steps(int(TIME_PER_EPISODE/TIME_STEP), additional_info=True, stop_when_collision_or_goal=True)
                    ## SAVE METRICS
                    test_time_to_goal.append(trial_time_to_goal)
                    test_success.append(trial_success)
                    test_collisions[trial] = trial_collisions
                logging.info(f"Success rate for test w/ {N_HUMANS[i]} humans: {sum(test_success)/TRIALS}")
                logging.info(f"Average time to goal for test w/ {N_HUMANS[i]} humans: {sum(filter(None, test_time_to_goal))/TRIALS}")
                logging.info(f"Collisions for test w/ {N_HUMANS[i]} humans: {np.sum(test_collisions)}")