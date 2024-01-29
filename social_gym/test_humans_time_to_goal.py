import numpy as np 
import pickle
import os
import sys
import logging
from social_gym.social_nav_sim import SocialNavSim

# This file can be used to test how much time does humans require to reach the goal
# when the robot moves with different policies. In orderd to run this test you should run "test_robot_policy.py"
# first so that there we can extract how many humans and in how much time do they reach the goal. Then, here, the same
# trials are tested without the robot for the same duration of the tests with the robot in order to compute the time 
# humans require to reach the goal (just for the first time). All tests are done within the circular crossing setting.

### GLOBAL VARIABLES TO BE SET TO RUN THE TEST
RESULTS_FILES = ["bp_on_orca.pkl","bp_on_sfm_guo.pkl","bp_on_hsfm_new_guo.pkl","ssp_on_orca.pkl","ssp_on_sfm_guo.pkl","ssp_on_hsfm_new_guo.pkl",
                 "cadrl_on_orca_on_orca.pkl","cadrl_on_orca_on_sfm_guo.pkl","cadrl_on_orca_on_hsfm_new_guo.pkl",
                 "cadrl_on_sfm_guo_on_orca.pkl","cadrl_on_sfm_guo_on_sfm_guo.pkl","cadrl_on_sfm_guo_on_hsfm_new_guo.pkl",
                 "cadrl_on_hsfm_new_guo_on_orca.pkl","cadrl_on_hsfm_new_guo_on_sfm_guo.pkl","cadrl_on_hsfm_new_guo_on_hsfm_new_guo.pkl",
                 "sarl_on_orca_on_orca.pkl","sarl_on_orca_on_sfm_guo.pkl","sarl_on_orca_on_hsfm_new_guo.pkl",
                 "sarl_on_sfm_guo_on_orca.pkl","sarl_on_sfm_guo_on_sfm_guo.pkl","sarl_on_sfm_guo_on_hsfm_new_guo.pkl",
                 "sarl_on_hsfm_new_guo_on_orca.pkl","sarl_on_hsfm_new_guo_on_sfm_guo.pkl","sarl_on_hsfm_new_guo_on_hsfm_new_guo.pkl",
                 "lstm_rl_on_orca_on_orca.pkl","lstm_rl_on_orca_on_sfm_guo.pkl","lstm_rl_on_orca_on_hsfm_new_guo.pkl",
                 "lstm_rl_on_sfm_guo_on_orca.pkl","lstm_rl_on_sfm_guo_on_sfm_guo.pkl","lstm_rl_on_sfm_guo_on_hsfm_new_guo.pkl",
                 "lstm_rl_on_hsfm_new_guo_on_orca.pkl","lstm_rl_on_hsfm_new_guo_on_sfm_guo.pkl","lstm_rl_on_hsfm_new_guo_on_hsfm_new_guo.pkl"]
OUTPUT_FILE_WITH_ROBOT = "human_times_with_robot.pkl"
HUMAN_STATES_FILE = "human_states_tests_without_robot.pkl"
OUTPUT_FILE_WITHOUT_ROBOT = "human_times_without_robot.pkl"
SKIP_PHASE_1 = False # If true, data of experiments with robot will be extracted from the OUTPUT_FILE_WITH_ROBOT specified, otherwise data will be computed from passed RESULTS_FILES
SKIP_PHASE_2 = False # If true, data of experiments without robot will be extracted from the HUMAN_STATES_FILE specified, otherwise tests without humans will be executed
SKIP_PHASE_3 = True # If true, humans' time to goal is not extracted from experiments without robot
HEADLESS = True
### IMPLEMENTATION VARIABLES (do not change)
ENVIRONMENTS = ["orca", "sfm_guo", "hsfm_new_guo"]
TESTS = ["5_humans","7_humans","14_humans","21_humans","28_humans","35_humans"]

metrics_dir = os.path.join(os.path.dirname(__file__),'tests','metrics')
if not os.path.exists(metrics_dir): os.makedirs(metrics_dir)
### PHASE 1: Extract humans time to goal from tests with robot
if not SKIP_PHASE_1:
    extracted_data = {}
    for i, results in enumerate(RESULTS_FILES):
        results_file_extracted_data = {}
        with open(os.path.join(os.path.dirname(__file__),'tests','results',results), "rb") as f: test_data = pickle.load(f)
        for k, test in enumerate(test_data):
            print(" ************************ ")
            print("")
            test_extracted_data = {}
            n_humans = int(test.replace("_humans",""))
            human_time_to_goal = np.empty((test_data[test]["specifics"]["trials"],n_humans), dtype=np.float64); human_time_to_goal[:] = np.NaN
            n_humans_reached_goal = np.zeros((test_data[test]["specifics"]["trials"],), dtype=int)
            episode_times = np.zeros((test_data[test]["specifics"]["trials"],), dtype=np.float64)
            for e, episode in enumerate(test_data[test]["results"]):
                human_goals = np.empty((n_humans,2), dtype=np.float64)
                human_reached_first_goal = [False for z in range(n_humans)]
                for t, human_states in enumerate(episode["human_states"]):
                    for h, human_state in enumerate(human_states):
                        # Take first human state position to compute humans' goal
                        if t == 0: human_goals[h] = -human_state[0:2]
                        else:
                            if human_reached_first_goal[h]: continue
                            human_position = human_state[0:2]
                            human_radius = test_data[test]["specifics"]["humans_radiuses"][h]
                            distance_to_goal = np.linalg.norm(human_position - human_goals[h])
                            robot_timestep = test_data[test]["specifics"]["robot_time_step"]
                            # If human has reached the goal (due to the fact that we immediately change goal when humans reach it)
                            if (distance_to_goal <= human_radius + (robot_timestep / 2)) and not human_reached_first_goal[h]: # We assume humans desired speed is 1 m/s (should be human_radius + (robot_timestep * human_des_speed / 2))
                                human_reached_first_goal[h] = True
                                human_time_to_goal[e,h] = t * robot_timestep
                    if t == len(episode["human_states"]) - 1: 
                        episode_times[e] = t * robot_timestep
                        n_humans_reached_goal[e] = sum(human_reached_first_goal)
            # Save final data
            test_extracted_data["times_to_goal"] = human_time_to_goal
            test_extracted_data["n_humans_reached_goal"] = n_humans_reached_goal
            test_extracted_data["episode_times"] = episode_times
            test_extracted_data["specifics"] = test_data[test]["specifics"]
            results_file_extracted_data[test] = test_extracted_data
            # Print some info
            print(f"TEST {results} - {test}")
            print(f"N° humans that reached the goal in {test_data[test]['specifics']['trials']} trials: ", np.sum(n_humans_reached_goal, axis = 0))
            print("Average humans time to goal: ", np.nansum(human_time_to_goal, axis=(0,1)) / np.sum(n_humans_reached_goal, axis = 0))
            print(f"Average episode duration: {np.sum(episode_times, axis = 0) / test_data[test]['specifics']['trials']}")
            print("")
        extracted_data[results] = results_file_extracted_data
    # Save extracted data in an output file
    with open(os.path.join(metrics_dir,OUTPUT_FILE_WITH_ROBOT), "wb") as f: pickle.dump(extracted_data, f); f.close()

### PHASE 2: Simulate episodes without robot
# Load data with robot
if SKIP_PHASE_1: 
    with open(os.path.join(metrics_dir,OUTPUT_FILE_WITH_ROBOT), "rb") as f: data_with_robot = pickle.load(f)
else: data_with_robot = extracted_data.copy()
if not SKIP_PHASE_2:
    # Setup logging
    base_dir = os.path.join(os.path.dirname(__file__),"tests")
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    file_handler = logging.FileHandler(os.path.join(base_dir,"humans_times_tests.log"), mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler], format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    # Simulate trials for the same duration as the ones with robot
    test_data = data_with_robot[RESULTS_FILES[0]] # Read one test data to gather info about simulation setting, we assume tests are all the same except for the human policy
    extracted_data_without_robot = {}
    for i, env in enumerate(ENVIRONMENTS):
        env_data_without_robot = {}
        for k, test in enumerate(TESTS):
            logging.info(f"Environment {env} with {test}")
            n_trials = test_data[test]["specifics"]["trials"]
            n_humans = int(test.replace("_humans",""))
            seed_offset = test_data[test]["specifics"]["seed_offset"]
            circle_radius = test_data[test]["specifics"]["circle_radius"]
            runge_kutta = test_data[test]["specifics"]["runge_kutta"]
            time_step = test_data[test]["specifics"]["time_step"]
            max_time = test_data[test]["specifics"]["max_episode_time"]
            episode_steps = max_time / time_step
            test_human_states = []
            for trial in range(n_trials):
                if trial % 20 == 0: logging.info(f"Start trial {trial+1}")
                np.random.seed(seed_offset + trial)
                simulator = SocialNavSim([circle_radius,n_humans,True,env,HEADLESS,runge_kutta,False,False,False], "circular_crossing")
                simulator.set_time_step(time_step)
                # Run episode of predefined duration
                human_states = simulator.run_k_steps(int(episode_steps), save_states_time_step=time_step)
                # Save data
                test_human_states.append(human_states)
            env_data_without_robot[test] = test_human_states
        extracted_data_without_robot[env] = env_data_without_robot
    # Save extracted data in an output file
    with open(os.path.join(metrics_dir,HUMAN_STATES_FILE), "wb") as f: pickle.dump(extracted_data_without_robot, f); f.close()

### PHASE 3: From experiments of humans without robot extract humans' time to goal
# Load data without robot
if SKIP_PHASE_2:
    with open(os.path.join(metrics_dir,HUMAN_STATES_FILE), "rb") as f: data_without_robot = pickle.load(f)
else: data_without_robot = extracted_data_without_robot.copy()
if not SKIP_PHASE_3: pass

