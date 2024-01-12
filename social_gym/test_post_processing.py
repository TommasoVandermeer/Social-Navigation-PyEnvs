import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## GLOBAL VARIABLES TO SET
SINGLE_PROCESSING = True # If true, a single results file is post-processed. Otherwise a list provided is post-processed
SPACE_COMPLIANCE_THRESHOLD = 0.5
EXPORT_ON_EXCEL = True # If true, resulting metrics are loaded on an Excel file
PLOT_METRICS = True # (only multi-processing) If true, some metrics are plotted to compare robot policies
## SINGLE POSTPROCESSING
RESULTS_FILE = "bp_on_orca.pkl"
## MULTIPLE POSTPROCESSING
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
## IMPLEMENTATION VARIABLES - DO NOT CHANGE
TESTS = ["5_humans","7_humans","14_humans","21_humans","28_humans","35_humans"]
METRICS = ['success_rate','collisions','truncated_eps','time_to_goal','min_speed','avg_speed',
           'max_speed','min_accel.','avg_accel.','max_accel.','min_jerk','avg_jerk','max_jerk',
           'min_dist','avg_dist','space_compliance','path_length','SPL']
BIG_FLOAT = 100000.0

metrics_dir = os.path.join(os.path.dirname(__file__),'tests','metrics')
if not os.path.exists(metrics_dir): os.makedirs(metrics_dir)

def single_results_file_post_processing(test_data:dict):
    """
    This function post-process the given results file to compute some metrics indicating the quality of
    the robot policy used for the test.

    params:
    - test_data (dict): contains test specifics and results

    output:
    - Metrics for the tests with [5,7,14,21,28,35] humans
    """
    # METRICS
    test_success_rate = np.empty((len(TESTS),), dtype=np.float64)
    test_collisions = np.empty((len(TESTS),), dtype=np.int64)
    test_truncated_eps = np.empty((len(TESTS),), dtype=np.int64)
    test_time_to_goal = np.empty((len(TESTS),), dtype=np.float64)
    test_avg_min_vel = np.empty((len(TESTS),), dtype=np.float64)
    test_avg_vel = np.empty((len(TESTS),), dtype=np.float64)
    test_max_vel = np.empty((len(TESTS),), dtype=np.float64)
    test_avg_min_accel = np.empty((len(TESTS),), dtype=np.float64)
    test_avg_accel = np.empty((len(TESTS),), dtype=np.float64)
    test_max_accel = np.empty((len(TESTS),), dtype=np.float64)
    test_avg_min_jerk = np.empty((len(TESTS),), dtype=np.float64)
    test_avg_jerk = np.empty((len(TESTS),), dtype=np.float64)
    test_max_jerk = np.empty((len(TESTS),), dtype=np.float64)
    test_avg_min_dist = np.empty((len(TESTS),), dtype=np.float64)
    test_avg_dist = np.empty((len(TESTS),), dtype=np.float64)
    test_avg_space_compl = np.empty((len(TESTS),), dtype=np.float64)
    test_avg_path_length = np.empty((len(TESTS),), dtype=np.float64)
    test_spl = np.empty((len(TESTS),), dtype=np.float64)
    # POST-PROCESSING
    for k, test in enumerate(TESTS):
        print(" ************************ ")
        print("")
        print(f"{test} TEST SPECIFICS")
        print(test_data[test]["specifics"])
        collisions = 0
        successes = 0
        truncated_epsiodes = 0
        time_to_goal = 0
        average_minimum_velocity = 0
        average_trial_velocity = 0
        average_maximum_velocity = 0
        average_minimum_acceleration = 0
        average_trial_acceleration = 0
        average_maximum_acceleration = 0
        average_minimum_jerk = 0
        average_trial_jerk = 0
        average_maximum_jerk = 0
        average_minimum_distance_to_humans = 0
        average_trial_distance_to_humans = 0
        average_trial_space_compliance = 0
        average_path_length = 0
        success_weighted_by_path_length = 0
        for episode in test_data[test]["results"]:
            collisions += int(episode["collision"])
            successes += int(episode["success"])
            truncated_epsiodes += int(episode["truncated"])
            if episode["time_to_goal"] != None: time_to_goal += episode["time_to_goal"]
            minimum_velocity = BIG_FLOAT
            average_velocity = 0
            maximum_velocity = 0
            minimum_acceleration = BIG_FLOAT
            average_acceleration = 0
            maximum_acceleration = 0
            minimum_jerk = BIG_FLOAT
            average_jerk = 0
            maximum_jerk = 0
            minimum_distance = BIG_FLOAT if not episode["collision"] else 0
            average_distance = 0
            trial_space_compliance = 0
            trial_path_length = 0
            shortest_path_length = 0
            for i, robot_state in enumerate(episode["robot_states"]):
                instant_position = robot_state[0:2]
                if i == 0: shortest_path_length = np.linalg.norm(robot_state[6:8] - instant_position) # Distance between goal and initial configuration
                if i > 0: 
                    previous_position = episode["robot_states"][i-1][0:2]
                    trial_path_length += np.linalg.norm(instant_position - previous_position)
                average_instant_distance = 0
                space_compliance = True
                for human_state in episode["human_states"][i]:
                    human_position = human_state[0:2]
                    instant_distance = np.linalg.norm(human_position - instant_position) - (test_data[test]["specifics"]["robot_radius"] * 2) # We suppose humans have the same size as robot
                    average_instant_distance += instant_distance
                    if instant_distance < minimum_distance and not episode["collision"]: minimum_distance = instant_distance
                    if instant_distance < SPACE_COMPLIANCE_THRESHOLD: space_compliance = False
                trial_space_compliance += int(space_compliance)
                average_distance += average_instant_distance / len(episode["human_states"][i])
                instant_velocity = robot_state[3:5]
                instant_velocity_norm = np.linalg.norm(instant_velocity)
                average_velocity += instant_velocity_norm
                if instant_velocity_norm < minimum_velocity: minimum_velocity = instant_velocity_norm
                if instant_velocity_norm > maximum_velocity: maximum_velocity = instant_velocity_norm
                if i < len(episode["robot_states"])-1: 
                    instant_acceleration = (episode["robot_states"][i+1][3:5] - instant_velocity) / test_data[test]["specifics"]["time_step"]
                    instant_acceleration_norm = np.linalg.norm(instant_acceleration)
                    average_acceleration += instant_acceleration_norm
                    if instant_acceleration_norm < minimum_acceleration: minimum_acceleration = instant_acceleration_norm
                    if instant_acceleration_norm > maximum_acceleration: maximum_acceleration = instant_acceleration_norm
                    if i < len(episode["robot_states"])-2: 
                        next_acceleration = (episode["robot_states"][i+2][3:5] - episode["robot_states"][i+1][3:5]) / test_data[test]["specifics"]["time_step"]
                        instant_jerk = (next_acceleration - instant_acceleration) / test_data[test]["specifics"]["time_step"]
                        instant_jerk_norm = np.linalg.norm(instant_jerk)
                        average_jerk += instant_jerk_norm
                        if instant_jerk_norm < minimum_jerk: minimum_jerk = instant_jerk_norm
                        if instant_jerk_norm > maximum_jerk: maximum_jerk = instant_jerk_norm
            average_velocity /= len(episode["robot_states"])
            average_acceleration /= (len(episode["robot_states"]) - 1)
            average_jerk /= (len(episode["robot_states"]) - 2)
            average_distance /= len(episode["robot_states"])
            trial_space_compliance /= len(episode["robot_states"])
            average_minimum_velocity += minimum_velocity
            average_trial_velocity += average_velocity
            average_maximum_velocity += maximum_velocity
            average_minimum_acceleration += minimum_acceleration
            average_trial_acceleration += average_acceleration
            average_maximum_acceleration += maximum_acceleration
            average_minimum_jerk += minimum_jerk
            average_trial_jerk += average_jerk
            average_maximum_jerk += maximum_jerk
            average_minimum_distance_to_humans += minimum_distance
            average_trial_distance_to_humans += average_distance
            average_trial_space_compliance += trial_space_compliance
            if episode["success"]: average_path_length += trial_path_length
            success_weighted_by_path_length += int(episode["success"]) * (shortest_path_length / max(shortest_path_length, trial_path_length))
        time_to_goal = time_to_goal/successes if successes > 0 else None
        average_minimum_velocity /= test_data[test]['specifics']['trials']
        average_trial_velocity /= test_data[test]['specifics']['trials']
        average_maximum_velocity /= test_data[test]['specifics']['trials']
        average_minimum_acceleration /= test_data[test]['specifics']['trials']
        average_trial_acceleration /= test_data[test]['specifics']['trials']
        average_maximum_acceleration /= test_data[test]['specifics']['trials']
        average_minimum_jerk /= test_data[test]['specifics']['trials']
        average_trial_jerk /= test_data[test]['specifics']['trials']
        average_maximum_jerk /= test_data[test]['specifics']['trials']
        average_minimum_distance_to_humans /= test_data[test]['specifics']['trials']
        average_trial_distance_to_humans /= test_data[test]['specifics']['trials']
        average_trial_space_compliance /= test_data[test]['specifics']['trials']
        average_path_length = average_path_length/successes if successes > 0 else None
        success_weighted_by_path_length = success_weighted_by_path_length/test_data[test]['specifics']['trials']
        print(f"SUCCESS RATE: {successes/test_data[test]['specifics']['trials']}")
        print(f"COLLISIONS OVER {test_data[test]['specifics']['trials']} TRIALS: {collisions}")
        print(f"TRUNCATED EPISODES OVER {test_data[test]['specifics']['trials']} TRIALS: {truncated_epsiodes}")
        print(f"AVERAGE TIME TO GOAL: {time_to_goal}")
        print(f"AVERAGE MINIMUM VELOCITY NORM: {average_minimum_velocity}")
        print(f"AVERAGE VELOCITY NORM: {average_trial_velocity}")
        print(f"AVERAGE MAXIMUM VELOCITY NORM: {average_maximum_velocity}")
        print(f"AVERAGE MINIMUM ACCELERATION NORM: {average_minimum_acceleration}")
        print(f"AVERAGE ACCELERATION NORM: {average_trial_acceleration}")
        print(f"AVERAGE MAXIMUM ACCELERATION NORM: {average_maximum_acceleration}")
        print(f"AVERAGE MINIMUM JERK NORM: {average_minimum_jerk}")
        print(f"AVERAGE JERK NORM: {average_trial_jerk}")
        print(f"AVERAGE MAXIMUM JERK NORM: {average_maximum_jerk}")
        print(f"AVERAGE MINIMUM DISTANCE TO HUMANS: {average_minimum_distance_to_humans}")
        print(f"AVERAGE DISTANCE TO HUMANS: {average_trial_distance_to_humans}")
        print(f"AVERAGE SPACE COMPLIANCE (with threshold {SPACE_COMPLIANCE_THRESHOLD}): {average_trial_space_compliance}")
        print(f"AVERAGE PATH LENGTH (counted only if the robot reaches the goal): {average_path_length}")
        print(f"SUCCESS WEIGHTED BY PATH LENGTH: {success_weighted_by_path_length}")
        print("")
        test_success_rate[k] = successes/test_data[test]['specifics']['trials']
        test_collisions[k] = collisions
        test_truncated_eps[k] = truncated_epsiodes
        test_time_to_goal[k] = time_to_goal
        test_avg_min_vel[k] = average_minimum_velocity
        test_avg_vel[k] = average_trial_velocity
        test_max_vel[k] = average_maximum_velocity
        test_avg_min_accel[k] = average_minimum_acceleration
        test_avg_accel[k] = average_trial_acceleration
        test_max_accel[k] = average_maximum_acceleration
        test_avg_min_jerk[k] = average_minimum_jerk
        test_avg_jerk[k] = average_trial_jerk
        test_max_jerk[k] = average_maximum_jerk
        test_avg_min_dist[k] = average_minimum_distance_to_humans
        test_avg_dist[k] = average_trial_distance_to_humans
        test_avg_space_compl[k] = average_trial_space_compliance
        test_avg_path_length[k] = average_path_length
        test_spl[k] = success_weighted_by_path_length
    return zip(test_success_rate,test_collisions,test_truncated_eps,test_time_to_goal,test_avg_min_vel,test_avg_vel, test_max_vel,test_avg_min_accel,test_avg_accel,test_max_accel,test_avg_min_jerk,test_avg_jerk,test_max_jerk, test_avg_min_dist,test_avg_dist,test_avg_space_compl,test_avg_path_length,test_spl) 

# SINGLE POST-PROCESSING
if SINGLE_PROCESSING:
    with open(os.path.join(os.path.dirname(__file__),'tests','results',RESULTS_FILE), "rb") as f:
        test_data = pickle.load(f)

    metrics  = single_results_file_post_processing(test_data)
    metrics_dataframe = pd.DataFrame(metrics, columns=METRICS, index=TESTS)
    print(metrics_dataframe.head())
    if EXPORT_ON_EXCEL:
        file_name = os.path.join(metrics_dir,f"Metrics_{test_data['5_humans']['specifics']['robot_policy']}_on_{test_data['5_humans']['specifics']['human_policy']}.xlsx")
        with pd.ExcelWriter(file_name) as writer: metrics_dataframe.to_excel(writer, sheet_name='metrics')
# MULTIPLE POST-PROCESSING
else:
    # Here we'll save the metrics for each policy for each type of test (based on n humans)
    five_humans_test_metrics = []
    seven_humans_test_metrics = []
    fourteen_humans_test_metrics = []
    twentyone_humans_test_metrics = []
    twentyeight_humans_test_metrics = []
    thirtyfive_humans_test_metrics = []
    # Post-processing
    for results in RESULTS_FILES:
        with open(os.path.join(os.path.dirname(__file__),'tests','results',results), "rb") as f:
            test_data = pickle.load(f)
        metrics = single_results_file_post_processing(test_data)
        metrics_for_each_test = [test_metrics for test_metrics in metrics]
        five_humans_test_metrics.append(metrics_for_each_test[0])
        seven_humans_test_metrics.append(metrics_for_each_test[1])
        fourteen_humans_test_metrics.append(metrics_for_each_test[2])
        twentyone_humans_test_metrics.append(metrics_for_each_test[3])
        twentyeight_humans_test_metrics.append(metrics_for_each_test[4])
        thirtyfive_humans_test_metrics.append(metrics_for_each_test[5])
    # Dataframes creation
    five_humans_metrics_dataframe = pd.DataFrame(five_humans_test_metrics, columns=METRICS, index=RESULTS_FILES)
    print(five_humans_metrics_dataframe.head())
    seven_humans_metrics_dataframe = pd.DataFrame(seven_humans_test_metrics, columns=METRICS, index=RESULTS_FILES)
    print(seven_humans_metrics_dataframe.head())
    fourteen_humans_metrics_dataframe = pd.DataFrame(fourteen_humans_test_metrics, columns=METRICS, index=RESULTS_FILES)
    print(fourteen_humans_metrics_dataframe.head())
    twentyone_humans_metrics_dataframe = pd.DataFrame(twentyone_humans_test_metrics, columns=METRICS, index=RESULTS_FILES)
    print(twentyone_humans_metrics_dataframe.head())
    twentyeight_humans_metrics_dataframe = pd.DataFrame(twentyeight_humans_test_metrics, columns=METRICS, index=RESULTS_FILES)
    print(twentyeight_humans_metrics_dataframe.head())
    thirtyfive_humans_metrics_dataframe = pd.DataFrame(thirtyfive_humans_test_metrics, columns=METRICS, index=RESULTS_FILES)
    print(thirtyfive_humans_metrics_dataframe.head())
    # Export on Excel
    if EXPORT_ON_EXCEL:
        file_name = os.path.join(metrics_dir,f"Metrics_multiple_robot_policies.xlsx")
        with pd.ExcelWriter(file_name) as writer: 
            five_humans_metrics_dataframe.to_excel(writer, sheet_name='5_humans')
            seven_humans_metrics_dataframe.to_excel(writer, sheet_name='7_humans')
            fourteen_humans_metrics_dataframe.to_excel(writer, sheet_name='14_humans')
            twentyone_humans_metrics_dataframe.to_excel(writer, sheet_name='21_humans')
            twentyeight_humans_metrics_dataframe.to_excel(writer, sheet_name='28_humans')
            thirtyfive_humans_metrics_dataframe.to_excel(writer, sheet_name='35_humans')
    # Plot some metrics
    if PLOT_METRICS: pass