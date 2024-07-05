import pickle
import os
import numpy as np
import pandas as pd

## GLOBAL VARIABLES TO SET
SINGLE_PROCESSING = False # If true, a single results file is post-processed. Otherwise a list provided is post-processed
SPACE_COMPLIANCE_THRESHOLD = 0.5
# Reward parameters
SUCCESS_REWARD = 1
COLLISION_PENALTY = -0.25
DISCOMFORT_DIST = 0.2
DISCOMFORT_PENALTY_FACTOR = 0.5
DISCOUNT_FACTOR = 0.9
EXPORT_DATA = True # If true, resulting metrics are exported
MULTIPLE_TESTS_EXCEL_OUTPUT_FILE_NAME = "PT_on_PT"
## SINGLE POSTPROCESSING
RESULTS_FILE = "bp_on_orca.pkl"
## MULTIPLE POSTPROCESSING
RESULTS_FILES = ["bp_on_orca.pkl","bp_on_sfm_guo.pkl","bp_on_hsfm_new_guo.pkl","ssp_on_orca.pkl","ssp_on_sfm_guo.pkl","ssp_on_hsfm_new_guo.pkl",
                 "orca_on_orca.pkl","orca_on_sfm_guo.pkl","orca_on_hsfm_new_guo.pkl",
                 "cadrl_on_orca_on_orca.pkl","cadrl_on_orca_on_sfm_guo.pkl","cadrl_on_orca_on_hsfm_new_guo.pkl",
                 "cadrl_on_sfm_guo_on_orca.pkl","cadrl_on_sfm_guo_on_sfm_guo.pkl","cadrl_on_sfm_guo_on_hsfm_new_guo.pkl",
                 "cadrl_on_hsfm_new_guo_on_orca.pkl","cadrl_on_hsfm_new_guo_on_sfm_guo.pkl","cadrl_on_hsfm_new_guo_on_hsfm_new_guo.pkl",
                 "sarl_on_orca_on_orca.pkl","sarl_on_orca_on_sfm_guo.pkl","sarl_on_orca_on_hsfm_new_guo.pkl",
                 "sarl_on_sfm_guo_on_orca.pkl","sarl_on_sfm_guo_on_sfm_guo.pkl","sarl_on_sfm_guo_on_hsfm_new_guo.pkl",
                 "sarl_on_hsfm_new_guo_on_orca.pkl","sarl_on_hsfm_new_guo_on_sfm_guo.pkl","sarl_on_hsfm_new_guo_on_hsfm_new_guo.pkl",
                 "lstm_rl_on_orca_on_orca.pkl","lstm_rl_on_orca_on_sfm_guo.pkl","lstm_rl_on_orca_on_hsfm_new_guo.pkl",
                 "lstm_rl_on_sfm_guo_on_orca.pkl","lstm_rl_on_sfm_guo_on_sfm_guo.pkl","lstm_rl_on_sfm_guo_on_hsfm_new_guo.pkl",
                 "lstm_rl_on_hsfm_new_guo_on_orca.pkl","lstm_rl_on_hsfm_new_guo_on_sfm_guo.pkl","lstm_rl_on_hsfm_new_guo_on_hsfm_new_guo.pkl"]
# results files of policies+ tests
# RESULTS_FILES = ["bp_on_hsfm_new_guo.pkl",
#                  "ssp_on_hsfm_new_guo.pkl",
#                  "orca_on_hsfm_new_guo.pkl",
#                  "cadrl_h_on_hsfm_new_guo_on_hsfm_new_guo.pkl",
#                  "sarl_h_on_hsfm_new_guo_on_hsfm_new_guo.pkl",
#                  "lstm_rl_h_on_hsfm_new_guo_on_hsfm_new_guo.pkl"]
## IMPLEMENTATION VARIABLES - DO NOT CHANGE
TESTS = ["5_humans","10_humans","15_humans","20_humans","25_humans"] # 35_humans
METRICS = ['success_rate','collisions','truncated_eps','time_to_goal','min_speed','avg_speed',
           'max_speed','min_accel.','avg_accel.','max_accel.','min_jerk','avg_jerk','max_jerk',
           'min_dist','avg_dist','space_compliance','path_length','SPL',"return"]
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
    # COMPLETE FINAL METRICS - We assume tests with different number of humans have same number of trials
    complete_data = np.zeros((len(TESTS),test_data["5_humans"]["specifics"]["trials"],len(METRICS)), dtype=np.float64)
    # AVERAGE FINAL METRICS
    average_data = np.empty((len(TESTS),len(METRICS)), dtype=np.float64)
    # POST-PROCESSING
    for k, test in enumerate(TESTS):
        print(" ************************ ")
        print("")
        print(f"{test} TEST SPECIFICS")
        print(test_data[test]["specifics"])
        collisions = 0
        successes = 0
        truncated_epsiodes = 0
        # METRICS COMPUTATION
        for t, episode in enumerate(test_data[test]["results"]):
            # Success, collision, truncated metrics
            collisions += int(episode["collision"])
            successes += int(episode["success"])
            truncated_epsiodes += int(episode["truncated"])
            # Initialize trial metrics
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
            space_compliance = 0
            path_length = 0
            shortest_path_length = 0
            discounted_return = 0
            # Compute "time-wise metrics"
            for i, robot_state in enumerate(episode["robot_states"]):
                instant_position = robot_state[0:2]
                if i == 0: shortest_path_length = np.linalg.norm(robot_state[6:8] - instant_position) # Distance between goal and initial configuration
                if i > 0: 
                    previous_position = episode["robot_states"][i-1][0:2]
                    path_length += np.linalg.norm(instant_position - previous_position)
                average_instant_distance = 0
                instant_space_compliance = True
                minimum_instant_distance = BIG_FLOAT
                for p, human_state in enumerate(episode["human_states"][i]):
                    human_position = human_state[0:2]
                    instant_distance = np.linalg.norm(human_position - instant_position) - (test_data[test]["specifics"]["robot_radius"] + test_data[test]["specifics"]["humans_radiuses"][p])
                    average_instant_distance += instant_distance
                    if instant_distance < minimum_distance and not episode["collision"]: minimum_distance = instant_distance
                    if instant_distance < SPACE_COMPLIANCE_THRESHOLD: instant_space_compliance = False
                    if instant_distance < minimum_instant_distance: minimum_instant_distance = instant_distance
                space_compliance += int(instant_space_compliance)
                average_distance += average_instant_distance / len(episode["human_states"][i])
                # Reward
                reward = 0
                reward += int(np.linalg.norm(robot_state[6:8] - instant_position) < test_data[test]["specifics"]["robot_radius"]) # Goal reward
                reward += int((minimum_instant_distance > 0) and (minimum_instant_distance < DISCOMFORT_DIST)) * DISCOMFORT_PENALTY_FACTOR * (minimum_instant_distance - DISCOMFORT_DIST) * test_data[test]["specifics"]["robot_time_step"]
                reward += int(minimum_instant_distance <= 0) * COLLISION_PENALTY
                # Velocity
                instant_velocity = robot_state[3:5]
                instant_velocity_norm = np.linalg.norm(instant_velocity)
                average_velocity += instant_velocity_norm
                if instant_velocity_norm < minimum_velocity: minimum_velocity = instant_velocity_norm
                if instant_velocity_norm > maximum_velocity: maximum_velocity = instant_velocity_norm
                # Acceleration
                if i < len(episode["robot_states"])-1: 
                    instant_acceleration = (episode["robot_states"][i+1][3:5] - instant_velocity) / test_data[test]["specifics"]["robot_time_step"]
                    instant_acceleration_norm = np.linalg.norm(instant_acceleration)
                    average_acceleration += instant_acceleration_norm
                    if instant_acceleration_norm < minimum_acceleration: minimum_acceleration = instant_acceleration_norm
                    if instant_acceleration_norm > maximum_acceleration: maximum_acceleration = instant_acceleration_norm
                # Jerk
                if i < len(episode["robot_states"])-2: 
                    next_acceleration = (episode["robot_states"][i+2][3:5] - episode["robot_states"][i+1][3:5]) / test_data[test]["specifics"]["robot_time_step"]
                    instant_jerk = (next_acceleration - instant_acceleration) / test_data[test]["specifics"]["robot_time_step"]
                    instant_jerk_norm = np.linalg.norm(instant_jerk)
                    average_jerk += instant_jerk_norm
                    if instant_jerk_norm < minimum_jerk: minimum_jerk = instant_jerk_norm
                    if instant_jerk_norm > maximum_jerk: maximum_jerk = instant_jerk_norm
                # Discounted return
                discounted_return += (pow(DISCOUNT_FACTOR,i*test_data[test]["specifics"]["robot_time_step"]) * reward) # Desired speed is assumed to be 1
            # Averaged metrics over time
            average_velocity /= len(episode["robot_states"])
            if len(episode["robot_states"]) > 1: average_acceleration /= (len(episode["robot_states"]) - 1)
            else: average_acceleration = 0
            if len(episode["robot_states"]) > 2: average_jerk /= (len(episode["robot_states"]) - 2)
            else: average_jerk = 0
            average_distance /= len(episode["robot_states"])
            space_compliance /= len(episode["robot_states"])
            # Save metrics for each trial
            complete_data[k][t][0] = int(episode["success"])
            complete_data[k][t][1] = int(episode["collision"])
            complete_data[k][t][2] = int(episode["truncated"])
            complete_data[k][t][17] = int(episode["success"]) * (shortest_path_length / max(shortest_path_length, path_length))
            complete_data[k][t][18] = discounted_return
            if episode["success"]:
                complete_data[k][t][3] = episode["time_to_goal"]
                complete_data[k][t][4] = minimum_velocity
                complete_data[k][t][5] = average_velocity
                complete_data[k][t][6] = maximum_velocity
                complete_data[k][t][7] = minimum_acceleration
                complete_data[k][t][8] = average_acceleration
                complete_data[k][t][9] = maximum_acceleration
                complete_data[k][t][10] = minimum_jerk
                complete_data[k][t][11] = average_jerk
                complete_data[k][t][12] = maximum_jerk
                complete_data[k][t][13] = minimum_distance
                complete_data[k][t][14] = average_distance
                complete_data[k][t][15] = space_compliance
                complete_data[k][t][16] = path_length
            else:
                complete_data[k][t][3] = np.NaN
                complete_data[k][t][4] = np.NaN
                complete_data[k][t][5] = np.NaN
                complete_data[k][t][6] = np.NaN
                complete_data[k][t][7] = np.NaN
                complete_data[k][t][8] = np.NaN
                complete_data[k][t][9] = np.NaN
                complete_data[k][t][10] = np.NaN
                complete_data[k][t][11] = np.NaN
                complete_data[k][t][12] = np.NaN
                complete_data[k][t][13] = np.NaN
                complete_data[k][t][14] = np.NaN
                complete_data[k][t][15] = np.NaN
                complete_data[k][t][16] = np.NaN
        # Save average data (over trials)
        average_data[k][0] = round(np.sum(complete_data[k,:,0])/test_data[test]['specifics']['trials'],2)
        average_data[k][1] = round(np.sum(complete_data[k,:,1]),2)
        average_data[k][2] = round(np.sum(complete_data[k,:,2]),2)
        average_data[k][3] = round(np.nansum(complete_data[k,:,3])/successes,2)
        average_data[k][4] = round(np.nansum(complete_data[k,:,4])/successes,2)
        average_data[k][5] = round(np.nansum(complete_data[k,:,5])/successes,2)
        average_data[k][6] = round(np.nansum(complete_data[k,:,6])/successes,2)
        average_data[k][7] = round(np.nansum(complete_data[k,:,7])/successes,2)
        average_data[k][8] = round(np.nansum(complete_data[k,:,8])/successes,2)
        average_data[k][9] = round(np.nansum(complete_data[k,:,9])/successes,2)
        average_data[k][10] = round(np.nansum(complete_data[k,:,10])/successes,2)
        average_data[k][11] = round(np.nansum(complete_data[k,:,11])/successes,2)
        average_data[k][12] = round(np.nansum(complete_data[k,:,12])/successes,2)
        average_data[k][13] = round(np.nansum(complete_data[k,:,13])/successes,2)
        average_data[k][14] = round(np.nansum(complete_data[k,:,14])/successes,2)
        average_data[k][15] = round(np.nansum(complete_data[k,:,15])/successes,2)
        average_data[k][16] = round(np.nansum(complete_data[k,:,16])/successes,2)
        average_data[k][17] = round(np.nansum(complete_data[k,:,17])/test_data[test]['specifics']['trials'],2)
        average_data[k][18] = round(np.sum(complete_data[k,:,18])/test_data[test]['specifics']['trials'],2)
        # Print computed metrics
        print(f"SUCCESS RATE: {average_data[k][0]}")
        print(f"COLLISIONS OVER {test_data[test]['specifics']['trials']} TRIALS: {average_data[k][1]}")
        print(f"TRUNCATED EPISODES OVER {test_data[test]['specifics']['trials']} TRIALS: {average_data[k][2]}")
        print(f"AVERAGE TIME TO GOAL: {average_data[k][3]}")
        print(f"AVERAGE MINIMUM VELOCITY NORM: {average_data[k][4]}")
        print(f"AVERAGE VELOCITY NORM: {average_data[k][5]}")
        print(f"AVERAGE MAXIMUM VELOCITY NORM: {average_data[k][6]}")
        print(f"AVERAGE MINIMUM ACCELERATION NORM: {average_data[k][7]}")
        print(f"AVERAGE ACCELERATION NORM: {average_data[k][8]}")
        print(f"AVERAGE MAXIMUM ACCELERATION NORM: {average_data[k][9]}")
        print(f"AVERAGE MINIMUM JERK NORM: {average_data[k][10]}")
        print(f"AVERAGE JERK NORM: {average_data[k][11]}")
        print(f"AVERAGE MAXIMUM JERK NORM: {average_data[k][12]}")
        print(f"AVERAGE MINIMUM DISTANCE TO HUMANS: {average_data[k][13]}")
        print(f"AVERAGE DISTANCE TO HUMANS: {average_data[k][14]}")
        print(f"AVERAGE SPACE COMPLIANCE (with threshold {SPACE_COMPLIANCE_THRESHOLD}): {average_data[k][15]}")
        print(f"AVERAGE PATH LENGTH (counted only if the robot reaches the goal): {average_data[k][16]}")
        print(f"SUCCESS WEIGHTED BY PATH LENGTH: {average_data[k][17]}")
        print(f"AVERAGE RETURN: {average_data[k][18]}")
        print("")
    return average_data, complete_data

# SINGLE POST-PROCESSING
if SINGLE_PROCESSING:
    with open(os.path.join(os.path.dirname(__file__),'tests','results',RESULTS_FILE), "rb") as f:
        test_data = pickle.load(f)

    metrics, complete_metrics  = single_results_file_post_processing(test_data)
    metrics_dataframe = pd.DataFrame(metrics, columns=METRICS, index=TESTS)
    print(metrics_dataframe.head())
    complete_metrics = []
    for i, test in enumerate(TESTS): complete_metrics.append(pd.DataFrame(complete_metrics[i], columns=METRICS))
    if EXPORT_DATA:
        file_name = os.path.join(metrics_dir,f"Metrics_{test_data['5_humans']['specifics']['robot_policy']}_on_{test_data['5_humans']['specifics']['human_policy']}.xlsx")
        with pd.ExcelWriter(file_name) as writer: 
            metrics_dataframe.to_excel(writer, sheet_name='average metrics')
            for i, test in enumerate(TESTS): complete_metrics[i].to_excel(writer, sheet_name=f'{test}')
# MULTIPLE POST-PROCESSING
else:
    # Here we'll save the metrics for each policy for each type of test (based on n humans)
    test_metrics = [[] for _ in range(len(TESTS))]
    # Here we'll save complete metrics data
    complete_metrics_data = np.empty((len(RESULTS_FILES),len(TESTS),100,len(METRICS))) # We assume there are 100 trials for each test
    # Post-processing
    for i, results in enumerate(RESULTS_FILES):
        with open(os.path.join(os.path.dirname(__file__),'tests','results',results), "rb") as f: test_data = pickle.load(f)
        metrics, complete_metrics = single_results_file_post_processing(test_data)
        complete_metrics_data[i] = complete_metrics
        metrics_for_each_test = [test_metrics for test_metrics in metrics]
        for j in range(len(TESTS)): test_metrics[j].append(metrics_for_each_test[j])    
    # Dataframes creation
    test_dataframes = []
    for i in range(len(TESTS)):
        test_dataframes.append(pd.DataFrame(test_metrics[i], columns=METRICS, index=RESULTS_FILES)) 
        print(test_dataframes[i].head())
    # Export data
    if EXPORT_DATA:
        excel_file_name = os.path.join(metrics_dir,f"{MULTIPLE_TESTS_EXCEL_OUTPUT_FILE_NAME}_average_metrics.xlsx")
        with pd.ExcelWriter(excel_file_name) as writer:
            for i, test in enumerate(TESTS): test_dataframes[i].to_excel(writer, sheet_name=f'{test}') 
        with open(os.path.join(metrics_dir,f"{MULTIPLE_TESTS_EXCEL_OUTPUT_FILE_NAME}.pkl"), "wb") as f: pickle.dump(complete_metrics_data, f); f.close()
        