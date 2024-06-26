import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axis import Axis
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import ListedColormap
from matplotlib import rc
from itertools import zip_longest, product
from scipy.stats import ttest_ind, f_oneway
import pickle
import numpy as np
from social_gym.social_nav_sim import SocialNavSim

## Input files
METRICS_FILE = "PT_on_PT_average_metrics.xlsx"
COMPLETE_METRICS_FILE = "PT_on_PT.pkl"
HUMAN_TIMES_FILE = "human_times.pkl"
COMPLETE_METRICS_FILE_NAMES = ["CC_on_CC.pkl","CC_on_PT.pkl","PT_on_CC.pkl","PT_on_PT.pkl","HS_on_CC.pkl","HS_on_PT.pkl"]
## Plots booleans - set true to plot
BAR_PLOTS = False # If true, barplots are shown
MORE_BAR_PLOTS = False # If true, more barplots are plotted
BOX_PLOTS = False # If true, boxplot are printed
HEAT_MAP = False # If true, heatmaps are plotted
SARL_ONLY_HEATMAPS = False # If true, heatmaps are plotted considering only sarl policies
SARL_ONLY_BOXPLOTS = False # If true, boxplots showing performances based on training and testing env are plotted considering only sarl policies
CURVE_PLOTS = False # If true, curves are plotted
SPACE_COMPLIANCE_OVER_SPL = False # If true, space compliance over SPL is plotted
SARL_ONLY_METRICS_OVER_N_HUMANS_TESTS  = False # If true, metrics over n° humans tests are plotted considering only sarl policies
HUMAN_TIMES_BOX_PLOTS = False # If true, humans' time to goal with and without robot are plotted
METRICS_OVER_DIFFERENT_TRAINING_ENVIRONMENT = False # If true, metrics over different training environments are plotted
METRICS_OVER_DIFFERENT_TRAINING_ENV_AND_SCENARIO = False # If true, metrics over different training env and scenarios are plotted
METRICS_BOXPLOTS_OVER_DIFFERENT_TRAINING_ENVS = False # If true, boxplots showing performances based on training env are plotted
METRICS_OVER_DIFFERENT_TESTING_ENVIRONMENT_ONLY_HS = False # If true, metrics over different testing environments are plotted
### PAPER FIGURES
METRICS_OVER_DIFFERENT_POLICIES = False # If true, metrics over different scenarios are plotted
METRICS_OVER_DIFFERENT_TRAINING_SCENARIO = False # If true, metrics over different training scenarios are plotted
METRICS_OVER_DIFFERENT_TRAINING_ENVIRONMENT_ONLY_HS = False # If true, metrics over different training environments are plotted considering only Hybrid train scenario 
METRICS_OVER_BASELINE_POLICIES_AND_SARL = False # If true, metrics over baseline policies and sarl on hybrid scenario are plotted
METRICS_CROSS_TEST_AND_CROSS_TRAIN_ENVS = False # If true, metrics over different training and testing environments are plotted
METRICS_CROSS_SCENARIO_SARL = False # If true, metrics over different scenarios are plotted considering only sarl policies
METRICS_CROSS_TRAIN_ENV_CC = False # If true, metrics over different training environments on CC are plotted
RATES_CROSS_TRAIN_ENV_CC = False # If true, rates over different training environments on CC are plotted
RETURN_SARL_HS_OVER_TRAIN_ENV = True # If true, SARL on HSFM reward over training environments is plotted
PLOT_SARL_HS_TRAJECTORIES = False # If true, SARL on HSFM trajectories are plotted
PLOT_HSFM_HUMANS_TRAJECTORIES_CC = False # If true, HSFM humans' trajectories in CC scenario are plotted
PLOT_HSFM_HUMANS_TRAJECTORIES_PT = False # If true, HSFM humans' trajectories in PT scenario are plotted
PLOT_HS_RETURN_OVER_TRAINING = False # If true, HS return over training environments is plotted
## Specifics
T_TEST_P_VALUE_THRESHOLD = 0.05
NO_TITLES = True # If true, titles are omitted from plots.
PRINT_SUCCESSFUL_EPSIODES = False # If true, the number of successful episodes is printed on some plots
SAVE_FIGURES = True # If true, figures are saved, else, they are showed.
FONTSIZE = 23
## IMPLEMENTATION VARIABLES - DO NOT CHANGE
FIGURES_SAVING_PATH = os.path.join(os.path.dirname(__file__),"tests","plots")
if not os.path.exists(FIGURES_SAVING_PATH): os.makedirs(FIGURES_SAVING_PATH)
TESTS = ["5_humans","10_humans","15_humans","20_humans","25_humans"] # ["5_humans","15_humans","25_humans","35_humans"]
TESTED_ON_ORCA = ["bp_on_orca.pkl",
                  "ssp_on_orca.pkl",
                  "orca_on_orca.pkl",
                  "cadrl_on_orca_on_orca.pkl",
                  "cadrl_on_sfm_guo_on_orca.pkl",
                  "cadrl_on_hsfm_new_guo_on_orca.pkl",
                  "sarl_on_orca_on_orca.pkl",
                  "sarl_on_sfm_guo_on_orca.pkl",
                  "sarl_on_hsfm_new_guo_on_orca.pkl",
                  "lstm_rl_on_orca_on_orca.pkl",
                  "lstm_rl_on_sfm_guo_on_orca.pkl",
                  "lstm_rl_on_hsfm_new_guo_on_orca.pkl"]
TESTED_ON_SFM_GUO = ["bp_on_sfm_guo.pkl",
                     "ssp_on_sfm_guo.pkl",
                     "orca_on_sfm_guo.pkl",
                     "cadrl_on_orca_on_sfm_guo.pkl",
                     "cadrl_on_sfm_guo_on_sfm_guo.pkl",
                     "cadrl_on_hsfm_new_guo_on_sfm_guo.pkl",
                     "sarl_on_orca_on_sfm_guo.pkl",
                     "sarl_on_sfm_guo_on_sfm_guo.pkl",
                     "sarl_on_hsfm_new_guo_on_sfm_guo.pkl",
                     "lstm_rl_on_orca_on_sfm_guo.pkl",
                     "lstm_rl_on_sfm_guo_on_sfm_guo.pkl",
                     "lstm_rl_on_hsfm_new_guo_on_sfm_guo.pkl"]
TESTED_ON_HSFM_NEW_GUO = ["bp_on_hsfm_new_guo.pkl",
                          "ssp_on_hsfm_new_guo.pkl",
                          "orca_on_hsfm_new_guo.pkl",
                          "cadrl_on_orca_on_hsfm_new_guo.pkl",
                          "cadrl_on_sfm_guo_on_hsfm_new_guo.pkl",
                          "cadrl_on_hsfm_new_guo_on_hsfm_new_guo.pkl",
                          "sarl_on_orca_on_hsfm_new_guo.pkl",
                          "sarl_on_sfm_guo_on_hsfm_new_guo.pkl",
                          "sarl_on_hsfm_new_guo_on_hsfm_new_guo.pkl",
                          "lstm_rl_on_orca_on_hsfm_new_guo.pkl",
                          "lstm_rl_on_sfm_guo_on_hsfm_new_guo.pkl",
                          "lstm_rl_on_hsfm_new_guo_on_hsfm_new_guo.pkl"]
TRAINED_ON_ORCA = ["cadrl_on_orca_on_orca.pkl","cadrl_on_orca_on_sfm_guo.pkl","cadrl_on_orca_on_hsfm_new_guo.pkl",
                   "sarl_on_orca_on_orca.pkl","sarl_on_orca_on_sfm_guo.pkl","sarl_on_orca_on_hsfm_new_guo.pkl",
                   "lstm_rl_on_orca_on_orca.pkl","lstm_rl_on_orca_on_sfm_guo.pkl","lstm_rl_on_orca_on_hsfm_new_guo.pkl"]
TRAINED_ON_SFM_GUO = ["cadrl_on_sfm_guo_on_orca.pkl","cadrl_on_sfm_guo_on_sfm_guo.pkl","cadrl_on_sfm_guo_on_hsfm_new_guo.pkl",
                      "sarl_on_sfm_guo_on_orca.pkl","sarl_on_sfm_guo_on_sfm_guo.pkl","sarl_on_sfm_guo_on_hsfm_new_guo.pkl",
                      "lstm_rl_on_sfm_guo_on_orca.pkl","lstm_rl_on_sfm_guo_on_sfm_guo.pkl","lstm_rl_on_sfm_guo_on_hsfm_new_guo.pkl"]
TRAINED_ON_HSFM_NEW_GUO = ["cadrl_on_hsfm_new_guo_on_orca.pkl","cadrl_on_hsfm_new_guo_on_sfm_guo.pkl","cadrl_on_hsfm_new_guo_on_hsfm_new_guo.pkl",
                           "sarl_on_hsfm_new_guo_on_orca.pkl","sarl_on_hsfm_new_guo_on_sfm_guo.pkl","sarl_on_hsfm_new_guo_on_hsfm_new_guo.pkl",
                           "lstm_rl_on_hsfm_new_guo_on_orca.pkl","lstm_rl_on_hsfm_new_guo_on_sfm_guo.pkl","lstm_rl_on_hsfm_new_guo_on_hsfm_new_guo.pkl"]
TRAINED_POLICIES_TESTS = TRAINED_ON_ORCA + TRAINED_ON_SFM_GUO + TRAINED_ON_HSFM_NEW_GUO
TRAINED_POLICIES = ["cadrl_on_orca","cadrl_on_sfm_guo","cadrl_on_hsfm_new_guo","sarl_on_orca","sarl_on_sfm_guo","sarl_on_hsfm_new_guo","lstm_rl_on_orca","lstm_rl_on_sfm_guo","lstm_rl_on_hsfm_new_guo"]
SARL_POLICIES = ["sarl_on_orca","sarl_on_sfm_guo","sarl_on_hsfm_new_guo"]
SARL_POLICIES_RESULTS = ["sarl_on_orca_on_orca.pkl","sarl_on_orca_on_sfm_guo.pkl","sarl_on_orca_on_hsfm_new_guo.pkl",
                 "sarl_on_sfm_guo_on_orca.pkl","sarl_on_sfm_guo_on_sfm_guo.pkl","sarl_on_sfm_guo_on_hsfm_new_guo.pkl",
                 "sarl_on_hsfm_new_guo_on_orca.pkl","sarl_on_hsfm_new_guo_on_sfm_guo.pkl","sarl_on_hsfm_new_guo_on_hsfm_new_guo.pkl"]
POLICY_NAMES = ["bp",
                "ssp",
                "orca",
                "cadrl_on_orca",
                "cadrl_on_sfm_guo",
                "cadrl_on_hsfm_new_guo",
                "sarl_on_orca",
                "sarl_on_sfm_guo",
                "sarl_on_hsfm_new_guo",
                "lstm_rl_on_orca",
                "lstm_rl_on_sfm_guo",
                "lstm_rl_on_hsfm_new_guo"]
TRAINABLE_POLICIES =  ["cadrl","sarl","lstm_rl"]
TRAINABLE_POLICIES_DISPLAY = ["CADRL","SARL","LSTM"]
ENVIRONMENTS = ["ORCA","SFM_GUO","HSFM_NEW_GUO"]
ENVIRONMENTS_DISPLAY_NAME = ["ORCA","SFM","HSFM"]
COLORS = list(mcolors.TABLEAU_COLORS.values())
OTHER_COLORS = ['peachpuff', 'orange', 'tomato']
OTHER_COLORS_2 = ['deeppink','mediumorchid','dodgerblue']
METRICS = ['success_rate','collisions','truncated_eps','time_to_goal','min_speed','avg_speed',
           'max_speed','min_accel.','avg_accel.','max_accel.','min_jerk','avg_jerk','max_jerk',
           'min_dist','avg_dist','space_compliance','path_length','SPL','return']
PLOT_COUNTER = 1
TEST_DIMENSIONS = {0: "robot_policy", 1: "train_env", 2: "train_scenario", 
                   3: "test_env", 4: "test_scenario", 5: "n_humans"}
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
ROBOT_POLICIES_RESULTS_FILES_INDEXES = {"bp": [0,1,2], "ssp": [3,4,5], "orca": [6,7,8], "cadrl": [9,10,11,12,13,14,15,16,17],
                                        "sarl": [18,19,20,21,22,23,24,25,26], "lstm_rl": [27,28,29,30,31,32,33,34,35]}
TRAIN_ENV_RESULTS_FILES_INDEXES = {"ORCA": [9,10,11,18,19,20,27,28,29], "SFM_GUO": [12,13,14,21,22,23,30,31,32], "HSFM_NEW_GUO": [15,16,17,24,25,26,33,34,35]}
TEST_ENV_RESULTS_FILES_INDEXES = {"ORCA": [0,3,6,9,12,15,18,21,24,27,30,33], "SFM_GUO": [1,4,7,10,13,16,19,22,25,28,31,34], "HSFM_NEW_GUO": [2,5,8,11,14,17,20,23,26,29,32,35]}
SCENARIOS = ["CC","PT","HS"]
SCENARIOS_DISPLAY_NAME = ["Circular crossing","Parallel traffic","Hybrid scenario"]
NON_TRAINABLE_POLICIES = POLICY_NAMES[0:3]
LINE_STYLES = ["solid","dashed","dotted"]
TESTS_DISPLAY_NAME = [t.split("_humans")[0] for t in TESTS]

### Matplotlib font
font = {'weight' : 'regular',
        'size'   : FONTSIZE}
rc('font', **font)

def find_key_containing_a_certain_value_in_dict(dictionary:dict, value:str):
    for key, values in dictionary.items(): 
        if value in values: return key
    return None

def aggregate_data(complete_metrics_files:list[str], metrics_dir:str, aggregation_dimensions:list[int], only_sarl=False):
    ### Aggregates only trainable policies data
    ### First we create a dictionary containing all the test organized by the aggregation dimensions
    if not only_sarl: n_tests = len(complete_metrics_files) * len(RESULTS_FILES) * len(TESTS)
    else: n_tests = len(complete_metrics_files) * len(SARL_POLICIES_RESULTS) * len(TESTS)
    tests_data = {}
    n_tests_non_trainable_policies = 0
    for i, compl_metrics_file in enumerate(complete_metrics_files):
        with open(os.path.join(metrics_dir,compl_metrics_file), "rb") as f: complete_data_i = pickle.load(f)
        ## complete_data_i shape: [n_tests, n_humans, n_trials (i.e., 100), n_metrics]
        assert len(complete_data_i) == len(RESULTS_FILES), f"Error: {compl_metrics_file} does not contain the standard number of tests"
        train_scenario = compl_metrics_file[0:2]
        assert train_scenario in SCENARIOS, f"Error: {train_scenario} is not a valid train scenario"
        test_scenario = compl_metrics_file[6:8]
        assert test_scenario in SCENARIOS[0:2], f"Error: {test_scenario} is not a valid test scenario"
        ## loop through tests of each file
        for t, test in enumerate(complete_data_i):
            ## now we need to find out the robot policy, train env, and test env
            results_files_index = RESULTS_FILES.index(RESULTS_FILES[t])
            # robot policy
            robot_policy = find_key_containing_a_certain_value_in_dict(ROBOT_POLICIES_RESULTS_FILES_INDEXES, results_files_index)
            # train env
            train_env = find_key_containing_a_certain_value_in_dict(TRAIN_ENV_RESULTS_FILES_INDEXES, results_files_index)
            # test env
            test_env = find_key_containing_a_certain_value_in_dict(TEST_ENV_RESULTS_FILES_INDEXES, results_files_index)
            ## now we further split by n_humans
            for h, true_test in enumerate(test):
                if h < len(TESTS): n_humans = TESTS[h]
                else: continue
                if robot_policy in NON_TRAINABLE_POLICIES and not only_sarl: n_tests_non_trainable_policies += 1
                ## now we save the data in our dict - remember that true_test is in the shape: [n_trials, n_metrics]
                if robot_policy not in NON_TRAINABLE_POLICIES: 
                    if only_sarl and robot_policy != "sarl": continue
                    else: tests_data[len(tests_data)] = {"robot_policy": robot_policy, "train_env": train_env, "train_scenario": train_scenario, "test_env": test_env, "test_scenario": test_scenario, "n_humans": n_humans, "data": true_test}
    n_tests -= n_tests_non_trainable_policies
    assert len(tests_data) == n_tests, f"Error: the number of tests is not correct. Expected {n_tests}, got {len(tests_data)}"
    ### Now we aggregate the data
    if len(aggregation_dimensions) == 0: 
        return tests_data
    else:
        print(f"Aggregating test data by {[v for k,v in TEST_DIMENSIONS.items() if k in aggregation_dimensions]}...")
        ## Let's find the final number of different tests
        if only_sarl: aggregation_tp_divisors = {0: 1, 1: len(ENVIRONMENTS), 2: len(SCENARIOS), 3: len(ENVIRONMENTS), 4: len(SCENARIOS[0:2]), 5: len(TESTS)}
        else: aggregation_tp_divisors = {0: len(ROBOT_POLICIES_RESULTS_FILES_INDEXES) - len(NON_TRAINABLE_POLICIES), 1: len(ENVIRONMENTS), 2: len(SCENARIOS), 3: len(ENVIRONMENTS), 4: len(SCENARIOS[0:2]), 5: len(TESTS)}
        trainable_policies_divisor = np.prod([aggregation_tp_divisors[dimension] for dimension in aggregation_dimensions])
        n_aggregated_tests = int(n_tests / trainable_policies_divisor)
        print(f"Total number of different tests: {n_tests} - Number of aggregated tests: {n_aggregated_tests}")
        ## Now we aggregate the data
        non_aggregation_dimensions = [i for i in range(len(TEST_DIMENSIONS)) if i not in aggregation_dimensions]
        # create the set of all possible test settings
        tp_dims = []
        if 0 not in aggregation_dimensions: tp_dims.append(TRAINABLE_POLICIES)
        if 1 not in aggregation_dimensions: tp_dims.append(ENVIRONMENTS)
        if 2 not in aggregation_dimensions: tp_dims.append(SCENARIOS)
        if 3 not in aggregation_dimensions: tp_dims.append(ENVIRONMENTS)
        if 4 not in aggregation_dimensions: tp_dims.append(SCENARIOS[0:2])
        if 5 not in aggregation_dimensions: tp_dims.append(TESTS)
        sets = list(product(*tp_dims))
        aggregated_tests_data = {}
        for i in range(n_aggregated_tests):
            cset = sets[i]
            aggr_test = {}
            for d, dim in enumerate(non_aggregation_dimensions): aggr_test[TEST_DIMENSIONS[dim]] = cset[d]
            # print(cset, aggr_test)
            for t in range(len(tests_data)):
                test = tests_data[t]
                if all([bool(test[TEST_DIMENSIONS[dim]] == aggr_test[TEST_DIMENSIONS[dim]]) for dim in non_aggregation_dimensions]):
                    if "data" not in aggr_test: aggr_test["data"] = test["data"]
                    else: aggr_test["data"] = np.concatenate((aggr_test["data"], test["data"]), axis=0)
            aggregated_tests_data[i] = aggr_test
        print(f"Aggregation ended. N° trials for each test: {len(aggregated_tests_data[0]['data'])}")
        return aggregated_tests_data

def aggregate_non_trainable_policies_data(complete_metrics_files:list[str], metrics_dir:str, aggregation_dimensions:list[int]):
    tests_data = {}
    n_tests = len(NON_TRAINABLE_POLICIES) * len(ENVIRONMENTS) * len(SCENARIOS[0:2]) * len(TESTS)
    n_tests_non_trainable_policies = 0
    explored_test_scenarios = []
    assert 1 not in aggregation_dimensions and 2 not in aggregation_dimensions, "Error: cannot aggregate non-trainable policies data by train settings"
    for i, compl_metrics_file in enumerate(complete_metrics_files):
        with open(os.path.join(metrics_dir,compl_metrics_file), "rb") as f: complete_data_i = pickle.load(f)
        ## complete_data_i shape: [n_tests, n_humans, n_trials (i.e., 100), n_metrics]
        assert len(complete_data_i) == len(RESULTS_FILES), f"Error: {compl_metrics_file} does not contain the standard number of tests"
        train_scenario = compl_metrics_file[0:2]
        assert train_scenario in SCENARIOS, f"Error: {train_scenario} is not a valid train scenario"
        test_scenario = compl_metrics_file[6:8]
        if test_scenario not in explored_test_scenarios: 
            explored_test_scenarios.append(test_scenario)
            assert test_scenario in SCENARIOS[0:2], f"Error: {test_scenario} is not a valid test scenario"
            ## loop through tests of each file
            for t, test in enumerate(complete_data_i):
                ## now we need to find out the robot policy, train env, and test env
                results_files_index = RESULTS_FILES.index(RESULTS_FILES[t])
                # robot policy
                robot_policy = find_key_containing_a_certain_value_in_dict(ROBOT_POLICIES_RESULTS_FILES_INDEXES, results_files_index)
                # train env
                train_env = find_key_containing_a_certain_value_in_dict(TRAIN_ENV_RESULTS_FILES_INDEXES, results_files_index)
                # test env
                test_env = find_key_containing_a_certain_value_in_dict(TEST_ENV_RESULTS_FILES_INDEXES, results_files_index)
                ## now we further split by n_humans
                for h, true_test in enumerate(test):
                    if h < len(TESTS): n_humans = TESTS[h]
                    else: continue
                    if robot_policy in NON_TRAINABLE_POLICIES: 
                        n_tests_non_trainable_policies += 1
                        tests_data[len(tests_data)] = {"robot_policy": robot_policy, "train_env": train_env, "train_scenario": train_scenario, "test_env": test_env, "test_scenario": test_scenario, "n_humans": n_humans, "data": true_test}
    assert n_tests == n_tests_non_trainable_policies, f"Error: the number of tests is not correct. Expected {n_tests}, got {n_tests_non_trainable_policies}" 
    ### Now we aggregate the data
    if len(aggregation_dimensions) == 0: return tests_data
    else:
        print(f"Aggregating non trainable policies test data by {[v for k,v in TEST_DIMENSIONS.items() if k in aggregation_dimensions]}...")
        ## Let's find the final number of different tests
        aggregation_tp_divisors = {0: len(NON_TRAINABLE_POLICIES), 3: len(ENVIRONMENTS), 4: len(SCENARIOS[0:2]), 5: len(TESTS)}
        trainable_policies_divisor = np.prod([aggregation_tp_divisors[dimension] for dimension in aggregation_dimensions])
        n_aggregated_tests = int(n_tests / trainable_policies_divisor)
        print(f"Total number of different tests: {n_tests} - Number of aggregated tests: {n_aggregated_tests}")
        ## Now we aggregate the data
        non_aggregation_dimensions = [i for i in range(len(TEST_DIMENSIONS)) if (i not in aggregation_dimensions) and (i != 2) and (i != 1)]
        # create the set of all possible test settings
        tp_dims = []
        if 0 not in aggregation_dimensions: tp_dims.append(NON_TRAINABLE_POLICIES)
        if 3 not in aggregation_dimensions: tp_dims.append(ENVIRONMENTS)
        if 4 not in aggregation_dimensions: tp_dims.append(SCENARIOS[0:2])
        if 5 not in aggregation_dimensions: tp_dims.append(TESTS)
        sets = list(product(*tp_dims))
        aggregated_tests_data = {}
        for i in range(n_aggregated_tests):
            cset = sets[i]
            aggr_test = {}
            for d, dim in enumerate(non_aggregation_dimensions): aggr_test[TEST_DIMENSIONS[dim]] = cset[d]
            # print(cset, aggr_test)
            for t in range(len(tests_data)):
                test = tests_data[t]
                if all([bool(test[TEST_DIMENSIONS[dim]] == aggr_test[TEST_DIMENSIONS[dim]]) for dim in non_aggregation_dimensions]):
                    if "data" not in aggr_test: aggr_test["data"] = test["data"]
                    else: aggr_test["data"] = np.concatenate((aggr_test["data"], test["data"]), axis=0)
            aggregated_tests_data[i] = aggr_test
        print(f"Aggregation ended. N° trials for each test: {len(aggregated_tests_data[0]['data'])}")
        return aggregated_tests_data

def save_figure(figure:plt.Figure, title=False, name=None):
    global PLOT_COUNTER
    if NO_TITLES and not title: figure.suptitle("")
    if name is None: 
        figure.savefig(os.path.join(FIGURES_SAVING_PATH,f"{PLOT_COUNTER}.eps"), format='eps')
        PLOT_COUNTER = PLOT_COUNTER + 1
    else: figure.savefig(os.path.join(FIGURES_SAVING_PATH,f"{name}.eps"), format='eps')
    plt.close(figure)

def extract_data_from_human_times_file(results_file_list:list[str], test:str, human_times_data:dict):
    # Lengths are variable with respect to results files
    episode_times = []
    times = []
    humans = []
    indexer_list = ["_with_robot","_without_robot"]
    for results in results_file_list:
        episode_times.append(np.copy(human_times_data[results][test]["episode_times"]))
        times.append(np.array([np.copy(human_times_data[results][test]["times_to_goal" + idx]) for idx in indexer_list], dtype=np.float64))
        humans.append(np.array([np.copy(human_times_data[results][test]["n_humans_reached_goal" + idx]) for idx in indexer_list], dtype=int))
    return episode_times, times, humans

def add_labels(ax:Axis, x:list[str], y:pd.Series):
    bar_labels = []
    for i, value in y.items(): bar_labels.append(round(value, 2))
    for i, name in enumerate(x): ax.text(name, y.iloc[i]/2, bar_labels[i], ha = 'center', bbox = dict(facecolor = 'white', alpha = .5))

def plot_single_heatmap(matrix:np.array, ax, metric_name:str, anova_data:np.array):
    ax.imshow(matrix.T)
    ax.set_xlabel("Training environment")
    ax.set_ylabel("Testing environment")
    ax.set_xticks(np.arange(len(ENVIRONMENTS)), labels=ENVIRONMENTS_DISPLAY_NAME)
    ax.set_yticks(np.arange(len(ENVIRONMENTS)), labels=ENVIRONMENTS_DISPLAY_NAME)
    for i in range(len(ENVIRONMENTS)):
        for j in range(len(ENVIRONMENTS)): ax.text(i, j, matrix[i, j], ha="center", va="center", color="w", weight='bold')
    ax.set_title(metric_name + " - anova pvalue: " + str(round(anova_data[1],2)))

def plot_single_ttest_map(matrix:np.array, ax):
    # Create custom colormap
    color_values = np.array([[1.0, 1.0, 1.0, 1.0],[0.0, 1.0, 0.0, 1.0],[1.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    pvalue_colormap = ListedColormap(color_values)
    color_matrix = np.empty((len(ENVIRONMENTS)*len(ENVIRONMENTS),len(ENVIRONMENTS)*len(ENVIRONMENTS)), dtype=np.float64)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]): 
            if matrix[i,j] == 1.0: color_matrix[i,matrix.shape[1] - j - 1] = 0.1
            else: color_matrix[i,matrix.shape[1] - j - 1] = 0.4 if matrix[i,j] <= T_TEST_P_VALUE_THRESHOLD else 0.7
    # Plot
    ax.matshow(color_matrix.T, cmap=pvalue_colormap)
    ax.hlines([-0.5,2.5,5.5,8.5], -0.5, 8.5, linewidth=2, colors='black')
    ax.vlines([-0.5,2.5,5.5,8.5], -0.5, 8.5, linewidth=2, colors='black')
    ax.hlines([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5], -0.5, 8.5, linewidth=1, colors='black')
    ax.vlines([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5], -0.5, 8.5, linewidth=1, colors='black')
    ax.set(xticklabels='', yticklabels='', xlim=[-0.5,8.5], ylim=[-0.5,8.5], title=f'pvalue of t-test', xticks=[], yticks=[])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]): 
            if matrix[j,i] == 1.0: continue
            else: ax.text(i,matrix.shape[1] - j - 1, str(round(matrix[i,j],2)), va='center', ha='center')

def plot_single_test_metrics(test:str, environment:str, dataframe:pd.DataFrame, more_plots:bool):
    if not more_plots:
        figure, ax = plt.subplots(2,2, figsize=(20,10))
        figure.subplots_adjust(right=0.80)
        figure.suptitle(f"Metrics for {environment} environment - {test}")
        # Success rate
        ax[0,0].bar(POLICY_NAMES,dataframe["success_rate"], color = COLORS, label=POLICY_NAMES)
        add_labels(ax[0,0], POLICY_NAMES, dataframe["success_rate"])
        ax[0,0].set(xlabel='Policy', ylabel='Success rate', xticklabels=[], ylim=[0,1])
        # Success weighted by path length
        ax[0,1].bar(POLICY_NAMES,dataframe["SPL"], color = COLORS, label=POLICY_NAMES)
        add_labels(ax[0,1], POLICY_NAMES, dataframe["SPL"])
        ax[0,1].set(xlabel='Policy', ylabel='SPL', xticklabels=[], ylim=[0,1])
        # Space compliance
        ax[1,0].bar(POLICY_NAMES,dataframe["space_compliance"], color = COLORS, label=POLICY_NAMES)
        add_labels(ax[1,0], POLICY_NAMES, dataframe["space_compliance"])
        ax[1,0].set(xlabel='Policy', ylabel='Space compliance', xticklabels=[], ylim=[0,1])
        # Time to goal
        ax[1,1].bar(POLICY_NAMES,dataframe["time_to_goal"], color = COLORS, label=POLICY_NAMES)
        add_labels(ax[1,1], POLICY_NAMES, dataframe["time_to_goal"])
        ax[1,1].set(xlabel='Policy', ylabel='Time to goal', xticklabels=[])
        handles, labels = ax[0,0].get_legend_handles_labels()
        figure.legend(handles, labels, bbox_to_anchor=(0.90, 0.5), loc='center')
    else:
        figure, ax = plt.subplots(3,3, figsize=(20,10))
        figure.subplots_adjust(right=0.80)
        figure.suptitle(f"Metrics for {environment} environment - {test}")
        # Success rate
        ax[0,0].bar(POLICY_NAMES,dataframe["success_rate"], color = COLORS, label=POLICY_NAMES)
        ax[0,0].set(xlabel='Policy', ylabel='Success rate', xticklabels=[], ylim=[0,1])
        # Success weighted by path length
        ax[0,1].bar(POLICY_NAMES,dataframe["SPL"], color = COLORS, label=POLICY_NAMES)
        ax[0,1].set(xlabel='Policy', ylabel='SPL', xticklabels=[], ylim=[0,1])
        # Path length
        ax[0,2].bar(POLICY_NAMES,dataframe["path_length"], color = COLORS, label=POLICY_NAMES)
        ax[0,2].set(xlabel='Policy', ylabel='Path length', xticklabels=[])
        # Space compliance
        ax[1,0].bar(POLICY_NAMES,dataframe["space_compliance"], color = COLORS, label=POLICY_NAMES)
        ax[1,0].set(xlabel='Policy', ylabel='Space compliance', xticklabels=[], ylim=[0,1])
        # Time to goal
        ax[1,1].bar(POLICY_NAMES,dataframe["time_to_goal"], color = COLORS, label=POLICY_NAMES)
        ax[1,1].set(xlabel='Policy', ylabel='Time to goal', xticklabels=[])
        # Average distance to pedestrians
        ax[1,2].bar(POLICY_NAMES,dataframe["avg_dist"], color = COLORS, label=POLICY_NAMES)
        ax[1,2].set(xlabel='Policy', ylabel='Average dist. to ped.', xticklabels=[])
        # Average speed
        ax[2,0].bar(POLICY_NAMES,dataframe["avg_speed"], color = COLORS, label=POLICY_NAMES)
        ax[2,0].set(xlabel='Policy', ylabel='Average speed', xticklabels=[])
        # Average acceleration
        ax[2,1].bar(POLICY_NAMES,dataframe["avg_accel."], color = COLORS, label=POLICY_NAMES)
        ax[2,1].set(xlabel='Policy', ylabel='Average acceleration', xticklabels=[])
        # Average jerk
        ax[2,2].bar(POLICY_NAMES,dataframe["avg_jerk"], color = COLORS, label=POLICY_NAMES)
        ax[2,2].set(xlabel='Policy', ylabel='Average jerk', xticklabels=[])
        handles, labels = ax[0,0].get_legend_handles_labels()
        figure.legend(handles, labels, bbox_to_anchor=(0.90, 0.5), loc='center')
    # Save figure
    if SAVE_FIGURES: save_figure(figure)

def plot_single_test_complete_metrics(test:str, environment:str, data:np.array):
    figure, ax = plt.subplots(2,2, figsize=(20,10))
    figure.subplots_adjust(right=0.80)
    figure.suptitle(f"Metrics for {environment} environment - {test}")
    # Filter Nan values
    nan_mask = ~np.isnan(data)
    filtered_data = []
    for row, mask_row in zip(data.T, nan_mask.T): filtered_data.append([column[mask_column] for column, mask_column in zip(row.T, mask_row.T)])
    # Time to goal
    bplot1 = ax[0,0].boxplot(filtered_data[:][:][METRICS.index("time_to_goal")], showmeans=True, labels=POLICY_NAMES, patch_artist=True)
    ax[0,0].set(xlabel='Policy', ylabel='Time to goal', xticklabels=[])
    # Path length
    bplot2 = ax[0,1].boxplot(filtered_data[:][:][METRICS.index("path_length")], showmeans=True, labels=POLICY_NAMES, patch_artist=True)
    ax[0,1].set(xlabel='Policy', ylabel='Path length', xticklabels=[])
    # Space compliance
    bplot3 = ax[1,0].boxplot(filtered_data[:][:][METRICS.index("space_compliance")], showmeans=True, labels=POLICY_NAMES, patch_artist=True)
    ax[1,0].set(xlabel='Policy', ylabel='Space compliance', xticklabels=[], ylim=[0,1])
    # SPL
    bplot4 = ax[1,1].boxplot(filtered_data[:][:][METRICS.index("SPL")], showmeans=True, labels=POLICY_NAMES, patch_artist=True)
    ax[1,1].set(xlabel='Policy', ylabel='SPL', xticklabels=[], ylim=[0,1])
    # Set color of boxplots
    for bplot in (bplot1, bplot2, bplot3, bplot4):
        for patch, color in zip(bplot['boxes'], COLORS):
            patch.set_facecolor(color)
    # Legend
    figure.legend(bplot1["boxes"], POLICY_NAMES, bbox_to_anchor=(0.90, 0.5), loc='center')
    # Save figure
    if SAVE_FIGURES: save_figure(figure)
    
def plot_heatmaps(data:list[np.array], test:str, ttest_data:np.array, anova_results:np.array, metrics:list[str], only_sarl=False):
    # Data shape (test & train env combination, n_metrics, samples)
    # T-test Data shape (metric, test_env_combination, test_env_combination, 3)
    ## Initialize metrics matrices
    n_metrics = len(metrics)
    average_metrics = np.empty((n_metrics,len(ENVIRONMENTS),len(ENVIRONMENTS)), dtype=np.float64)
    ## Extract data
    for i in range(len(data)):
        for j in range(n_metrics): average_metrics[j,i//len(ENVIRONMENTS),i%len(ENVIRONMENTS)] = round(np.sum(data[i][j]) / len(data[i][j]), 2)
    ## Plot heatmaps with T-test pvalues
    figure = plt.figure(figsize=(20,10))
    if only_sarl: figure.suptitle("Average metrics for SARL robot policies - " + test)
    else: figure.suptitle("Average metrics over all trained robot policies - " + test)
    outer = GridSpec(int(n_metrics/2), int(n_metrics/2), figure=figure, wspace=0.2, hspace=0.2)
    for i in range(n_metrics): # For each metric
        inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        ax1 = figure.add_subplot(inner[0])
        ax2 = figure.add_subplot(inner[1])
        plot_single_heatmap(average_metrics[i], ax1, metrics[i], anova_results[i]) # Heatmap with average values
        plot_single_ttest_map(ttest_data[i,:,:,1], ax2) # P-value of T-tests
    # Save figure
    if SAVE_FIGURES: save_figure(figure)
        
def plot_curves(data:list[pd.DataFrame], test_env:str):
    # Extract data
    time_to_goal_data = np.empty((len(data[0].index.values),len(TESTS)), dtype=np.float64)
    tests = []
    for i, test in enumerate(data[0].index.values):
        tests.append(test)
        for j, df in enumerate(data): time_to_goal_data[i,j] = df.loc[test]["time_to_goal"]
    # Plot curves
    figure, ax = plt.subplots(1,1, figsize=(20,10))
    figure.subplots_adjust(right=0.80)
    figure.suptitle("Average time to goal over tests with increasing crowd density - Environment: " + test_env)
    ax.set(xlabel="N° humans", ylabel="Average time to Goal", xticks=np.arange(len(TESTS)), xticklabels=TESTS)
    for i, test in enumerate(tests): ax.plot(time_to_goal_data[i], label=test, color=COLORS[i % 10], linewidth=2.5)
    ax.grid()
    handles, _ = ax.get_legend_handles_labels()
    figure.legend(handles, POLICY_NAMES, bbox_to_anchor=(0.90, 0.5), loc='center')
    # Save figure
    if SAVE_FIGURES: save_figure(figure)

def plot_human_times_boxplots(test:str, environment:str, ep_times:list, hu_times:list, n_humans:list):
    # ep_times(list(np.array)) - (11,successful_trials) - first dimension is list
    # hu_times(list(np.array)) - (11,2,successful_trials,n_humans) - first dimension is list
    # n_humans(list(np.array)) - (11,2,successful_trials,n_humans) - first dimension is list
    figure = plt.figure(figsize=(20,10))
    figure.suptitle(f"Testing environment: {environment} - {test}")
    gs = GridSpec(2, 3, figure=figure)
    ax1 = figure.add_subplot(gs[0,0])
    ax2 = figure.add_subplot(gs[0,1])
    ax3 = figure.add_subplot(gs[1,0])
    ax4 = figure.add_subplot(gs[1,1])
    ax5 = figure.add_subplot(gs[:,2])
    figure.subplots_adjust(right=0.80)
    # Compute average time to goal among humans who reached the goal for each trial - Reduce shape from (11,2,successful_trials,n_humans) to (11,2,successful_trials)
    avg_hu_times = []
    for i, results in enumerate(hu_times):
        avg_hu_times_results = np.empty((len(results),len(results[0])), dtype=np.float64)
        for j, test in enumerate(results): # With or without robot
            for k, human_times in enumerate(test):
                if n_humans[i][j,k] == 0: avg_hu_times_results[j,k] = np.NaN
                else: avg_hu_times_results[j,k] = np.nansum(human_times) / n_humans[i][j,k]
        avg_hu_times.append(avg_hu_times_results)
    # Compute min and max avg_hu_time to fix the scale in the graphs
    min_value = np.nanmin(np.array([np.nanmin(avg_hu_times_results, axis=(0,1)) for avg_hu_times_results in avg_hu_times]))
    max_value = np.nanmax(np.array([np.nanmax(avg_hu_times_results, axis=(0,1)) for avg_hu_times_results in avg_hu_times]))
    # Divide data with robot and without robot - Reduce shape from (11,2,successful_trials) to (11,successful_trials)
    avg_hu_times_w_robot = [np.copy(avg_hu_times_results[0,:]) for avg_hu_times_results in avg_hu_times]
    avg_hu_times_wout_robot = [np.copy(avg_hu_times_results[1,:]) for avg_hu_times_results in avg_hu_times]
    n_humans_w_robot = [np.copy(n_humans_results[0,:]) for n_humans_results in n_humans]
    n_humans_wout_robot = [np.copy(n_humans_results[1,:]) for n_humans_results in n_humans]
    # Filter NaNs from avg_hu_times (otherwise box plot does not work) - avg_hu_times_w_robot.shape and avg_hu_times_wout_robot.shape = (11,successful_trials), NaNs are in the last dimension
    filtered_avg_hu_times_w_robot = [avg_hu_times_w_robot_results[~np.isnan(avg_hu_times_w_robot_results)] for avg_hu_times_w_robot_results in avg_hu_times_w_robot]
    filtered_avg_hu_times_wout_robot = [avg_hu_times_wout_robot_results[~np.isnan(avg_hu_times_wout_robot_results)] for avg_hu_times_wout_robot_results in avg_hu_times_wout_robot]
    # Time to goal with robot
    bplot1 = ax1.boxplot(filtered_avg_hu_times_w_robot, showmeans=True, labels=POLICY_NAMES, patch_artist=True)
    ax1.set(xlabel='Robot policy', ylabel="Average humans' Time to goal", xticklabels=[], title="With robot", ylim=[min_value,max_value])
    ax1.grid()
    # Time to goal without robot
    bplot2 = ax2.boxplot(filtered_avg_hu_times_wout_robot, showmeans=True, labels=POLICY_NAMES, patch_artist=True)
    ax2.set(xlabel='Set of trials with same duration of the ones with robot', ylabel="Average humans' Time to goal", xticklabels=[], title="Without robot", ylim=[min_value,max_value])
    ax2.grid()
    # N° humans that reached the goal with robot
    bplot3 = ax3.boxplot(n_humans_w_robot, showmeans=True, labels=POLICY_NAMES, patch_artist=True)
    ax3.set(xlabel='Robot policy', ylabel="N° of successful humans", xticklabels=[])
    ax3.grid()
    # N° humans that reached the goal without robot
    bplot4 = ax4.boxplot(n_humans_wout_robot, showmeans=True, labels=POLICY_NAMES, patch_artist=True)
    ax4.set(xlabel='Set of trials with same duration of the ones with robot', ylabel="N° of successful humans", xticklabels=[])
    ax4.grid()
    # Episode times 
    bplot5 = ax5.boxplot(ep_times, showmeans=True, labels=POLICY_NAMES, patch_artist=True)
    ax5.set(xlabel='Robot Policy', ylabel="Duration of the trials", xticklabels=[], title="Duration of the trials (seconds)")
    ax5.grid()
    # Set color of boxplots
    for bplot in (bplot1, bplot2, bplot3, bplot4, bplot5):
        for patch, color in zip(bplot['boxes'], COLORS):
            patch.set_facecolor(color)
    # Legend
    figure.legend(bplot1["boxes"], POLICY_NAMES, bbox_to_anchor=(0.90, 0.5), loc='center')
    # Save figure
    if SAVE_FIGURES: save_figure(figure)

def plot_boxplots_for_enviroments(test:str, data:list[np.array], metrics:list[str], train=True, only_sarl=False):
    figure, ax = plt.subplots(2,2, figsize=(20,10))
    # figure.tight_layout()
    figure.subplots_adjust(right=0.80)
    if only_sarl: figure.suptitle("Metrics for SARL robot policies - " + test)
    else: figure.suptitle("Metrics over all trained robot policies - " + test)
    if train: x_ax_label = "Training environment"
    else: x_ax_label = "Testing environment"
    if train: legend_title = "Training environment"
    else: legend_title = "Testing environment"
    ## Prepare data
    time_to_goal = [data[e][0][:] for e in range(len(ENVIRONMENTS))]
    path_length = [data[e][1][:] for e in range(len(ENVIRONMENTS))]
    space_compliance = [data[e][2][:] for e in range(len(ENVIRONMENTS))]
    spl = [data[e][3][:] for e in range(len(ENVIRONMENTS))]
    # Time to goal
    bplot1 = ax[0,0].boxplot(time_to_goal, showmeans=True, labels=ENVIRONMENTS, patch_artist=True)
    ax[0,0].set(xlabel=x_ax_label, ylabel='Time to goal', xticklabels=[], ylim = [10,50])
    ax[0,0].grid()
    # Path length
    bplot2 = ax[0,1].boxplot(path_length, showmeans=True, labels=ENVIRONMENTS, patch_artist=True)
    ax[0,1].set(xlabel=x_ax_label, ylabel='Path length', xticklabels=[], ylim = [10,50])
    ax[0,1].grid()
    # Space compliance
    bplot3 = ax[1,0].boxplot(space_compliance, showmeans=True, labels=ENVIRONMENTS, patch_artist=True)
    ax[1,0].set(xlabel=x_ax_label, ylabel='Space compliance', xticklabels=[], ylim=[0,1])
    ax[1,0].grid()
    # SPL
    bplot4 = ax[1,1].boxplot(spl, showmeans=True, labels=ENVIRONMENTS, patch_artist=True)
    ax[1,1].set(xlabel=x_ax_label, ylabel='SPL', xticklabels=[], ylim=[0,1])
    ax[1,1].grid()
    # Set color of boxplots
    for bplot in (bplot1, bplot2, bplot3, bplot4):
        for patch, color in zip(bplot['boxes'], COLORS):
            patch.set_facecolor(color)
    # Legend
    figure.legend(bplot1["boxes"], ENVIRONMENTS, bbox_to_anchor=(0.90, 0.5), loc='center', title=legend_title)
    # Save figure
    if SAVE_FIGURES: save_figure(figure)

def plot_space_compliance_over_spl_boxplots(test:str, data:list[list[np.ndarray]]):
    # Data is in the form: (trained_policy, metrics, samples) - (9, 2, ~300) - list[list[np.ndarray]]
    figure, ax = plt.subplots(1,1, figsize=(20,10))
    figure.subplots_adjust(right=0.80)
    figure.suptitle("Space compliance over SPL (averaged over all test environments) - " + test)
    # Prepare data
    space_compliance = [data[e][0][:] for e in range(len(TRAINED_POLICIES))]
    spl = [np.mean(data[e][1][:]) for e in range(len(TRAINED_POLICIES))]
    # Plot boxplots
    bplot = ax.boxplot(space_compliance, showmeans=True, labels=TRAINED_POLICIES, patch_artist=True, positions=spl, widths=[0.02 for _ in range(len(data))])
    ax.set(xlabel='Success weighted by path length (SPL)', ylabel='Space compliance', ylim = [0,1], xlim = [0,1], xticks=[(i+1)/10 for i in range(10)], xticklabels=[(i+1)/10 for i in range(10)])
    ax.grid()
    # Set color of boxplots
    for patch, color in zip(bplot['boxes'], COLORS): patch.set_facecolor(color)
    # Legend
    figure.legend(bplot["boxes"], TRAINED_POLICIES, bbox_to_anchor=(0.90, 0.5), loc='center', title="Training environment")
    # Save figure
    if SAVE_FIGURES: save_figure(figure)

def plot_curves_over_n_humans_tests(data:list[list[list[np.ndarray]]]):
    # Data is in the form: (n_humans_test, trained_policy, metrics, samples) - (4, 3, 4, ~300) - list[list[list[np.ndarray]]]
    figure, ax = plt.subplots(2,2, figsize=(20,10))
    figure.subplots_adjust(right=0.80)
    figure.suptitle("Metrics of SARL policies over tests with increasing number of humans (averaged over all test environments)")
    for a in ax: a[0].set_xticks([i for i in range(len(TESTS))]); a[0].set_xticklabels(TESTS_DISPLAY_NAME); a[1].set_xticks([i for i in range(len(TESTS))]); a[1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,0].set(ylabel="Time to goal")
    ax[1,0].set(ylabel="Space compliance", ylim=[0,1], xlabel="Number of humans")
    ax[0,1].set(ylabel="SPL", ylim=[0,1], xlabel="Number of humans")
    ax[1,1].set(ylabel="Collisions", ylim=[0,100])
    # Prepare data
    mean_time_to_goal = np.zeros((len(data),len(data[0]))) # (n_humans_test, trained_policy)
    mean_space_compliance = np.zeros((len(data),len(data[0]))) # (n_humans_test, trained_policy)
    mean_spl = np.zeros((len(data),len(data[0]))) # (n_humans_test, trained_policy)
    mean_collisions = np.zeros((len(data),len(data[0]))) # (n_humans_test, trained_policy)
    for i in range(len(TESTS)):
        for j in range(len(SARL_POLICIES)):
            mean_time_to_goal[i,j] = np.mean(data[i][j][0][:])
            mean_space_compliance[i,j] = np.mean(data[i][j][1][:])
            mean_spl[i,j] = np.mean(data[i][j][2][:])
            mean_collisions[i,j] = np.mean(data[i][j][3][:])
    # Plot 
    for i in range(len(SARL_POLICIES)):
        ax[0,0].plot(mean_time_to_goal[:,i], label=SARL_POLICIES[i], color=COLORS[(i+6) % 10], linewidth=2.5)
        ax[0,1].plot(mean_spl[:,i], label=SARL_POLICIES[i], color=COLORS[(i+6) % 10], linewidth=2.5)
        ax[1,0].plot(mean_space_compliance[:,i], label=SARL_POLICIES[i], color=COLORS[(i+6) % 10], linewidth=2.5)
        ax[1,1].plot(mean_collisions[:,i], label=SARL_POLICIES[i], color=COLORS[(i+6) % 10], linewidth=2.5)
    handles, _ = ax[0,0].get_legend_handles_labels()
    figure.legend(handles, SARL_POLICIES, bbox_to_anchor=(0.90, 0.5), loc='center')
    # Save figure
    if SAVE_FIGURES: save_figure(figure)

metrics_dir = os.path.join(os.path.dirname(__file__),'tests','metrics')
file_name = os.path.join(metrics_dir,METRICS_FILE)
# Complete data is in the following shape (test, n_humans_test, trials, metrics)
if os.path.exists(os.path.join(metrics_dir,COMPLETE_METRICS_FILE)):
    with open(os.path.join(metrics_dir,COMPLETE_METRICS_FILE), "rb") as f: complete_data = pickle.load(f)
else: print("Complete metrics file not found. Proceding without it...")
for k, test in enumerate(TESTS):
    ## Load average metrics dataframe
    if os.path.exists(file_name): dataframe = pd.read_excel(file_name, sheet_name=test, index_col=0)
    else: print("Average metrics file not found. Proceding without it...")
    if BOX_PLOTS:
        # Find numerical indexes of testing environments in the dataframe
        indexes = {i: dataframe.index.get_loc(i) for i, row in dataframe.iterrows()}
        # Complete data has dimensions (n_results_files, n_humans_tests, n_trials, n_metrics)
        for i, environment in enumerate(ENVIRONMENTS):
            if i == 0: env_indexes = [indexes[a] for a in TESTED_ON_ORCA]
            elif i == 1: env_indexes = [indexes[a] for a in TESTED_ON_SFM_GUO]
            else: env_indexes = [indexes[a] for a in TESTED_ON_HSFM_NEW_GUO]
            # Extracting data
            data = complete_data[env_indexes,k]
            # Plotting
            plot_single_test_complete_metrics(test, environment, data)
    if BAR_PLOTS:
        for i, environment in enumerate(ENVIRONMENTS):
            # Extracting data
            if i == 0: df_env = dataframe.loc[TESTED_ON_ORCA, :]
            elif i == 1: df_env = dataframe.loc[TESTED_ON_SFM_GUO, :]
            else: df_env = dataframe.loc[TESTED_ON_HSFM_NEW_GUO, :]
            # Plotting
            plot_single_test_metrics(test, environment, df_env, False)  
    if MORE_BAR_PLOTS:
        for i, environment in enumerate(ENVIRONMENTS):
            # Extracting data
            if i == 0: df_env = dataframe.loc[TESTED_ON_ORCA, :]
            elif i == 1: df_env = dataframe.loc[TESTED_ON_SFM_GUO, :]
            else: df_env = dataframe.loc[TESTED_ON_HSFM_NEW_GUO, :]
            # Plotting
            plot_single_test_metrics(test, environment, df_env, True) 
    if HEAT_MAP:
        ## Extracting data
        # Find numerical indexes of testing environments in the dataframe
        indexes = {i: dataframe.index.get_loc(i) for i, row in dataframe.iterrows()}
        metrics_names = ["time_to_goal","path_length","space_compliance","SPL"]
        metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
        # Complete data has dimensions (n_results_files, n_humans_tests, n_trials, n_metrics)
        train_tests = [TRAINED_ON_ORCA, TRAINED_ON_SFM_GUO, TRAINED_ON_HSFM_NEW_GUO]
        test_tests = [TESTED_ON_ORCA, TESTED_ON_SFM_GUO, TESTED_ON_HSFM_NEW_GUO]
        data = [] # (train_env & test_env, metrics, non-nan realizations)
        for i in range(len(train_tests)):
            for j in range(len(test_tests)):
                test_set = list(set(train_tests[i]) & set(test_tests[j]))
                env_indexes = [indexes[a] for a in test_set]
                # Extracting data
                one_data = complete_data[env_indexes,k]
                ij_data = []
                for m, metric in enumerate(metrics_idxs): 
                    not_filtered_data = np.reshape(np.array([one_data[env,:,metric] for env in range(len(one_data))], dtype=np.float64),(300,))
                    ij_data.append(not_filtered_data[~np.isnan(not_filtered_data)])
                data.append(ij_data)
        # if k == 0: 
        #     for i, d in enumerate(data): print(f"Train env: {ENVIRONMENTS[i//len(ENVIRONMENTS)]} - Test env: {ENVIRONMENTS[i%len(ENVIRONMENTS)]} - Average time to goal: {round(np.sum(d[0]) / len(d[0]),2)}")
        ## Anova tests
        anova_results = np.empty((len(metrics_names),2), dtype = np.float64)
        for m, metric in enumerate(metrics_names): anova_results[m] = f_oneway(*[data[i][m] for i in range(len(data))])
        ## T-test
        ttest_data = np.empty((len(metrics_idxs),len(ENVIRONMENTS)*len(ENVIRONMENTS),len(ENVIRONMENTS)*len(ENVIRONMENTS),3), dtype=np.float64) # (metric, test_env_combination, test_env_combination, 3)
        for r in range(len(ENVIRONMENTS)):
            for c in range(len(ENVIRONMENTS)):
                for i in range(len(data)):
                        for m in range(len(metrics_idxs)):
                            ttest = ttest_ind(data[(r*len(ENVIRONMENTS)) + c][m][:],data[i][m][:])
                            # if k == 0 and m == 0: print(f"Time to goal T-test {ENVIRONMENTS[r]} - {ENVIRONMENTS[c]} VS {ENVIRONMENTS[i//len(ENVIRONMENTS)]} - {ENVIRONMENTS[i%len(ENVIRONMENTS)]}: {ttest.pvalue}")
                            ttest_data[m,(r*len(ENVIRONMENTS)) + i//len(ENVIRONMENTS), (c*len(ENVIRONMENTS)) + i%len(ENVIRONMENTS)] = np.array([ttest.statistic, ttest.pvalue, ttest.df], dtype=np.float64)
        ## Heatmaps
        plot_heatmaps(data, test, ttest_data, anova_results, metrics_names)
        # Heatmap for average above all n_humans tests
        if k == 0: average_data = data.copy()
        else: 
            for i, combination in enumerate(average_data): average_data[i] = [np.append(combination[metric], data[i][metric], axis = 0) for metric in range(len(combination))]
            if k == len(TESTS) - 1: 
                # Anova
                ## Anova tests
                anova_results = np.empty((len(metrics_names),2), dtype = np.float64)
                for m, metric in enumerate(metrics_names): anova_results[m] = f_oneway(*[average_data[i][m] for i in range(len(average_data))])
                # T-test
                ttest_average_data = np.empty((len(metrics_idxs),len(ENVIRONMENTS)*len(ENVIRONMENTS),len(ENVIRONMENTS)*len(ENVIRONMENTS),3), dtype=np.float64) # (metric, test_env_combination, test_env_combination, 3)
                for r in range(len(ENVIRONMENTS)):
                    for c in range(len(ENVIRONMENTS)):
                        for i in range(len(data)):
                                for m in range(len(metrics_idxs)):
                                    ttest = ttest_ind(average_data[(r*len(ENVIRONMENTS)) + c][m][:],average_data[i][m][:])
                                    ttest_average_data[m,(r*len(ENVIRONMENTS)) + i//len(ENVIRONMENTS), (c*len(ENVIRONMENTS)) + i%len(ENVIRONMENTS)] = np.array([ttest.statistic, ttest.pvalue, ttest.df], dtype=np.float64)
                plot_heatmaps(average_data, "Average of all tests", ttest_average_data, anova_results, metrics_names, only_sarl=False)
    if SARL_ONLY_HEATMAPS:
        ## Extracting data
        # Find numerical indexes of testing environments in the dataframe
        indexes = {i: dataframe.index.get_loc(i) for i, row in dataframe.iterrows()}
        metrics_names = ["time_to_goal","path_length","space_compliance","SPL"]
        metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
        # Complete data has dimensions (n_results_files, n_humans_tests, n_trials, n_metrics)
        data = [] # (train_env & test_env, metrics, non-nan realizations)
        for i, file in enumerate(SARL_POLICIES_RESULTS):
            env_indexes = indexes[file]
            # Extracting data
            one_data = complete_data[env_indexes,k]
            ij_data = []
            for m, metric in enumerate(metrics_idxs): 
                not_filtered_data = np.reshape(np.array(one_data[:,metric], dtype=np.float64),(100,))
                ij_data.append(not_filtered_data[~np.isnan(not_filtered_data)])
            data.append(ij_data)
        ## Anova tests
        anova_results = np.empty((len(metrics_names),2), dtype = np.float64)
        for m, metric in enumerate(metrics_names): anova_results[m] = f_oneway(*[data[i][m] for i in range(len(data))])
        ## T-test
        ttest_data = np.empty((len(metrics_idxs),len(ENVIRONMENTS)*len(ENVIRONMENTS),len(ENVIRONMENTS)*len(ENVIRONMENTS),3), dtype=np.float64) # (metric, test_env_combination, test_env_combination, 3)
        for r in range(len(ENVIRONMENTS)):
            for c in range(len(ENVIRONMENTS)):
                for i in range(len(data)):
                        for m in range(len(metrics_idxs)):
                            ttest = ttest_ind(data[(r*len(ENVIRONMENTS)) + c][m][:],data[i][m][:])
                            # if k == 0 and m == 0: print(f"Time to goal T-test {ENVIRONMENTS[r]} - {ENVIRONMENTS[c]} VS {ENVIRONMENTS[i//len(ENVIRONMENTS)]} - {ENVIRONMENTS[i%len(ENVIRONMENTS)]}: {ttest.pvalue}")
                            ttest_data[m,(r*len(ENVIRONMENTS)) + i//len(ENVIRONMENTS), (c*len(ENVIRONMENTS)) + i%len(ENVIRONMENTS)] = np.array([ttest.statistic, ttest.pvalue, ttest.df], dtype=np.float64)
        ## Heatmaps
        plot_heatmaps(data, test, ttest_data, anova_results, metrics_names, only_sarl=True)
        # Heatmap for average above all n_humans tests
        if k == 0: average_data = data.copy()
        else: 
            for i, combination in enumerate(average_data): average_data[i] = [np.append(combination[metric], data[i][metric], axis = 0) for metric in range(len(combination))]
            if k == len(TESTS) - 1: 
                ## Anova tests
                anova_results = np.empty((len(metrics_names),2), dtype = np.float64)
                for m, metric in enumerate(metrics_names): anova_results[m] = f_oneway(*[average_data[i][m] for i in range(len(average_data))])
                # T-test
                ttest_average_data = np.empty((len(metrics_idxs),len(ENVIRONMENTS)*len(ENVIRONMENTS),len(ENVIRONMENTS)*len(ENVIRONMENTS),3), dtype=np.float64) # (metric, test_env_combination, test_env_combination, 3)
                for r in range(len(ENVIRONMENTS)):
                    for c in range(len(ENVIRONMENTS)):
                        for i in range(len(data)):
                                for m in range(len(metrics_idxs)):
                                    ttest = ttest_ind(average_data[(r*len(ENVIRONMENTS)) + c][m][:],average_data[i][m][:])
                                    ttest_average_data[m,(r*len(ENVIRONMENTS)) + i//len(ENVIRONMENTS), (c*len(ENVIRONMENTS)) + i%len(ENVIRONMENTS)] = np.array([ttest.statistic, ttest.pvalue, ttest.df], dtype=np.float64)
                plot_heatmaps(average_data, "Average of all tests", ttest_average_data, anova_results, metrics_names, only_sarl=True)
    if SARL_ONLY_BOXPLOTS:
        ## Extracting data
        # Find numerical indexes of testing environments in the dataframe
        indexes = {i: dataframe.index.get_loc(i) for i, row in dataframe.iterrows()}
        metrics_names = ["time_to_goal","path_length","space_compliance","SPL"]
        metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
        # Complete data has dimensions (n_results_files, n_humans_tests, n_trials, n_metrics)
        train_data = [] # (train_env, metrics, non-nan realizations)
        test_data = [] # (test_env, metrics, non-nan realizations)
        train_tests = [TRAINED_ON_ORCA, TRAINED_ON_SFM_GUO, TRAINED_ON_HSFM_NEW_GUO]
        test_tests = [TESTED_ON_ORCA, TESTED_ON_SFM_GUO, TESTED_ON_HSFM_NEW_GUO]
        for i, env in enumerate(ENVIRONMENTS):
            train_set_env = list(set(SARL_POLICIES_RESULTS) & set(train_tests[i])) # This gives us all tests where SARL trained with env was used
            test_set_env = list(set(SARL_POLICIES_RESULTS) & set(test_tests[i])) # This gives us all tests where SARL tested with env was used
            train_env_indexes = [indexes[a] for a in train_set_env]
            test_env_indexes = [indexes[a] for a in test_set_env]
            # Extracting data
            train_one_data = complete_data[train_env_indexes,k]
            test_one_data = complete_data[test_env_indexes,k]
            train_ij_data = []
            test_ij_data = []
            for m, metric in enumerate(metrics_idxs): 
                # Train
                not_filtered_data = np.reshape(np.array([train_one_data[env,:,metric] for env in range(len(train_one_data))], dtype=np.float64),(300,))
                train_ij_data.append(not_filtered_data[~np.isnan(not_filtered_data)])
                # Test
                not_filtered_data = np.reshape(np.array([test_one_data[env,:,metric] for env in range(len(test_one_data))], dtype=np.float64),(300,))
                test_ij_data.append(not_filtered_data[~np.isnan(not_filtered_data)])
            train_data.append(train_ij_data)
            test_data.append(test_ij_data)
        # Plot boxplots with extracted data
        plot_boxplots_for_enviroments(test, train_data, metrics_names, only_sarl=True)
        plot_boxplots_for_enviroments(test, test_data, metrics_names, train=False, only_sarl=True)
        if k == 0: 
            average_train_data = train_data.copy()
            average_test_data = test_data.copy()
        else: 
            for i, combination in enumerate(average_train_data): average_train_data[i] = [np.append(combination[metric], train_data[i][metric], axis = 0) for metric in range(len(combination))]
            for i, combination in enumerate(average_test_data): average_test_data[i] = [np.append(combination[metric], test_data[i][metric], axis = 0) for metric in range(len(combination))]
            if k == len(TESTS) - 1: 
                plot_boxplots_for_enviroments("Average of all tests", average_train_data, metrics_names, only_sarl=True)
                plot_boxplots_for_enviroments("Average of all tests", average_test_data, metrics_names, train=False, only_sarl=True)
    if CURVE_PLOTS:
        if k == 0: all_test_on_orca_dataframes = []; all_test_on_sfm_dataframes = []; all_test_on_hsfm_dataframes = []
        all_test_on_orca_dataframes.append(dataframe.loc[TESTED_ON_ORCA, :])
        all_test_on_sfm_dataframes.append(dataframe.loc[TESTED_ON_SFM_GUO, :])
        all_test_on_hsfm_dataframes.append(dataframe.loc[TESTED_ON_HSFM_NEW_GUO, :])
        if k == len(TESTS) - 1: 
            plot_curves(all_test_on_orca_dataframes, ENVIRONMENTS[0])
            plot_curves(all_test_on_sfm_dataframes, ENVIRONMENTS[1])
            plot_curves(all_test_on_hsfm_dataframes, ENVIRONMENTS[2])
    if HUMAN_TIMES_BOX_PLOTS:
        # Load human times data
        with open(os.path.join(metrics_dir,HUMAN_TIMES_FILE), "rb") as f: human_times = pickle.load(f)
        # Complete data has dimensions (n_results_files, n_humans_tests, 5, n_trials)
        for i, environment in enumerate(ENVIRONMENTS):
            if i == 0: episode_times, times, humans = extract_data_from_human_times_file(TESTED_ON_ORCA, test, human_times)
            elif i == 1: episode_times, times, humans = extract_data_from_human_times_file(TESTED_ON_SFM_GUO, test, human_times)
            else: episode_times, times, humans = extract_data_from_human_times_file(TESTED_ON_HSFM_NEW_GUO, test, human_times)
            # Plotting
            plot_human_times_boxplots(test, environment, episode_times, times, humans)
    if SPACE_COMPLIANCE_OVER_SPL:
        ## Extracting data
        # Find numerical indexes of testing environments in the dataframe
        indexes = {i: dataframe.index.get_loc(i) for i, row in dataframe.iterrows()}
        metrics_names = ["space_compliance","SPL"]
        metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
        data = [] # (train_env & test_env, metrics, non-nan realizations)
        for i, file in enumerate(TRAINED_POLICIES_TESTS):
            env_indexes = indexes[file]
            # Extracting data
            one_data = complete_data[env_indexes,k]
            ij_data = []
            for m, metric in enumerate(metrics_idxs): 
                not_filtered_data = np.reshape(np.array(one_data[:,metric], dtype=np.float64),(100,))
                ij_data.append(not_filtered_data[~np.isnan(not_filtered_data)])
            data.append(ij_data)
        # We want to stack tests with same policy but different test environment
        stacked_data = [] # (train_env, metrics, non-nan realizations)
        for ii, trained_policy in enumerate(TRAINED_POLICIES):
            first_stacking = False
            for jj, file in enumerate(TRAINED_POLICIES_TESTS):
                if trained_policy in file:
                    data_to_stack = data[jj]
                    if not first_stacking: 
                        stacked_data.append(data_to_stack)
                        first_stacking = True
                    else:
                        for mm, metric in enumerate(metrics_names): stacked_data[ii][mm] = np.concatenate([stacked_data[ii][mm],data_to_stack[mm]])
        ## Boxplot
        plot_space_compliance_over_spl_boxplots(test, stacked_data)
    if SARL_ONLY_METRICS_OVER_N_HUMANS_TESTS:
        ## Extracting data
        # Find numerical indexes of testing environments in the dataframe
        indexes = {i: dataframe.index.get_loc(i) for i, row in dataframe.iterrows()}
        metrics_names = ["time_to_goal","space_compliance","SPL","collisions"]
        metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
        data = [] # (train_env & test_env, metrics, non-nan realizations)
        for i, file in enumerate(SARL_POLICIES_RESULTS):
            env_indexes = indexes[file]
            # Extracting data
            one_data = complete_data[env_indexes,k]
            ij_data = []
            for m, metric in enumerate(metrics_idxs): 
                not_filtered_data = np.reshape(np.array(one_data[:,metric], dtype=np.float64),(100,))
                ij_data.append(not_filtered_data[~np.isnan(not_filtered_data)])
            data.append(ij_data)
        # We want to stack tests with same policy but different test environment
        stacked_data = [] # (train_env, metrics, non-nan realizations)
        for ii, trained_policy in enumerate(SARL_POLICIES):
            first_stacking = False
            for jj, file in enumerate(SARL_POLICIES_RESULTS):
                if trained_policy in file:
                    data_to_stack = data[jj]
                    if not first_stacking: 
                        stacked_data.append(data_to_stack)
                        first_stacking = True
                    else:
                        for mm, metric in enumerate(metrics_names): stacked_data[ii][mm] = np.concatenate([stacked_data[ii][mm],data_to_stack[mm]])
        if k == 0: all_data = [stacked_data.copy()]
        else: all_data.append(stacked_data)
        if k == len(TESTS) - 1: plot_curves_over_n_humans_tests(all_data)
if METRICS_OVER_DIFFERENT_TRAINING_ENVIRONMENT:
    # Extract and aggregate data
    dataa = aggregate_data(COMPLETE_METRICS_FILE_NAMES, metrics_dir, [0,2,3,4], only_sarl=True)
    metrics_names = ["success_rate","time_to_goal","space_compliance","SPL","avg_speed","avg_accel.","avg_jerk","min_dist"]
    metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
    # Compute final data to plot
    data_to_plot = np.zeros((len(ENVIRONMENTS),len(TESTS),len(metrics_idxs)), np.float64)
    for train_env in ENVIRONMENTS:
        for n_humans in TESTS:
            for k, d in dataa.items():
                if (d["train_env"] == train_env) and (d["n_humans"] == n_humans):
                    for m, metric in enumerate(metrics_idxs): data_to_plot[ENVIRONMENTS.index(train_env),TESTS.index(n_humans),m] = np.mean(d["data"][:,metric][~np.isnan(d["data"][:,metric])])
    ## Plot
    figure, ax = plt.subplots(4,2, figsize=(18,18))
    figure.subplots_adjust(right=0.90, top=0.985, bottom=0.05, left=0.08, hspace=0.3)
    figure.suptitle("Metrics over SARL tests with increasing number of humans (averaged over all test environments and scenarios, train scenario and robot policies)")
    # success_rate
    ax[0,0].set_xticks([i for i in range(len(TESTS))])
    ax[0,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,0].set_yticks([i/10 for i in range(11)])
    ax[0,0].set(ylabel="Success rate", ylim=[0,1], xlabel="Number of humans")
    ax[0,0].grid()
    for i in range(len(ENVIRONMENTS)): ax[0,0].plot(data_to_plot[i,:,0], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i%10+3], linewidth=2.5)
    # time_to_goal
    ax[0,1].set_xticks([i for i in range(len(TESTS))])
    ax[0,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,1].set(ylabel="Time to goal ($s$)", xlabel="Number of humans")
    ax[0,1].grid()
    for i in range(len(ENVIRONMENTS)): ax[0,1].plot(data_to_plot[i,:,1], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i%10+3], linewidth=2.5)
    # space_compliance
    ax[2,0].set_xticks([i for i in range(len(TESTS))])
    ax[2,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,0].set_yticks([i/10 for i in range(11)])
    ax[2,0].set(ylabel="Space compliance", ylim=[0,1], xlabel="Number of humans")
    ax[2,0].grid()
    for i in range(len(ENVIRONMENTS)): ax[2,0].plot(data_to_plot[i,:,2], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i%10+3], linewidth=2.5)
    # SPL
    ax[1,1].set_xticks([i for i in range(len(TESTS))])
    ax[1,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,1].set_yticks([i/10 for i in range(11)])
    ax[1,1].set(ylabel="SPL", ylim=[0,1], xlabel="Number of humans")
    ax[1,1].grid()
    for i in range(len(ENVIRONMENTS)): ax[1,1].plot(data_to_plot[i,:,3], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i%10+3], linewidth=2.5)
    # speed
    ax[1,0].set_xticks([i for i in range(len(TESTS))])
    ax[1,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,0].set(ylabel="Speed ($m/s$)", xlabel="Number of humans")
    ax[1,0].grid()
    for i in range(len(ENVIRONMENTS)): ax[1,0].plot(data_to_plot[i,:,4], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i%10+3], linewidth=2.5)
    # acceleration
    ax[2,1].set_xticks([i for i in range(len(TESTS))])
    ax[2,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,1].set(ylabel="Acceleration ($m/s^2$)", xlabel="Number of humans")
    ax[2,1].grid()
    for i in range(len(ENVIRONMENTS)): ax[2,1].plot(data_to_plot[i,:,5], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i%10+3], linewidth=2.5)
    # jerk
    ax[3,0].set_xticks([i for i in range(len(TESTS))])
    ax[3,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,0].set(ylabel="Jerk ($m/s^3$)", xlabel="Number of humans")
    ax[3,0].grid()
    for i in range(len(ENVIRONMENTS)): ax[3,0].plot(data_to_plot[i,:,6], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i%10+3], linewidth=2.5)
    # min dist to humans
    ax[3,1].set_xticks([i for i in range(len(TESTS))])
    ax[3,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,1].set(ylabel="Min. dist. to humans ($m$)", xlabel="Number of humans")
    ax[3,1].grid()
    for i in range(len(ENVIRONMENTS)): ax[3,1].plot(data_to_plot[i,:,7], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i%10+3], linewidth=2.5)
    # legend
    handles, _ = ax[0,0].get_legend_handles_labels()
    figure.legend(handles, ENVIRONMENTS_DISPLAY_NAME, bbox_to_anchor=(0.95, 0.5), loc='center', title="Training \nenviron.")
    # Save figure
    if SAVE_FIGURES: save_figure(figure)
if METRICS_BOXPLOTS_OVER_DIFFERENT_TRAINING_ENVS:
    # Extract and aggregate data
    dataa = aggregate_data(COMPLETE_METRICS_FILE_NAMES, metrics_dir, [0,2,3,4,5], only_sarl=True)
    metrics_names = ["path_length","time_to_goal","space_compliance","SPL","avg_speed","avg_accel.","avg_jerk","min_dist"]
    metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
    # Compute final data to plot
    data_to_plot = []
    for m, metric in enumerate(metrics_idxs):
        env_data = []
        for train_env in ENVIRONMENTS:
            for k, d in dataa.items():
                if (d["train_env"] == train_env): env_data.append(d["data"][:,metric][~np.isnan(d["data"][:,metric])])
        data_to_plot.append(env_data)
    ## Plot
    figure, ax = plt.subplots(4,2, figsize=(18,18))
    figure.subplots_adjust(right=0.98, top=0.985, bottom=0.03, left=0.05)
    figure.suptitle("Metrics over SARL tests (averaged over all test environments and scenarios, train scenario, humans density, and robot policies)")
    # success_rate
    bplot1 = ax[0,0].boxplot(data_to_plot[0], showmeans=True, patch_artist=True, showfliers=False)
    ax[0,0].set(xlabel="Training environment", ylabel='Path length ($m$)', xticklabels=ENVIRONMENTS_DISPLAY_NAME)
    ax[0,0].grid()
    # time to goal
    bplot2 = ax[0,1].boxplot(data_to_plot[1], showmeans=True, patch_artist=True, showfliers=False)
    ax[0,1].set(xlabel="Training environment", ylabel='Time to goal ($s$)', xticklabels=ENVIRONMENTS_DISPLAY_NAME)
    ax[0,1].grid()
    # space compliance
    bplot3 = ax[2,0].boxplot(data_to_plot[2], showmeans=True, patch_artist=True, showfliers=False)
    ax[2,0].set(xlabel="Training environment", ylabel='Space compliance', xticklabels=ENVIRONMENTS_DISPLAY_NAME, ylim=[0,1])
    ax[2,0].set_yticks([i/10 for i in range(11)])
    ax[2,0].grid()
    # SPL
    bplot4 = ax[1,1].boxplot(data_to_plot[3], showmeans=True, patch_artist=True, showfliers=False)
    ax[1,1].set(xlabel="Training environment", ylabel='SPL', xticklabels=ENVIRONMENTS_DISPLAY_NAME, ylim=[0,1])
    ax[1,1].set_yticks([i/10 for i in range(11)])
    ax[1,1].grid()
    # speed
    bplot5 = ax[1,0].boxplot(data_to_plot[4], showmeans=True, patch_artist=True, showfliers=False)
    ax[1,0].set(xlabel="Training environment", ylabel='Speed ($m/s$)', xticklabels=ENVIRONMENTS_DISPLAY_NAME)
    ax[1,0].grid()
    # acceleration
    bplot6 = ax[2,1].boxplot(data_to_plot[5], showmeans=True, patch_artist=True, showfliers=False)
    ax[2,1].set(xlabel="Training environment", ylabel='Acceleration ($m/s^2$)', xticklabels=ENVIRONMENTS_DISPLAY_NAME)
    ax[2,1].grid()
    # jerk
    bplot7 = ax[3,0].boxplot(data_to_plot[6], showmeans=True, patch_artist=True, showfliers=False)
    ax[3,0].set(xlabel="Training environment", ylabel='Jerk ($m/s^3$)', xticklabels=ENVIRONMENTS_DISPLAY_NAME)
    ax[3,0].grid()
    # min distance to humans
    bplot8 = ax[3,1].boxplot(data_to_plot[7], showmeans=True, patch_artist=True, showfliers=False)
    ax[3,1].set(xlabel="Training environment", ylabel='Min. dist. to humans ($m$)', xticklabels=ENVIRONMENTS_DISPLAY_NAME)
    ax[3,1].grid()
    # Set color of boxplots
    for bplot in (bplot1, bplot2, bplot3, bplot4, bplot5, bplot6, bplot7, bplot8):
        for patch, color in zip(bplot['boxes'], COLORS[3:]):
            patch.set_facecolor(color)
    # Save figure
    if SAVE_FIGURES: save_figure(figure)
if METRICS_OVER_DIFFERENT_TRAINING_ENV_AND_SCENARIO:
    # Extract and aggregate data
    dataa = aggregate_data(COMPLETE_METRICS_FILE_NAMES, metrics_dir, [0,3,4], only_sarl=True)
    metrics_names = ["success_rate","time_to_goal","space_compliance","SPL","avg_speed","avg_accel.","avg_jerk","min_dist"]
    metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
    # Compute final data to plot
    data_to_plot = np.zeros((len(ENVIRONMENTS)*len(SCENARIOS),len(TESTS),len(metrics_idxs)), np.float64)
    successful_episodes = np.zeros((len(ENVIRONMENTS)*len(SCENARIOS),len(TESTS)), np.float64)
    for train_env in ENVIRONMENTS:
        for train_scenario in SCENARIOS:
            for n_humans in TESTS:
                for k, d in dataa.items():
                    if (d["train_env"] == train_env) and (d["n_humans"] == n_humans) and (d["train_scenario"] == train_scenario):
                        for m, metric in enumerate(metrics_idxs): data_to_plot[ENVIRONMENTS.index(train_env) * len(SCENARIOS) + SCENARIOS.index(train_scenario),TESTS.index(n_humans),m] = np.mean(d["data"][:,metric][~np.isnan(d["data"][:,metric])])
                        successful_episodes[ENVIRONMENTS.index(train_env) * len(SCENARIOS) + SCENARIOS.index(train_scenario),TESTS.index(n_humans)] = np.sum(d["data"][:,0] == 1)
    ## Plot
    figure, ax = plt.subplots(4,2, figsize=(18,18))
    figure.subplots_adjust(right=0.87, top=0.985, bottom=0.05, left=0.08, hspace=0.3)
    figure.suptitle("Metrics over SARL tests with increasing number of humans (averaged over all test environments and scenarios, and robot policies)")
    # success_rate
    ax[0,0].set_xticks([i for i in range(len(TESTS))])
    ax[0,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,0].set_yticks([i/10 for i in range(11)])
    ax[0,0].set(ylabel="Success rate", ylim=[0,1], xlabel="Number of humans")
    ax[0,0].grid()
    if PRINT_SUCCESSFUL_EPSIODES:
        for i in range(len(ENVIRONMENTS)*len(SCENARIOS)): ax[0,0].plot(data_to_plot[i,:,0], label=ENVIRONMENTS_DISPLAY_NAME[i//3] + "-" + SCENARIOS[i%3] + " " + str(successful_episodes[i]), color=COLORS[i//3], linewidth=2.5, linestyle=LINE_STYLES[i%3])
    else:
        for i in range(len(ENVIRONMENTS)*len(SCENARIOS)): ax[0,0].plot(data_to_plot[i,:,0], label=ENVIRONMENTS_DISPLAY_NAME[i//3] + "-" + SCENARIOS[i%3], color=COLORS[i//3], linewidth=2.5, linestyle=LINE_STYLES[i%3])
    # time_to_goal
    ax[0,1].set_xticks([i for i in range(len(TESTS))])
    ax[0,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,1].set(ylabel="Time to goal ($s$)", xlabel="Number of humans")
    ax[0,1].grid()
    if PRINT_SUCCESSFUL_EPSIODES:
        for i in range(len(ENVIRONMENTS)*len(SCENARIOS)): ax[0,1].plot(data_to_plot[i,:,1], label=ENVIRONMENTS_DISPLAY_NAME[i//3] + "-" + SCENARIOS[i%3] + " " + str(successful_episodes[i]), color=COLORS[i//3], linewidth=2.5, linestyle=LINE_STYLES[i%3])
    else:
        for i in range(len(ENVIRONMENTS)*len(SCENARIOS)): ax[0,1].plot(data_to_plot[i,:,1], label=ENVIRONMENTS_DISPLAY_NAME[i//3] + "-" + SCENARIOS[i%3], color=COLORS[i//3], linewidth=2.5, linestyle=LINE_STYLES[i%3])
    # space_compliance
    ax[2,0].set_xticks([i for i in range(len(TESTS))])
    ax[2,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,0].set_yticks([i/10 for i in range(11)])
    ax[2,0].set(ylabel="Space compliance", ylim=[0,1], xlabel="Number of humans")
    ax[2,0].grid()
    if PRINT_SUCCESSFUL_EPSIODES:
        for i in range(len(ENVIRONMENTS)*len(SCENARIOS)): ax[2,0].plot(data_to_plot[i,:,2], label=ENVIRONMENTS_DISPLAY_NAME[i//3] + "-" + SCENARIOS[i%3] + " " + str(successful_episodes[i]), color=COLORS[i//3], linewidth=2.5, linestyle=LINE_STYLES[i%3])
    else:
        for i in range(len(ENVIRONMENTS)*len(SCENARIOS)): ax[2,0].plot(data_to_plot[i,:,2], label=ENVIRONMENTS_DISPLAY_NAME[i//3] + "-" + SCENARIOS[i%3], color=COLORS[i//3], linewidth=2.5, linestyle=LINE_STYLES[i%3])
    # SPL
    ax[1,1].set_xticks([i for i in range(len(TESTS))])
    ax[1,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,1].set_yticks([i/10 for i in range(11)])
    ax[1,1].set(ylabel="SPL", ylim=[0,1], xlabel="Number of humans")
    ax[1,1].grid()
    if PRINT_SUCCESSFUL_EPSIODES:
        for i in range(len(ENVIRONMENTS)*len(SCENARIOS)): ax[1,1].plot(data_to_plot[i,:,3], label=ENVIRONMENTS_DISPLAY_NAME[i//3] + "-" + SCENARIOS[i%3] + " " + str(successful_episodes[i]), color=COLORS[i//3], linewidth=2.5, linestyle=LINE_STYLES[i%3])
    else:
        for i in range(len(ENVIRONMENTS)*len(SCENARIOS)): ax[1,1].plot(data_to_plot[i,:,3], label=ENVIRONMENTS_DISPLAY_NAME[i//3] + "-" + SCENARIOS[i%3], color=COLORS[i//3], linewidth=2.5, linestyle=LINE_STYLES[i%3])
    # speed
    ax[1,0].set_xticks([i for i in range(len(TESTS))])
    ax[1,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,0].set(ylabel="Speed ($m/s$)", xlabel="Number of humans")
    ax[1,0].grid()
    if PRINT_SUCCESSFUL_EPSIODES:
        for i in range(len(ENVIRONMENTS)*len(SCENARIOS)): ax[1,0].plot(data_to_plot[i,:,4], label=ENVIRONMENTS_DISPLAY_NAME[i//3] + "-" + SCENARIOS[i%3] + " " + str(successful_episodes[i]), color=COLORS[i//3], linewidth=2.5, linestyle=LINE_STYLES[i%3])
    else:
        for i in range(len(ENVIRONMENTS)*len(SCENARIOS)): ax[1,0].plot(data_to_plot[i,:,4], label=ENVIRONMENTS_DISPLAY_NAME[i//3] + "-" + SCENARIOS[i%3], color=COLORS[i//3], linewidth=2.5, linestyle=LINE_STYLES[i%3])
    # acceleration
    ax[2,1].set_xticks([i for i in range(len(TESTS))])
    ax[2,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,1].set(ylabel="Acceleration ($m/s^2$)", xlabel="Number of humans")
    ax[2,1].grid()
    if PRINT_SUCCESSFUL_EPSIODES:
        for i in range(len(ENVIRONMENTS)*len(SCENARIOS)): ax[2,1].plot(data_to_plot[i,:,5], label=ENVIRONMENTS_DISPLAY_NAME[i//3] + "-" + SCENARIOS[i%3] + " " + str(successful_episodes[i]), color=COLORS[i//3], linewidth=2.5, linestyle=LINE_STYLES[i%3])
    else:
        for i in range(len(ENVIRONMENTS)*len(SCENARIOS)): ax[2,1].plot(data_to_plot[i,:,5], label=ENVIRONMENTS_DISPLAY_NAME[i//3] + "-" + SCENARIOS[i%3], color=COLORS[i//3], linewidth=2.5, linestyle=LINE_STYLES[i%3])
    # jerk
    ax[3,0].set_xticks([i for i in range(len(TESTS))])
    ax[3,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,0].set(ylabel="Jerk ($m/s^3$)", xlabel="Number of humans")
    ax[3,0].grid()
    if PRINT_SUCCESSFUL_EPSIODES:
        for i in range(len(ENVIRONMENTS)*len(SCENARIOS)): ax[3,0].plot(data_to_plot[i,:,6], label=ENVIRONMENTS_DISPLAY_NAME[i//3] + "-" + SCENARIOS[i%3] + " " + str(successful_episodes[i]), color=COLORS[i//3], linewidth=2.5, linestyle=LINE_STYLES[i%3])
    else:
        for i in range(len(ENVIRONMENTS)*len(SCENARIOS)): ax[3,0].plot(data_to_plot[i,:,6], label=ENVIRONMENTS_DISPLAY_NAME[i//3] + "-" + SCENARIOS[i%3], color=COLORS[i//3], linewidth=2.5, linestyle=LINE_STYLES[i%3])
    # min dist to humans
    ax[3,1].set_xticks([i for i in range(len(TESTS))])
    ax[3,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,1].set(ylabel="Min. dist. to humans ($m$)", xlabel="Number of humans")
    ax[3,1].grid()
    if PRINT_SUCCESSFUL_EPSIODES:
        for i in range(len(ENVIRONMENTS)*len(SCENARIOS)): ax[3,1].plot(data_to_plot[i,:,7], label=ENVIRONMENTS_DISPLAY_NAME[i//3] + "-" + SCENARIOS[i%3] + " " + str(successful_episodes[i]), color=COLORS[i//3], linewidth=2.5, linestyle=LINE_STYLES[i%3])
    else:
        for i in range(len(ENVIRONMENTS)*len(SCENARIOS)): ax[3,1].plot(data_to_plot[i,:,7], label=ENVIRONMENTS_DISPLAY_NAME[i//3] + "-" + SCENARIOS[i%3], color=COLORS[i//3], linewidth=2.5, linestyle=LINE_STYLES[i%3])
    # legend
    handles, labels = ax[0,0].get_legend_handles_labels()
    if PRINT_SUCCESSFUL_EPSIODES: figure.legend(handles, labels, bbox_to_anchor=(0.90, 0.5), loc='center', title="Training \nsetting \n[successful eps. \nfor each test]")
    else:  figure.legend(handles, labels, bbox_to_anchor=(0.92, 0.5), loc='center', title="Training \nsetting")
    # Save figure
    if SAVE_FIGURES: save_figure(figure)
if METRICS_OVER_DIFFERENT_TESTING_ENVIRONMENT_ONLY_HS:
    # Extract and aggregate data
    dataa = aggregate_data(COMPLETE_METRICS_FILE_NAMES, metrics_dir, [0,1,4], only_sarl=True)
    metrics_names = ["success_rate","time_to_goal","space_compliance","SPL","avg_speed","avg_accel.","avg_jerk","min_dist"]
    metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
    # Compute final data to plot
    data_to_plot = np.zeros((len(ENVIRONMENTS),len(TESTS),len(metrics_idxs)), np.float64)
    for test_env in ENVIRONMENTS:
        for n_humans in TESTS:
            for k, d in dataa.items():
                if (d["test_env"] == test_env) and (d["n_humans"] == n_humans)  and (d["train_scenario"] == SCENARIOS[2]):
                    for m, metric in enumerate(metrics_idxs): data_to_plot[ENVIRONMENTS.index(test_env),TESTS.index(n_humans),m] = np.mean(d["data"][:,metric][~np.isnan(d["data"][:,metric])])
    ## Plot
    figure, ax = plt.subplots(4,2, figsize=(18,18))
    figure.subplots_adjust(right=0.90, top=0.985, bottom=0.05, left=0.08, hspace=0.3)
    figure.suptitle("Metrics over SARL_HS tests with increasing number of humans (averaged over all train environments and scenarios, test scenario and robot policies)")
    # success_rate
    ax[0,0].set_xticks([i for i in range(len(TESTS))])
    ax[0,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,0].set_yticks([i/10 for i in range(11)])
    ax[0,0].set(ylabel="Success rate", ylim=[0,1], xlabel="Number of humans")
    ax[0,0].grid()
    for i in range(len(ENVIRONMENTS)): ax[0,0].plot(data_to_plot[i,:,0], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i%10+3], linewidth=2.5)
    # time_to_goal
    ax[0,1].set_xticks([i for i in range(len(TESTS))])
    ax[0,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,1].set(ylabel="Time to goal ($s$)", xlabel="Number of humans")
    ax[0,1].grid()
    for i in range(len(ENVIRONMENTS)): ax[0,1].plot(data_to_plot[i,:,1], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i%10+3], linewidth=2.5)
    # space_compliance
    ax[2,0].set_xticks([i for i in range(len(TESTS))])
    ax[2,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,0].set_yticks([i/10 for i in range(11)])
    ax[2,0].set(ylabel="Space compliance", ylim=[0,1], xlabel="Number of humans")
    ax[2,0].grid()
    for i in range(len(ENVIRONMENTS)): ax[2,0].plot(data_to_plot[i,:,2], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i%10+3], linewidth=2.5)
    # SPL
    ax[1,1].set_xticks([i for i in range(len(TESTS))])
    ax[1,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,1].set_yticks([i/10 for i in range(11)])
    ax[1,1].set(ylabel="SPL", ylim=[0,1], xlabel="Number of humans")
    ax[1,1].grid()
    for i in range(len(ENVIRONMENTS)): ax[1,1].plot(data_to_plot[i,:,3], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i%10+3], linewidth=2.5)
    # speed
    ax[1,0].set_xticks([i for i in range(len(TESTS))])
    ax[1,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,0].set(ylabel="Speed ($m/s$)", xlabel="Number of humans")
    ax[1,0].grid()
    for i in range(len(ENVIRONMENTS)): ax[1,0].plot(data_to_plot[i,:,4], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i%10+3], linewidth=2.5)
    # acceleration
    ax[2,1].set_xticks([i for i in range(len(TESTS))])
    ax[2,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,1].set(ylabel="Acceleration ($m/s^2$)", xlabel="Number of humans")
    ax[2,1].grid()
    for i in range(len(ENVIRONMENTS)): ax[2,1].plot(data_to_plot[i,:,5], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i%10+3], linewidth=2.5)
    # jerk
    ax[3,0].set_xticks([i for i in range(len(TESTS))])
    ax[3,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,0].set(ylabel="Jerk ($m/s^3$)", xlabel="Number of humans")
    ax[3,0].grid()
    for i in range(len(ENVIRONMENTS)): ax[3,0].plot(data_to_plot[i,:,6], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i%10+3], linewidth=2.5)
    # min dist to humans
    ax[3,1].set_xticks([i for i in range(len(TESTS))])
    ax[3,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,1].set(ylabel="Min. dist. to humans ($m$)", xlabel="Number of humans")
    ax[3,1].grid()
    for i in range(len(ENVIRONMENTS)): ax[3,1].plot(data_to_plot[i,:,7], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i%10+3], linewidth=2.5)
    # legend
    handles, _ = ax[0,0].get_legend_handles_labels()
    figure.legend(handles, ENVIRONMENTS_DISPLAY_NAME, bbox_to_anchor=(0.95, 0.5), loc='center', title="Testing \nenviron.")
    # Save figure
    if SAVE_FIGURES: save_figure(figure)
### PAPER FIGURES
if METRICS_OVER_DIFFERENT_POLICIES:
    # Extract and aggregate data
    dataa = aggregate_data(COMPLETE_METRICS_FILE_NAMES, metrics_dir, [1,2,3,4])
    metrics_names = ["success_rate","time_to_goal","space_compliance","SPL","avg_speed","avg_accel.","avg_jerk","min_dist"]
    metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
    # Compute final data to plot
    data_to_plot = np.zeros((len(TRAINABLE_POLICIES),len(TESTS),len(metrics_idxs)), np.float64)
    for train_policy in TRAINABLE_POLICIES:
        for n_humans in TESTS:
            for k, d in dataa.items():
                if (d["robot_policy"] == train_policy) and (d["n_humans"] == n_humans):
                    for m, metric in enumerate(metrics_idxs): data_to_plot[TRAINABLE_POLICIES.index(train_policy),TESTS.index(n_humans),m] = np.mean(d["data"][:,metric][~np.isnan(d["data"][:,metric])])
    ### Plot
    figure, ax = plt.subplots(4,2, figsize=(18,18))
    figure.subplots_adjust(right=0.86, top=0.985, bottom=0.05, left=0.08, hspace=0.3)
    figure.suptitle("Metrics over tests with increasing number of humans (averaged over all test and train environments and scenarios)")
    # success_rate
    ax[0,0].set_xticks([i for i in range(len(TESTS))])
    ax[0,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,0].set_yticks([i/10 for i in range(11)])
    ax[0,0].set(ylabel="Success rate", ylim=[0,1], xlabel="Number of humans")
    ax[0,0].grid()
    for i in range(len(TRAINABLE_POLICIES)): ax[0,0].plot(data_to_plot[i,:,0], label=TRAINABLE_POLICIES[i].upper(), color=COLORS[i%10], linewidth=2.5)
    # time_to_goal
    ax[0,1].set_xticks([i for i in range(len(TESTS))])
    ax[0,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,1].set(ylabel="Time to goal ($s$)", xlabel="Number of humans")
    ax[0,1].grid()
    for i in range(len(TRAINABLE_POLICIES)): ax[0,1].plot(data_to_plot[i,:,1], label=TRAINABLE_POLICIES[i].upper(), color=COLORS[i%10], linewidth=2.5)
    # speed
    ax[1,0].set_xticks([i for i in range(len(TESTS))])
    ax[1,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,0].set(ylabel="Speed ($m/s$)", xlabel="Number of humans")
    ax[1,0].grid()
    for i in range(len(TRAINABLE_POLICIES)): ax[1,0].plot(data_to_plot[i,:,4], label=TRAINABLE_POLICIES[i].upper(), color=COLORS[i%10], linewidth=2.5)
    # SPL
    ax[1,1].set_xticks([i for i in range(len(TESTS))])
    ax[1,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,1].set_yticks([i/10 for i in range(11)])
    ax[1,1].set(ylabel="SPL", ylim=[0,1], xlabel="Number of humans")
    ax[1,1].grid()
    for i in range(len(TRAINABLE_POLICIES)): ax[1,1].plot(data_to_plot[i,:,3], label=TRAINABLE_POLICIES[i].upper(), color=COLORS[i%10], linewidth=2.5)
    # space_compliance
    ax[2,0].set_xticks([i for i in range(len(TESTS))])
    ax[2,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,0].set_yticks([i/10 for i in range(11)])
    ax[2,0].set(ylabel="Space compliance", ylim=[0,1], xlabel="Number of humans")
    ax[2,0].grid()
    for i in range(len(TRAINABLE_POLICIES)): ax[2,0].plot(data_to_plot[i,:,2], label=TRAINABLE_POLICIES[i].upper(), color=COLORS[i%10], linewidth=2.5)
    # acceleration
    ax[2,1].set_xticks([i for i in range(len(TESTS))])
    ax[2,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,1].set(ylabel="Acceleration ($m/s^2$)", xlabel="Number of humans")
    ax[2,1].grid()
    for i in range(len(TRAINABLE_POLICIES)): ax[2,1].plot(data_to_plot[i,:,5], label=TRAINABLE_POLICIES[i].upper(), color=COLORS[i%10], linewidth=2.5)
    # jerk
    ax[3,0].set_xticks([i for i in range(len(TESTS))])
    ax[3,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,0].set(ylabel="Jerk ($m/s^3$)", xlabel="Number of humans")
    ax[3,0].grid()
    for i in range(len(TRAINABLE_POLICIES)): ax[3,0].plot(data_to_plot[i,:,6], label=TRAINABLE_POLICIES[i].upper(), color=COLORS[i%10], linewidth=2.5)
    # min dist to humans
    ax[3,1].set_xticks([i for i in range(len(TESTS))])
    ax[3,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,1].set(ylabel="Min. dist. to humans ($m$)", xlabel="Number of humans")
    ax[3,1].grid()
    for i in range(len(TRAINABLE_POLICIES)): ax[3,1].plot(data_to_plot[i,:,7], label=TRAINABLE_POLICIES[i].upper(), color=COLORS[i%10], linewidth=2.5)
    # legend
    handles, _ = ax[0,0].get_legend_handles_labels()
    figure.legend(handles, [p.upper() for p in TRAINABLE_POLICIES_DISPLAY], bbox_to_anchor=(0.93, 0.5), loc='center', title="Robot \npolicy")
    # Save figure
    if SAVE_FIGURES: save_figure(figure, name="policies_metrics")
if METRICS_OVER_DIFFERENT_TRAINING_SCENARIO:
    # Extract and aggregate data
    dataa = aggregate_data(COMPLETE_METRICS_FILE_NAMES, metrics_dir, [0,1,3,4], only_sarl=True)
    metrics_names = ["success_rate","time_to_goal","space_compliance","SPL","avg_speed","avg_accel.","avg_jerk","min_dist"]
    metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
    # Compute final data to plot
    data_to_plot = np.zeros((len(SCENARIOS),len(TESTS),len(metrics_idxs)), np.float64)
    for train_scenario in SCENARIOS:
        for n_humans in TESTS:
            for k, d in dataa.items():
                if (d["train_scenario"] == train_scenario) and (d["n_humans"] == n_humans):
                    for m, metric in enumerate(metrics_idxs): data_to_plot[SCENARIOS.index(train_scenario),TESTS.index(n_humans),m] = np.mean(d["data"][:,metric][~np.isnan(d["data"][:,metric])])
    ## Plot
    figure, ax = plt.subplots(4,2, figsize=(18,18))
    figure.subplots_adjust(right=0.90, top=0.985, bottom=0.05, left=0.08, hspace=0.3)
    figure.suptitle("Metrics over SARL tests with increasing number of humans (averaged over all test environments and scenarios, train environment)")
    # success_rate
    ax[0,0].set_xticks([i for i in range(len(TESTS))])
    ax[0,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,0].set_yticks([i/10 for i in range(11)])
    ax[0,0].set(ylabel="Success rate", ylim=[0,1], xlabel="Number of humans")
    ax[0,0].grid()
    for i in range(len(SCENARIOS)): ax[0,0].plot(data_to_plot[i,:,0], label=SCENARIOS[i], color=COLORS[i%10+6], linewidth=2.5)
    # time_to_goal
    ax[0,1].set_xticks([i for i in range(len(TESTS))])
    ax[0,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,1].set(ylabel="Time to goal ($s$)", xlabel="Number of humans")
    ax[0,1].grid()
    for i in range(len(SCENARIOS)): ax[0,1].plot(data_to_plot[i,:,1], label=SCENARIOS[i], color=COLORS[i%10+6], linewidth=2.5)
    # space_compliance
    ax[2,0].set_xticks([i for i in range(len(TESTS))])
    ax[2,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,0].set_yticks([i/10 for i in range(11)])
    ax[2,0].set(ylabel="Space compliance", ylim=[0,1], xlabel="Number of humans")
    ax[2,0].grid()
    for i in range(len(SCENARIOS)): ax[2,0].plot(data_to_plot[i,:,2], label=SCENARIOS[i], color=COLORS[i%10+6], linewidth=2.5)
    # SPL
    ax[1,1].set_xticks([i for i in range(len(TESTS))])
    ax[1,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,1].set_yticks([i/10 for i in range(11)])
    ax[1,1].set(ylabel="SPL", ylim=[0,1], xlabel="Number of humans")
    ax[1,1].grid()
    for i in range(len(SCENARIOS)): ax[1,1].plot(data_to_plot[i,:,3], label=SCENARIOS[i], color=COLORS[i%10+6], linewidth=2.5)
    # speed
    ax[1,0].set_xticks([i for i in range(len(TESTS))])
    ax[1,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,0].set(ylabel="Speed ($m/s$)", xlabel="Number of humans")
    ax[1,0].grid()
    for i in range(len(SCENARIOS)): ax[1,0].plot(data_to_plot[i,:,4], label=SCENARIOS[i], color=COLORS[i%10+6], linewidth=2.5)
    # acceleration
    ax[2,1].set_xticks([i for i in range(len(TESTS))])
    ax[2,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,1].set(ylabel="Acceleration ($m/s^2$)", xlabel="Number of humans")
    ax[2,1].grid()
    for i in range(len(SCENARIOS)): ax[2,1].plot(data_to_plot[i,:,5], label=SCENARIOS[i], color=COLORS[i%10+6], linewidth=2.5)
    # jerk
    ax[3,0].set_xticks([i for i in range(len(TESTS))])
    ax[3,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,0].set(ylabel="Jerk ($m/s^3$)", xlabel="Number of humans")
    ax[3,0].grid()
    for i in range(len(SCENARIOS)): ax[3,0].plot(data_to_plot[i,:,6], label=SCENARIOS[i], color=COLORS[i%10+6], linewidth=2.5)
    # min dist to humans
    ax[3,1].set_xticks([i for i in range(len(TESTS))])
    ax[3,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,1].set(ylabel="Min. dist. to humans ($m$)", xlabel="Number of humans")
    ax[3,1].grid()
    for i in range(len(SCENARIOS)): ax[3,1].plot(data_to_plot[i,:,7], label=SCENARIOS[i], color=COLORS[i%10+6], linewidth=2.5)
    # legend
    handles, _ = ax[0,0].get_legend_handles_labels()
    figure.legend(handles, SCENARIOS, bbox_to_anchor=(0.95, 0.5), loc='center', title="Training \nscenario \nfor SARL")
    # Save figure
    if SAVE_FIGURES: save_figure(figure, name="training_scenarios_metrics")
if METRICS_OVER_DIFFERENT_TRAINING_ENVIRONMENT_ONLY_HS:
    # Extract and aggregate trainable policies data
    dataa = aggregate_data(COMPLETE_METRICS_FILE_NAMES, metrics_dir, [0,3,4], only_sarl=True)
    metrics_names = ["success_rate","time_to_goal","space_compliance","SPL","avg_speed","avg_accel.","avg_jerk","min_dist"]
    metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
    # Compute final data to plot
    data_to_plot = np.zeros((len(ENVIRONMENTS),len(TESTS),len(metrics_idxs)), np.float64)
    successful_episodes = np.zeros((len(ENVIRONMENTS),len(TESTS)), np.float64)
    for train_env in ENVIRONMENTS:
        for n_humans in TESTS:
            for k, d in dataa.items():
                if (d["train_env"] == train_env) and (d["n_humans"] == n_humans) and (d["train_scenario"] == SCENARIOS[2]):
                    for m, metric in enumerate(metrics_idxs): data_to_plot[ENVIRONMENTS.index(train_env),TESTS.index(n_humans),m] = np.mean(d["data"][:,metric][~np.isnan(d["data"][:,metric])])
                    successful_episodes[ENVIRONMENTS.index(train_env),TESTS.index(n_humans)] = np.sum(d["data"][:,0] == 1)
    ## Plot
    figure, ax = plt.subplots(4,2, figsize=(18,18))
    figure.subplots_adjust(right=0.86, top=0.985, bottom=0.05, left=0.08, hspace=0.3)
    figure.suptitle("Metrics over SARL trained on Hybrid Scenario tests with increasing number of humans (averaged over all test environments and scenarios, and robot policies)")
    # success_rate
    ax[0,0].set_xticks([i for i in range(len(TESTS))])
    ax[0,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,0].set_yticks([i/10 for i in range(11)])
    ax[0,0].set(ylabel="Success rate", ylim=[0,1], xlabel="Number of humans")
    ax[0,0].grid()
    if PRINT_SUCCESSFUL_EPSIODES: 
        for i in range(len(ENVIRONMENTS)): ax[0,0].plot(data_to_plot[i,:,0], label=ENVIRONMENTS_DISPLAY_NAME[i] + " " + str(successful_episodes[i]), color=COLORS[i+3], linewidth=2.5)
    else: 
        for i in range(len(ENVIRONMENTS)): ax[0,0].plot(data_to_plot[i,:,0], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i+3], linewidth=2.5)
    # time_to_goal
    ax[0,1].set_xticks([i for i in range(len(TESTS))])
    ax[0,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,1].set(ylabel="Time to goal ($s$)", xlabel="Number of humans")
    ax[0,1].grid()
    if PRINT_SUCCESSFUL_EPSIODES: 
        for i in range(len(ENVIRONMENTS)): ax[0,1].plot(data_to_plot[i,:,1], label=ENVIRONMENTS_DISPLAY_NAME[i] + " " + str(successful_episodes[i]), color=COLORS[i+3], linewidth=2.5)
    else:
        for i in range(len(ENVIRONMENTS)): ax[0,1].plot(data_to_plot[i,:,1], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i+3], linewidth=2.5)
    # space_compliance
    ax[2,0].set_xticks([i for i in range(len(TESTS))])
    ax[2,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,0].set_yticks([i/10 for i in range(11)])
    ax[2,0].set(ylabel="Space compliance", ylim=[0,1], xlabel="Number of humans")
    ax[2,0].grid()
    if PRINT_SUCCESSFUL_EPSIODES: 
        for i in range(len(ENVIRONMENTS)): ax[2,0].plot(data_to_plot[i,:,2], label=ENVIRONMENTS_DISPLAY_NAME[i] + " " + str(successful_episodes[i]), color=COLORS[i+3], linewidth=2.5)
    else:
        for i in range(len(ENVIRONMENTS)): ax[2,0].plot(data_to_plot[i,:,2], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i+3], linewidth=2.5)
    # SPL
    ax[1,1].set_xticks([i for i in range(len(TESTS))])
    ax[1,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,1].set_yticks([i/10 for i in range(11)])
    ax[1,1].set(ylabel="SPL", ylim=[0,1], xlabel="Number of humans")
    ax[1,1].grid()
    if PRINT_SUCCESSFUL_EPSIODES: 
        for i in range(len(ENVIRONMENTS)): ax[1,1].plot(data_to_plot[i,:,3], label=ENVIRONMENTS_DISPLAY_NAME[i] + " " + str(successful_episodes[i]), color=COLORS[i+3], linewidth=2.5)
    else:
        for i in range(len(ENVIRONMENTS)): ax[1,1].plot(data_to_plot[i,:,3], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i+3], linewidth=2.5)
    # speed
    ax[1,0].set_xticks([i for i in range(len(TESTS))])
    ax[1,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,0].set(ylabel="Speed ($m/s$)", xlabel="Number of humans")
    ax[1,0].grid()
    if PRINT_SUCCESSFUL_EPSIODES: 
        for i in range(len(ENVIRONMENTS)): ax[1,0].plot(data_to_plot[i,:,4], label=ENVIRONMENTS_DISPLAY_NAME[i] + " " + str(successful_episodes[i]), color=COLORS[i+3], linewidth=2.5)
    else:
        for i in range(len(ENVIRONMENTS)): ax[1,0].plot(data_to_plot[i,:,4], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i+3], linewidth=2.5)
    # acceleration
    ax[2,1].set_xticks([i for i in range(len(TESTS))])
    ax[2,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,1].set(ylabel="Acceleration ($m/s^2$)", xlabel="Number of humans")
    ax[2,1].grid()
    if PRINT_SUCCESSFUL_EPSIODES: 
        for i in range(len(ENVIRONMENTS)): ax[2,1].plot(data_to_plot[i,:,5], label=ENVIRONMENTS_DISPLAY_NAME[i] + " " + str(successful_episodes[i]), color=COLORS[i+3], linewidth=2.5)
    else:
        for i in range(len(ENVIRONMENTS)): ax[2,1].plot(data_to_plot[i,:,5], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i+3], linewidth=2.5)
    # jerk
    ax[3,0].set_xticks([i for i in range(len(TESTS))])
    ax[3,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,0].set(ylabel="Jerk ($m/s^3$)", xlabel="Number of humans")
    ax[3,0].grid()
    if PRINT_SUCCESSFUL_EPSIODES: 
        for i in range(len(ENVIRONMENTS)): ax[3,0].plot(data_to_plot[i,:,6], label=ENVIRONMENTS_DISPLAY_NAME[i] + " " + str(successful_episodes[i]), color=COLORS[i+3], linewidth=2.5)
    else:
        for i in range(len(ENVIRONMENTS)): ax[3,0].plot(data_to_plot[i,:,6], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i+3], linewidth=2.5)
    # min distance to humans
    ax[3,1].set_xticks([i for i in range(len(TESTS))])
    ax[3,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,1].set(ylabel="Min. dist. to humans ($m$)", xlabel="Number of humans")
    ax[3,1].grid()
    if PRINT_SUCCESSFUL_EPSIODES: 
        for i in range(len(ENVIRONMENTS)): ax[3,1].plot(data_to_plot[i,:,7], label=ENVIRONMENTS_DISPLAY_NAME[i] + " " + str(successful_episodes[i]), color=COLORS[i+3], linewidth=2.5)
    else: 
        for i in range(len(ENVIRONMENTS)): ax[3,1].plot(data_to_plot[i,:,7], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i+3], linewidth=2.5)
    # legend
    handles, labels = ax[0,0].get_legend_handles_labels()
    if PRINT_SUCCESSFUL_EPSIODES: figure.legend(handles, labels, bbox_to_anchor=(0.90, 0.5), loc='center', title="Robot \npolicy \n[successful eps. \nfor each test]")
    else: figure.legend(handles, labels, bbox_to_anchor=(0.93, 0.5), loc='center', title="Training \nenviron. \nfor \nSARL-HS")
    # Save figure
    if SAVE_FIGURES: save_figure(figure, name="train_envs_metrics")
if METRICS_OVER_BASELINE_POLICIES_AND_SARL:
    # Extract and aggregate trainable policies data
    dataa = aggregate_data(COMPLETE_METRICS_FILE_NAMES, metrics_dir, [0,1,2,3,4], only_sarl=True)
    metrics_names = ["success_rate","time_to_goal","space_compliance","SPL","avg_speed","avg_accel.","avg_jerk","min_dist"]
    metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
    # Compute final data to plot
    data_to_plot = np.zeros((len(TESTS),len(metrics_idxs)), np.float64)
    successful_episodes = np.zeros((len(TESTS),), np.float64)
    for n_humans in TESTS:
        for k, d in dataa.items():
            if (d["n_humans"] == n_humans):
                for m, metric in enumerate(metrics_idxs): data_to_plot[TESTS.index(n_humans),m] = np.mean(d["data"][:,metric][~np.isnan(d["data"][:,metric])])
                successful_episodes[TESTS.index(n_humans)] = np.sum(d["data"][:,0] == 1)
    # Extract and aggregate non trainable policies data
    ntp_dataa = aggregate_non_trainable_policies_data(COMPLETE_METRICS_FILE_NAMES, metrics_dir, [3,4])
    ntp_data_to_plot = np.zeros((len(NON_TRAINABLE_POLICIES),len(TESTS),len(metrics_idxs)), np.float64)
    ntp_successful_episodes = np.zeros((len(NON_TRAINABLE_POLICIES),len(TESTS)), np.float64)
    for ntp in NON_TRAINABLE_POLICIES:
        for n_humans in TESTS:
            for k, d in ntp_dataa.items():
                if (d["robot_policy"] == ntp) and (d["n_humans"] == n_humans):
                    for m, metric in enumerate(metrics_idxs): ntp_data_to_plot[NON_TRAINABLE_POLICIES.index(ntp),TESTS.index(n_humans),m] = np.mean(d["data"][:,metric][~np.isnan(d["data"][:,metric])])
                    ntp_successful_episodes[NON_TRAINABLE_POLICIES.index(ntp),TESTS.index(n_humans)] = np.sum(d["data"][:,0] == 1)
    ## Plot
    figure, ax = plt.subplots(4,2, figsize=(18,18))
    figure.subplots_adjust(right=0.87, top=0.985, bottom=0.05, left=0.08, hspace=0.3)
    figure.suptitle("Metrics over SARL and baseline policies tests with increasing number of humans (averaged over all test environments and scenarios, and robot policies)")
    # success_rate
    ax[0,0].set_xticks([i for i in range(len(TESTS))])
    ax[0,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,0].set_yticks([i/10 for i in range(11)])
    ax[0,0].set(ylabel="Success rate", ylim=[0,1], xlabel="Number of humans")
    ax[0,0].grid()
    if PRINT_SUCCESSFUL_EPSIODES: 
        ax[0,0].plot(data_to_plot[:,0], label="SARL" + " " + str(successful_episodes[i]), color=COLORS[1], linewidth=2.5)
        for i in range(len(NON_TRAINABLE_POLICIES)): ax[0,0].plot(ntp_data_to_plot[i,:,0], label=NON_TRAINABLE_POLICIES[i].upper() + " " + str(ntp_successful_episodes[i]), color=OTHER_COLORS_2[i], linewidth=2.5, linestyle=LINE_STYLES[1])
    else: 
        ax[0,0].plot(data_to_plot[:,0], label="SARL", color=COLORS[1], linewidth=2.5)
        for i in range(len(NON_TRAINABLE_POLICIES)): ax[0,0].plot(ntp_data_to_plot[i,:,0], label=NON_TRAINABLE_POLICIES[i].upper(), color=OTHER_COLORS_2[i], linewidth=2.5, linestyle=LINE_STYLES[1])
    # time_to_goal
    ax[0,1].set_xticks([i for i in range(len(TESTS))])
    ax[0,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,1].set(ylabel="Time to goal ($s$)", xlabel="Number of humans")
    ax[0,1].grid()
    if PRINT_SUCCESSFUL_EPSIODES: 
        ax[0,1].plot(data_to_plot[:,1], label="SARL" + " " + str(successful_episodes[i]), color=COLORS[1], linewidth=2.5)
        for i in range(len(NON_TRAINABLE_POLICIES)): ax[0,1].plot(ntp_data_to_plot[i,:,1], label=NON_TRAINABLE_POLICIES[i].upper() + " " + str(ntp_successful_episodes[i]), color=OTHER_COLORS_2[i], linewidth=2.5, linestyle=LINE_STYLES[1])
    else:
        ax[0,1].plot(data_to_plot[:,1], label="SARL", color=COLORS[1], linewidth=2.5)
        for i in range(len(NON_TRAINABLE_POLICIES)): ax[0,1].plot(ntp_data_to_plot[i,:,1], label=NON_TRAINABLE_POLICIES[i].upper(), color=OTHER_COLORS_2[i], linewidth=2.5, linestyle=LINE_STYLES[1])
    # space_compliance
    ax[2,0].set_xticks([i for i in range(len(TESTS))])
    ax[2,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,0].set_yticks([i/10 for i in range(11)])
    ax[2,0].set(ylabel="Space compliance", ylim=[0,1], xlabel="Number of humans")
    ax[2,0].grid()
    if PRINT_SUCCESSFUL_EPSIODES: 
        ax[2,0].plot(data_to_plot[:,2], label="SARL" + " " + str(successful_episodes[i]), color=COLORS[1], linewidth=2.5)
        for i in range(len(NON_TRAINABLE_POLICIES)): ax[2,0].plot(ntp_data_to_plot[i,:,2], label=NON_TRAINABLE_POLICIES[i].upper() + " " + str(ntp_successful_episodes[i]), color=OTHER_COLORS_2[i], linewidth=2.5, linestyle=LINE_STYLES[1])
    else:
        ax[2,0].plot(data_to_plot[:,2], label="SARL", color=COLORS[1], linewidth=2.5)
        for i in range(len(NON_TRAINABLE_POLICIES)): ax[2,0].plot(ntp_data_to_plot[i,:,2], label=NON_TRAINABLE_POLICIES[i].upper(), color=OTHER_COLORS_2[i], linewidth=2.5, linestyle=LINE_STYLES[1])
    # SPL
    ax[1,1].set_xticks([i for i in range(len(TESTS))])
    ax[1,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,1].set_yticks([i/10 for i in range(11)])
    ax[1,1].set(ylabel="SPL", ylim=[0,1], xlabel="Number of humans")
    ax[1,1].grid()
    if PRINT_SUCCESSFUL_EPSIODES: 
        ax[1,1].plot(data_to_plot[:,3], label="SARL" + " " + str(successful_episodes[i]), color=COLORS[1], linewidth=2.5)
        for i in range(len(NON_TRAINABLE_POLICIES)): ax[1,1].plot(ntp_data_to_plot[i,:,3], label=NON_TRAINABLE_POLICIES[i].upper() + " " + str(ntp_successful_episodes[i]), color=OTHER_COLORS_2[i], linewidth=2.5, linestyle=LINE_STYLES[1])
    else:
        ax[1,1].plot(data_to_plot[:,3], label="SARL", color=COLORS[1], linewidth=2.5)
        for i in range(len(NON_TRAINABLE_POLICIES)): ax[1,1].plot(ntp_data_to_plot[i,:,3], label=NON_TRAINABLE_POLICIES[i].upper(), color=OTHER_COLORS_2[i], linewidth=2.5, linestyle=LINE_STYLES[1])
    # speed
    ax[1,0].set_xticks([i for i in range(len(TESTS))])
    ax[1,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,0].set(ylabel="Speed ($m/s$)", xlabel="Number of humans")
    ax[1,0].grid()
    if PRINT_SUCCESSFUL_EPSIODES: 
        ax[1,0].plot(data_to_plot[:,4], label="SARL" + " " + str(successful_episodes[i]), color=COLORS[1], linewidth=2.5)
        for i in range(len(NON_TRAINABLE_POLICIES)): ax[1,0].plot(ntp_data_to_plot[i,:,4], label=NON_TRAINABLE_POLICIES[i].upper() + " " + str(ntp_successful_episodes[i]), color=OTHER_COLORS_2[i], linewidth=2.5, linestyle=LINE_STYLES[1])
    else:
        ax[1,0].plot(data_to_plot[:,4], label="SARL", color=COLORS[1], linewidth=2.5)
        for i in range(len(NON_TRAINABLE_POLICIES)): ax[1,0].plot(ntp_data_to_plot[i,:,4], label=NON_TRAINABLE_POLICIES[i].upper(), color=OTHER_COLORS_2[i], linewidth=2.5, linestyle=LINE_STYLES[1])
    # acceleration
    ax[2,1].set_xticks([i for i in range(len(TESTS))])
    ax[2,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,1].set(ylabel="Acceleration ($m/s^2$)", xlabel="Number of humans")
    ax[2,1].grid()
    if PRINT_SUCCESSFUL_EPSIODES: 
        ax[2,1].plot(data_to_plot[:,5], label="SARL" + " " + str(successful_episodes[i]), color=COLORS[1], linewidth=2.5)
        for i in range(len(NON_TRAINABLE_POLICIES)): ax[2,1].plot(ntp_data_to_plot[i,:,5], label=NON_TRAINABLE_POLICIES[i].upper() + " " + str(ntp_successful_episodes[i]), color=OTHER_COLORS_2[i], linewidth=2.5, linestyle=LINE_STYLES[1])
    else:
        ax[2,1].plot(data_to_plot[:,5], label="SARL", color=COLORS[1], linewidth=2.5)
        for i in range(len(NON_TRAINABLE_POLICIES)): ax[2,1].plot(ntp_data_to_plot[i,:,5], label=NON_TRAINABLE_POLICIES[i].upper(), color=OTHER_COLORS_2[i], linewidth=2.5, linestyle=LINE_STYLES[1])
    # jerk
    ax[3,0].set_xticks([i for i in range(len(TESTS))])
    ax[3,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,0].set(ylabel="Jerk ($m/s^3$)", xlabel="Number of humans")
    ax[3,0].grid()
    if PRINT_SUCCESSFUL_EPSIODES: 
        ax[3,0].plot(data_to_plot[:,6], label="SARL" + " " + str(successful_episodes[i]), color=COLORS[1], linewidth=2.5)
        for i in range(len(NON_TRAINABLE_POLICIES)): ax[3,0].plot(ntp_data_to_plot[i,:,6], label=NON_TRAINABLE_POLICIES[i].upper() + " " + str(ntp_successful_episodes[i]), color=OTHER_COLORS_2[i], linewidth=2.5, linestyle=LINE_STYLES[1])
    else:
        ax[3,0].plot(data_to_plot[:,6], label="SARL", color=COLORS[1], linewidth=2.5)
        for i in range(len(NON_TRAINABLE_POLICIES)): ax[3,0].plot(ntp_data_to_plot[i,:,6], label=NON_TRAINABLE_POLICIES[i].upper(), color=OTHER_COLORS_2[i], linewidth=2.5, linestyle=LINE_STYLES[1])
    # min distance to humans
    ax[3,1].set_xticks([i for i in range(len(TESTS))])
    ax[3,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,1].set(ylabel="Min. dist. to humans ($m$)", xlabel="Number of humans")
    ax[3,1].grid()
    if PRINT_SUCCESSFUL_EPSIODES: 
        ax[3,1].plot(data_to_plot[:,7], label="SARL" + " " + str(successful_episodes[i]), color=COLORS[1], linewidth=2.5)
        for i in range(len(NON_TRAINABLE_POLICIES)): ax[3,1].plot(ntp_data_to_plot[i,:,7], label=NON_TRAINABLE_POLICIES[i].upper() + " " + str(ntp_successful_episodes[i]), color=OTHER_COLORS_2[i], linewidth=2.5, linestyle=LINE_STYLES[1])
    else: 
        ax[3,1].plot(data_to_plot[:,7], label="SARL", color=COLORS[1], linewidth=2.5)
        for i in range(len(NON_TRAINABLE_POLICIES)): ax[3,1].plot(ntp_data_to_plot[i,:,7], label=NON_TRAINABLE_POLICIES[i].upper(), color=OTHER_COLORS_2[i], linewidth=2.5, linestyle=LINE_STYLES[1])
    # legend
    handles, labels = ax[0,0].get_legend_handles_labels()
    if PRINT_SUCCESSFUL_EPSIODES: figure.legend(handles, labels, bbox_to_anchor=(0.90, 0.5), loc='center', title="Robot \npolicy \n[successful eps. \nfor each test]")
    else: figure.legend(handles, labels, bbox_to_anchor=(0.93, 0.5), loc='center', title="Robot \npolicy")
    # Save figure
    if SAVE_FIGURES: save_figure(figure, name="baselines_against_sarl")
if METRICS_CROSS_TEST_AND_CROSS_TRAIN_ENVS:
    # Extract and aggregate data
    metrics_names = ["path_length","time_to_goal","space_compliance","SPL","avg_speed","avg_accel.","avg_jerk","min_dist"]
    metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
    dataa = aggregate_data(COMPLETE_METRICS_FILE_NAMES, metrics_dir, [0,1,4,5], only_sarl=True)
    # Pre-processing data
    data_to_box_plot_cross_test = []
    for m, metric in enumerate(metrics_idxs):
        env_data = []
        for test_env in ENVIRONMENTS:
            for k, d in dataa.items():
                if (d["test_env"] == test_env) and (d["train_scenario"] == SCENARIOS[2]): env_data.append(d["data"][:,metric][~np.isnan(d["data"][:,metric])])
        data_to_box_plot_cross_test.append(env_data)
    # Extract and aggregate data
    dataa = aggregate_data(COMPLETE_METRICS_FILE_NAMES, metrics_dir, [0,3,4,5], only_sarl=True)
    # Pre-processing data
    data_to_box_plot_cross_train = []
    for m, metric in enumerate(metrics_idxs):
        env_data = []
        for train_env in ENVIRONMENTS:
            for k, d in dataa.items():
                if (d["train_env"] == train_env) and (d["train_scenario"] == SCENARIOS[2]): env_data.append(d["data"][:,metric][~np.isnan(d["data"][:,metric])])
        data_to_box_plot_cross_train.append(env_data)
    # Plot
    figure, ax = plt.subplots(2,5, figsize=(20,10))
    figure.suptitle("Metrics over SARL-HS tests (averaged over all train environments and scenarios, test scenario, human density, and robot policies)")
    figure.tight_layout()
    figure.subplots_adjust(wspace=0.5, hspace=0.35, top=0.95)
    # Cross test environments
    # path_length
    bplot1 = ax[0,0].boxplot(data_to_box_plot_cross_test[0], showmeans=True, patch_artist=True, showfliers=False)
    ax[0,0].set(xlabel="Testing environment", ylabel='Path length ($m$)', xticklabels=ENVIRONMENTS_DISPLAY_NAME, ylim=[13,22])
    ax[0,0].grid()
    # time to goal
    bplot2 = ax[0,1].boxplot(data_to_box_plot_cross_test[1], showmeans=True, patch_artist=True, showfliers=False)
    ax[0,1].set(xlabel="Testing environment", ylabel='Time to goal ($s$)', xticklabels=ENVIRONMENTS_DISPLAY_NAME, ylim=[13,25])
    ax[0,1].grid()
    # space compliance
    bplot3 = ax[0,2].boxplot(data_to_box_plot_cross_test[2], showmeans=True, patch_artist=True, showfliers=False)
    ax[0,2].set(xlabel="Testing environment", ylabel='Space compliance', xticklabels=ENVIRONMENTS_DISPLAY_NAME, ylim=[0,1])
    ax[0,2].set_yticks([i/10 for i in range(11)])
    ax[0,2].grid()
    # acceleration
    bplot4 = ax[0,3].boxplot(data_to_box_plot_cross_test[5], showmeans=True, patch_artist=True, showfliers=False)
    ax[0,3].set(xlabel="Testing environment", ylabel='Accelerarion ($m/s^2$)', xticklabels=ENVIRONMENTS_DISPLAY_NAME, ylim=[0.2,2.0])
    ax[0,3].grid()
    # jerk
    bplot45 = ax[0,4].boxplot(data_to_box_plot_cross_test[6], showmeans=True, patch_artist=True, showfliers=False)
    ax[0,4].set(xlabel="Testing environment", ylabel='Jerk ($m/s^3$)', xticklabels=ENVIRONMENTS_DISPLAY_NAME, ylim=[1,14])
    ax[0,4].grid()
    # Cross train environments
    # path_length
    bplot5 = ax[1,0].boxplot(data_to_box_plot_cross_train[0], showmeans=True, patch_artist=True, showfliers=False)
    ax[1,0].set(xlabel="Training environment", ylabel='Path length ($m$)', xticklabels=ENVIRONMENTS_DISPLAY_NAME, ylim=[13,22])
    ax[1,0].grid()
    # time to goal
    bplot6 = ax[1,1].boxplot(data_to_box_plot_cross_train[1], showmeans=True, patch_artist=True, showfliers=False)
    ax[1,1].set(xlabel="Training environment", ylabel='Time to goal ($s$)', xticklabels=ENVIRONMENTS_DISPLAY_NAME, ylim=[13,25])
    ax[1,1].grid()
    # space compliance
    bplot7 = ax[1,2].boxplot(data_to_box_plot_cross_train[2], showmeans=True, patch_artist=True, showfliers=False)
    ax[1,2].set(xlabel="Training environment", ylabel='Space compliance', xticklabels=ENVIRONMENTS_DISPLAY_NAME, ylim=[0,1])
    ax[1,2].set_yticks([i/10 for i in range(11)])
    ax[1,2].grid()
    # acceleration
    bplot8 = ax[1,3].boxplot(data_to_box_plot_cross_train[5], showmeans=True, patch_artist=True, showfliers=False)
    ax[1,3].set(xlabel="Training environment", ylabel='Accelerarion ($m/s^2$)', xticklabels=ENVIRONMENTS_DISPLAY_NAME, ylim=[0.2,2.0])
    ax[1,3].grid()
    # jerk
    bplot85 = ax[1,4].boxplot(data_to_box_plot_cross_train[6], showmeans=True, patch_artist=True, showfliers=False)
    ax[1,4].set(xlabel="Training environment", ylabel='Jerk ($m/s^3$)', xticklabels=ENVIRONMENTS_DISPLAY_NAME, ylim=[1,14])
    ax[1,4].grid()
    # Set color of boxplots
    for bplot in (bplot1, bplot2, bplot3, bplot4, bplot45):
        for patch, color in zip(bplot['boxes'], OTHER_COLORS):
            patch.set_facecolor(color)
    for bplot in (bplot5, bplot6, bplot7, bplot8, bplot85):
        for patch, color in zip(bplot['boxes'], COLORS[3:]):
            patch.set_facecolor(color)
    # Save figure
    if SAVE_FIGURES: save_figure(figure, name="cross_train_and_test_env")
if METRICS_CROSS_SCENARIO_SARL:
    # Extract and aggregate data
    dataa = aggregate_data(COMPLETE_METRICS_FILE_NAMES, metrics_dir, [0,1,3], only_sarl=True)
    metrics_names = ["success_rate","time_to_goal","space_compliance","SPL","avg_speed","avg_accel.","avg_jerk","min_dist"]
    metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
    # Compute final data to plot
    data_to_plot = np.zeros((len(SCENARIOS[0:2]),len(SCENARIOS[0:2]),len(TESTS),len(metrics_idxs)), np.float64)
    for train_scenario in SCENARIOS[0:2]:
        for test_scenario in SCENARIOS[0:2]:
            for n_humans in TESTS:
                for k, d in dataa.items():
                    if (d["train_scenario"] == train_scenario) and (d["test_scenario"] == test_scenario)  and (d["n_humans"] == n_humans):
                        for m, metric in enumerate(metrics_idxs): data_to_plot[SCENARIOS.index(train_scenario),SCENARIOS[0:2].index(test_scenario),TESTS.index(n_humans),m] = np.mean(d["data"][:,metric][~np.isnan(d["data"][:,metric])])
    colorsss = ["goldenrod", "mediumvioletred"]
    ## Plot
    figure, ax = plt.subplots(4,2, figsize=(18,18))
    figure.subplots_adjust(right=0.83, top=0.985, bottom=0.05, left=0.08, hspace=0.3)
    figure.suptitle("Metrics over SARL tests with increasing number of humans (averaged over all test and train environ.)")
    # success_rate
    ax[0,0].set_xticks([i for i in range(len(TESTS))])
    ax[0,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,0].set_yticks([i/10 for i in range(11)])
    ax[0,0].set(ylabel="Success rate", ylim=[0,1], xlabel="Number of humans")
    ax[0,0].grid()
    for i in range(len(SCENARIOS[0:2])): 
        for j in range(len(SCENARIOS[0:2])):
            if i != j: ax[0,0].plot(data_to_plot[i,j,:,0], label=SCENARIOS[i]+"-on-"+SCENARIOS[j], color=colorsss[i], linewidth=2.5)
    # time_to_goal
    ax[0,1].set_xticks([i for i in range(len(TESTS))])
    ax[0,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,1].set(ylabel="Time to goal ($s$)", xlabel="Number of humans")
    ax[0,1].grid()
    for i in range(len(SCENARIOS[0:2])): 
        for j in range(len(SCENARIOS[0:2])):
            if i != j:  ax[0,1].plot(data_to_plot[i,j,:,1], label=SCENARIOS[i]+"-on-"+SCENARIOS[j], color=colorsss[i], linewidth=2.5)
    # space_compliance
    ax[2,0].set_xticks([i for i in range(len(TESTS))])
    ax[2,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,0].set_yticks([i/10 for i in range(11)])
    ax[2,0].set(ylabel="Space compliance", ylim=[0,1], xlabel="Number of humans")
    ax[2,0].grid()
    for i in range(len(SCENARIOS[0:2])): 
        for j in range(len(SCENARIOS[0:2])):
            if i != j:  ax[2,0].plot(data_to_plot[i,j,:,2], label=SCENARIOS[i]+"-on-"+SCENARIOS[j], color=colorsss[i], linewidth=2.5)
    # SPL
    ax[1,1].set_xticks([i for i in range(len(TESTS))])
    ax[1,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,1].set_yticks([i/10 for i in range(11)])
    ax[1,1].set(ylabel="SPL", ylim=[0,1], xlabel="Number of humans")
    ax[1,1].grid()
    for i in range(len(SCENARIOS[0:2])): 
        for j in range(len(SCENARIOS[0:2])):
            if i != j:  ax[1,1].plot(data_to_plot[i,j,:,3], label=SCENARIOS[i]+"-on-"+SCENARIOS[j], color=colorsss[i], linewidth=2.5)
    # speed
    ax[1,0].set_xticks([i for i in range(len(TESTS))])
    ax[1,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,0].set(ylabel="Speed ($m/s$)", xlabel="Number of humans")
    ax[1,0].grid()
    for i in range(len(SCENARIOS[0:2])): 
        for j in range(len(SCENARIOS[0:2])):
            if i != j:  ax[1,0].plot(data_to_plot[i,j,:,4], label=SCENARIOS[i]+"-on-"+SCENARIOS[j], color=colorsss[i], linewidth=2.5)
    # acceleration
    ax[2,1].set_xticks([i for i in range(len(TESTS))])
    ax[2,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,1].set(ylabel="Acceleration ($m/s^2$)", xlabel="Number of humans")
    ax[2,1].grid()
    for i in range(len(SCENARIOS[0:2])): 
        for j in range(len(SCENARIOS[0:2])):
            if i != j:  ax[2,1].plot(data_to_plot[i,j,:,5], label=SCENARIOS[i]+"-on-"+SCENARIOS[j], color=colorsss[i], linewidth=2.5)
    # jerk
    ax[3,0].set_xticks([i for i in range(len(TESTS))])
    ax[3,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,0].set(ylabel="Jerk ($m/s^3$)", xlabel="Number of humans")
    ax[3,0].grid()
    for i in range(len(SCENARIOS[0:2])): 
        for j in range(len(SCENARIOS[0:2])):
            if i != j:  ax[3,0].plot(data_to_plot[i,j,:,6], label=SCENARIOS[i]+"-on-"+SCENARIOS[j], color=colorsss[i], linewidth=2.5)
    # min dist to humans
    ax[3,1].set_xticks([i for i in range(len(TESTS))])
    ax[3,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,1].set(ylabel="Min. dist. to humans ($m$)", xlabel="Number of humans")
    ax[3,1].grid()
    for i in range(len(SCENARIOS[0:2])): 
        for j in range(len(SCENARIOS[0:2])):
            if i != j:  ax[3,1].plot(data_to_plot[i,j,:,7], label=SCENARIOS[i]+"-on-"+SCENARIOS[j], color=colorsss[i], linewidth=2.5)
    # legend
    handles, labels = ax[0,0].get_legend_handles_labels()
    figure.legend(handles, labels, bbox_to_anchor=(0.92, 0.5), loc='center', title="Training \nand \ntesting \nscenarios \nfor SARL")
    # Save figure
    if SAVE_FIGURES: save_figure(figure, name="cross_scenario")
if METRICS_CROSS_TRAIN_ENV_CC:
    # Extract and aggregate data
    dataa = aggregate_data(COMPLETE_METRICS_FILE_NAMES, metrics_dir, [0], only_sarl=True)
    metrics_names = ["success_rate","time_to_goal","space_compliance","SPL","avg_speed","avg_accel.","avg_jerk","min_dist"]
    metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
    # Compute final data to plot
    data_to_plot = np.zeros((len(ENVIRONMENTS),len(ENVIRONMENTS),len(TESTS),len(metrics_idxs)), np.float64)
    for train_env in ENVIRONMENTS:
        for test_env in ENVIRONMENTS:
            for n_humans in TESTS:
                for k, d in dataa.items():
                    if (d["train_env"] == train_env) and (d["train_scenario"] == SCENARIOS[2]) and (d["test_env"] == test_env) and (d["test_scenario"] == SCENARIOS[0])  and (d["n_humans"] == n_humans):
                        for m, metric in enumerate(metrics_idxs): data_to_plot[ENVIRONMENTS.index(train_env),ENVIRONMENTS.index(test_env),TESTS.index(n_humans),m] = np.mean(d["data"][:,metric][~np.isnan(d["data"][:,metric])])
    colorsss = ["gold","lime","dodgerblue","purple","red","cyan"]
    ## Plot
    figure, ax = plt.subplots(4,2, figsize=(18,18))
    figure.subplots_adjust(right=0.78, top=0.985, bottom=0.05, left=0.08, hspace=0.3)
    figure.suptitle("Metrics of SARL-HS policies cross env")
    # success_rate
    ax[0,0].set_xticks([i for i in range(len(TESTS))])
    ax[0,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,0].set_yticks([i/10 for i in range(11)])
    ax[0,0].set(ylabel="Success rate", ylim=[0,1], xlabel="Number of humans")
    ax[0,0].grid()
    count = 0
    for i in range(len(ENVIRONMENTS)): 
        for j in range(len(ENVIRONMENTS)):
            if i != j: 
                ax[0,0].plot(data_to_plot[i,j,:,0], label=ENVIRONMENTS_DISPLAY_NAME[i]+"-on-"+ENVIRONMENTS_DISPLAY_NAME[j], color=colorsss[count], linewidth=2.5)
                count += 1
    # time_to_goal
    ax[0,1].set_xticks([i for i in range(len(TESTS))])
    ax[0,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,1].set(ylabel="Time to goal ($s$)", xlabel="Number of humans")
    ax[0,1].grid()
    count = 0
    for i in range(len(ENVIRONMENTS)): 
        for j in range(len(ENVIRONMENTS)):
            if i != j:
                  ax[0,1].plot(data_to_plot[i,j,:,1], label=ENVIRONMENTS_DISPLAY_NAME[i]+"-on-"+ENVIRONMENTS_DISPLAY_NAME[j], color=colorsss[count], linewidth=2.5)
                  count += 1
    # space_compliance
    ax[2,0].set_xticks([i for i in range(len(TESTS))])
    ax[2,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,0].set_yticks([i/10 for i in range(11)])
    ax[2,0].set(ylabel="Space compliance", ylim=[0,1], xlabel="Number of humans")
    ax[2,0].grid()
    count = 0
    for i in range(len(ENVIRONMENTS)): 
        for j in range(len(ENVIRONMENTS)):
            if i != j:
                  ax[2,0].plot(data_to_plot[i,j,:,2], label=ENVIRONMENTS_DISPLAY_NAME[i]+"-on-"+ENVIRONMENTS_DISPLAY_NAME[j], color=colorsss[count], linewidth=2.5)
                  count += 1
    # SPL
    ax[1,1].set_xticks([i for i in range(len(TESTS))])
    ax[1,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,1].set_yticks([i/10 for i in range(11)])
    ax[1,1].set(ylabel="SPL", ylim=[0,1], xlabel="Number of humans")
    ax[1,1].grid()
    count = 0
    for i in range(len(ENVIRONMENTS)): 
        for j in range(len(ENVIRONMENTS)):
            if i != j:
                  ax[1,1].plot(data_to_plot[i,j,:,3], label=ENVIRONMENTS_DISPLAY_NAME[i]+"-on-"+ENVIRONMENTS_DISPLAY_NAME[j], color=colorsss[count], linewidth=2.5)
                  count += 1
    # speed
    ax[1,0].set_xticks([i for i in range(len(TESTS))])
    ax[1,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,0].set(ylabel="Speed ($m/s$)", xlabel="Number of humans")
    ax[1,0].grid()
    count = 0
    for i in range(len(ENVIRONMENTS)): 
        for j in range(len(ENVIRONMENTS)):
            if i != j:
                  ax[1,0].plot(data_to_plot[i,j,:,4], label=ENVIRONMENTS_DISPLAY_NAME[i]+"-on-"+ENVIRONMENTS_DISPLAY_NAME[j], color=colorsss[count], linewidth=2.5)
                  count += 1
    # acceleration
    ax[2,1].set_xticks([i for i in range(len(TESTS))])
    ax[2,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,1].set(ylabel="Acceleration ($m/s^2$)", xlabel="Number of humans")
    ax[2,1].grid()
    count = 0
    for i in range(len(ENVIRONMENTS)): 
        for j in range(len(ENVIRONMENTS)):
            if i != j:
                  ax[2,1].plot(data_to_plot[i,j,:,5], label=ENVIRONMENTS_DISPLAY_NAME[i]+"-on-"+ENVIRONMENTS_DISPLAY_NAME[j], color=colorsss[count], linewidth=2.5)
                  count += 1
    # jerk
    ax[3,0].set_xticks([i for i in range(len(TESTS))])
    ax[3,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,0].set(ylabel="Jerk ($m/s^3$)", xlabel="Number of humans")
    ax[3,0].grid()
    count = 0
    for i in range(len(ENVIRONMENTS)): 
        for j in range(len(ENVIRONMENTS)):
            if i != j:
                  ax[3,0].plot(data_to_plot[i,j,:,6], label=ENVIRONMENTS_DISPLAY_NAME[i]+"-on-"+ENVIRONMENTS_DISPLAY_NAME[j], color=colorsss[count], linewidth=2.5)
                  count += 1
    # min dist to humans
    ax[3,1].set_xticks([i for i in range(len(TESTS))])
    ax[3,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,1].set(ylabel="Min. dist. to humans ($m$)", xlabel="Number of humans")
    ax[3,1].grid()
    count = 0
    for i in range(len(ENVIRONMENTS)): 
        for j in range(len(ENVIRONMENTS)):
            if i != j:
                  ax[3,1].plot(data_to_plot[i,j,:,7], label=ENVIRONMENTS_DISPLAY_NAME[i]+"-on-"+ENVIRONMENTS_DISPLAY_NAME[j], color=colorsss[count], linewidth=2.5)
                  count += 1
    # legend
    handles, labels = ax[0,0].get_legend_handles_labels()
    figure.legend(handles, labels, bbox_to_anchor=(0.89, 0.5), loc='center', title="Training and \ntesting environments \nfor SARL-HS \n(tested on CC)")
    # Save figure
    if SAVE_FIGURES: save_figure(figure, name="sarl_hs_cross_env_cc")
if RATES_CROSS_TRAIN_ENV_CC:
    # Plots the metrics of SARL-HS trained in one env and tested in the other two divided by the performance of the SARL-HS trained and tested in the same env
    # Extract and aggregate data
    dataa = aggregate_data(COMPLETE_METRICS_FILE_NAMES, metrics_dir, [0], only_sarl=True)
    metrics_names = ["success_rate","time_to_goal","space_compliance","SPL","avg_speed","avg_accel.","avg_jerk","min_dist"]
    metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
    # Compute final data to plot
    data_to_plot = np.zeros((len(ENVIRONMENTS),len(ENVIRONMENTS),len(TESTS),len(metrics_idxs)), np.float64)
    for train_env in ENVIRONMENTS:
        for test_env in ENVIRONMENTS:
            for n_humans in TESTS:
                for k, d in dataa.items():
                    if (d["train_env"] == train_env) and (d["train_scenario"] == SCENARIOS[2]) and (d["test_env"] == test_env) and (d["test_scenario"] == SCENARIOS[0])  and (d["n_humans"] == n_humans):
                        for m, metric in enumerate(metrics_idxs): data_to_plot[ENVIRONMENTS.index(train_env),ENVIRONMENTS.index(test_env),TESTS.index(n_humans),m] = np.mean(d["data"][:,metric][~np.isnan(d["data"][:,metric])])
    # Divide data by the performance of the SARL-HS trained and tested in the same env
    rate_data_to_plot = np.zeros((len(ENVIRONMENTS),len(ENVIRONMENTS),len(TESTS),len(metrics_idxs)), np.float64)
    for i in range(len(ENVIRONMENTS)): 
        for j in range(len(ENVIRONMENTS)):
            for k in range(len(TESTS)):
                for m in range(len(metrics_idxs)):
                    rate_data_to_plot[i,j,k,m] = data_to_plot[i,j,k,m] / data_to_plot[i,i,k,m]
    ## Plot
    colorsss = ["gold","lime","dodgerblue","purple","red","cyan"]
    figure, ax = plt.subplots(4,2, figsize=(18,18))
    figure.subplots_adjust(right=0.78, top=0.985, bottom=0.05, left=0.08, hspace=0.3, wspace=0.3)
    figure.suptitle("Metrics of SARL-HS policies cross env")
    # success_rate
    ax[0,0].set_xticks([i for i in range(len(TESTS))])
    ax[0,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,0].set(ylabel="Success rate", ylim=[0,np.max(rate_data_to_plot[:,:,:,0])], xlabel="Number of humans")
    ax[0,0].grid()
    count = 0
    for i in range(len(ENVIRONMENTS)): 
        for j in range(len(ENVIRONMENTS)):
            if i != j: 
                ax[0,0].plot(rate_data_to_plot[i,j,:,0], label=ENVIRONMENTS_DISPLAY_NAME[i]+"-on-"+ENVIRONMENTS_DISPLAY_NAME[j], color=colorsss[count], linewidth=2.5)
                count += 1
    # time_to_goal
    ax[0,1].set_xticks([i for i in range(len(TESTS))])
    ax[0,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0,1].set(ylabel="Time to goal", ylim=[0,np.max(rate_data_to_plot[:,:,:,1])], xlabel="Number of humans")
    ax[0,1].grid()
    count = 0
    for i in range(len(ENVIRONMENTS)): 
        for j in range(len(ENVIRONMENTS)):
            if i != j:
                  ax[0,1].plot(rate_data_to_plot[i,j,:,1], label=ENVIRONMENTS_DISPLAY_NAME[i]+"-on-"+ENVIRONMENTS_DISPLAY_NAME[j], color=colorsss[count], linewidth=2.5)
                  count += 1
    # space_compliance
    ax[2,0].set_xticks([i for i in range(len(TESTS))])
    ax[2,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,0].set(ylabel="Space compliance", ylim=[0,np.max(rate_data_to_plot[:,:,:,2])], xlabel="Number of humans")
    ax[2,0].grid()
    count = 0
    for i in range(len(ENVIRONMENTS)): 
        for j in range(len(ENVIRONMENTS)):
            if i != j:
                  ax[2,0].plot(rate_data_to_plot[i,j,:,2], label=ENVIRONMENTS_DISPLAY_NAME[i]+"-on-"+ENVIRONMENTS_DISPLAY_NAME[j], color=colorsss[count], linewidth=2.5)
                  count += 1
    # SPL
    ax[1,1].set_xticks([i for i in range(len(TESTS))])
    ax[1,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,1].set(ylabel="SPL", ylim=[0,np.max(rate_data_to_plot[:,:,:,3])], xlabel="Number of humans")
    ax[1,1].grid()
    count = 0
    for i in range(len(ENVIRONMENTS)): 
        for j in range(len(ENVIRONMENTS)):
            if i != j:
                  ax[1,1].plot(rate_data_to_plot[i,j,:,3], label=ENVIRONMENTS_DISPLAY_NAME[i]+"-on-"+ENVIRONMENTS_DISPLAY_NAME[j], color=colorsss[count], linewidth=2.5)
                  count += 1
    # speed
    ax[1,0].set_xticks([i for i in range(len(TESTS))])
    ax[1,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1,0].set(ylabel="Speed", ylim=[0,np.max(rate_data_to_plot[:,:,:,4])], xlabel="Number of humans")
    ax[1,0].grid()
    count = 0
    for i in range(len(ENVIRONMENTS)): 
        for j in range(len(ENVIRONMENTS)):
            if i != j:
                  ax[1,0].plot(rate_data_to_plot[i,j,:,4], label=ENVIRONMENTS_DISPLAY_NAME[i]+"-on-"+ENVIRONMENTS_DISPLAY_NAME[j], color=colorsss[count], linewidth=2.5)
                  count += 1
    # acceleration
    ax[2,1].set_xticks([i for i in range(len(TESTS))])
    ax[2,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2,1].set(ylabel="Acceleration", ylim=[0,np.max(rate_data_to_plot[:,:,:,5])], xlabel="Number of humans")
    ax[2,1].grid()
    count = 0
    for i in range(len(ENVIRONMENTS)): 
        for j in range(len(ENVIRONMENTS)):
            if i != j:
                  ax[2,1].plot(rate_data_to_plot[i,j,:,5], label=ENVIRONMENTS_DISPLAY_NAME[i]+"-on-"+ENVIRONMENTS_DISPLAY_NAME[j], color=colorsss[count], linewidth=2.5)
                  count += 1
    # jerk
    ax[3,0].set_xticks([i for i in range(len(TESTS))])
    ax[3,0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,0].set(ylabel="Jerk", ylim=[0,np.max(rate_data_to_plot[:,:,:,6])], xlabel="Number of humans")
    ax[3,0].grid()
    count = 0
    for i in range(len(ENVIRONMENTS)): 
        for j in range(len(ENVIRONMENTS)):
            if i != j:
                  ax[3,0].plot(rate_data_to_plot[i,j,:,6], label=ENVIRONMENTS_DISPLAY_NAME[i]+"-on-"+ENVIRONMENTS_DISPLAY_NAME[j], color=colorsss[count], linewidth=2.5)
                  count += 1
    # min dist to humans
    ax[3,1].set_xticks([i for i in range(len(TESTS))])
    ax[3,1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[3,1].set(ylabel="Min. dist. to humans", ylim=[0,np.max(rate_data_to_plot[:,:,:,7])], xlabel="Number of humans")
    ax[3,1].grid()
    count = 0
    for i in range(len(ENVIRONMENTS)): 
        for j in range(len(ENVIRONMENTS)):
            if i != j:
                  ax[3,1].plot(rate_data_to_plot[i,j,:,7], label=ENVIRONMENTS_DISPLAY_NAME[i]+"-on-"+ENVIRONMENTS_DISPLAY_NAME[j], color=colorsss[count], linewidth=2.5)
                  count += 1
    # legend
    handles, labels = ax[0,0].get_legend_handles_labels()
    figure.legend(handles, labels, bbox_to_anchor=(0.89, 0.5), loc='center', title="Training and \ntesting environments \nfor SARL-HS \n(tested on CC)")
    # Save figure
    if SAVE_FIGURES: save_figure(figure, name="sarl_hs_cross_env_cc_rates")
if RETURN_SARL_HS_OVER_TRAIN_ENV:
    metrics_names = ["return"]
    metrics_idxs = [METRICS.index(metric) for metric in metrics_names]
    # Extract policy data
    dataa = aggregate_data(COMPLETE_METRICS_FILE_NAMES, metrics_dir, [1,2,3,4])
    data_to_plot_policies = np.zeros((len(TRAINABLE_POLICIES),len(TESTS),len(metrics_idxs)), np.float64)
    for train_policy in TRAINABLE_POLICIES:
        for n_humans in TESTS:
            for k, d in dataa.items():
                if (d["robot_policy"] == train_policy) and (d["n_humans"] == n_humans):
                    for m, metric in enumerate(metrics_idxs): data_to_plot_policies[TRAINABLE_POLICIES.index(train_policy),TESTS.index(n_humans),m] = np.mean(d["data"][:,metric][~np.isnan(d["data"][:,metric])])
    # Extract scenario data
    dataa = aggregate_data(COMPLETE_METRICS_FILE_NAMES, metrics_dir, [0,1,3,4], only_sarl=True)
    data_to_plot_scenarios = np.zeros((len(SCENARIOS),len(TESTS),len(metrics_idxs)), np.float64)
    for train_scenario in SCENARIOS:
        for n_humans in TESTS:
            for k, d in dataa.items():
                if (d["train_scenario"] == train_scenario) and (d["n_humans"] == n_humans):
                    for m, metric in enumerate(metrics_idxs): data_to_plot_scenarios[SCENARIOS.index(train_scenario),TESTS.index(n_humans),m] = np.mean(d["data"][:,metric][~np.isnan(d["data"][:,metric])])
    # Extract env data
    dataa = aggregate_data(COMPLETE_METRICS_FILE_NAMES, metrics_dir, [0,3,4], only_sarl=True)
    data_to_plot_envs = np.zeros((len(ENVIRONMENTS),len(TESTS),len(metrics_idxs)), np.float64)
    for train_env in ENVIRONMENTS:
        for n_humans in TESTS:
            for k, d in dataa.items():
                if (d["train_env"] == train_env) and (d["n_humans"] == n_humans) and (d["train_scenario"] == SCENARIOS[2]):
                    for m, metric in enumerate(metrics_idxs): data_to_plot_envs[ENVIRONMENTS.index(train_env),TESTS.index(n_humans),m] = np.mean(d["data"][:,metric][~np.isnan(d["data"][:,metric])])
    ## Plot
    figure, ax = plt.subplots(3,1, figsize=(18,18))
    figure.subplots_adjust(right=0.86, top=0.985, bottom=0.05, left=0.1, hspace=0.3)
    figure.suptitle("Return over SARL trained on Hybrid Scenario tests with increasing number of humans (averaged over all test environments and scenarios)")
    # return over policies
    ax[0].set_xticks([i for i in range(len(TESTS))])
    ax[0].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[0].set(ylabel="Discounted return", ylim=[-0.05,0.15], xlabel="Number of humans")
    ax[0].grid()
    for i in range(len(ENVIRONMENTS)): ax[0].plot(data_to_plot_policies[i,:,0], label=TRAINABLE_POLICIES[i].upper(), color=COLORS[i], linewidth=2.5)
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles, [p.upper() for p in TRAINABLE_POLICIES_DISPLAY], bbox_to_anchor=(1.09, 0.5), loc='center', title="Robot \npolicy")
    # return over scenarios
    ax[1].set_xticks([i for i in range(len(TESTS))])
    ax[1].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[1].set(ylabel="Discounted return", ylim=[-0.05,0.15], xlabel="Number of humans")
    ax[1].grid()
    for i in range(len(ENVIRONMENTS)): ax[1].plot(data_to_plot_scenarios[i,:,0], label=SCENARIOS[i], color=COLORS[i+6], linewidth=2.5)
    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles, labels, bbox_to_anchor=(1.07, 0.5), loc='center', title="Training \nscenario \nfor SARL")
    # return over envs
    ax[2].set_xticks([i for i in range(len(TESTS))])
    ax[2].set_xticklabels(TESTS_DISPLAY_NAME)
    ax[2].set(ylabel="Discounted return", ylim=[-0.05,0.15], xlabel="Number of humans")
    ax[2].grid()
    for i in range(len(ENVIRONMENTS)): ax[2].plot(data_to_plot_envs[i,:,0], label=ENVIRONMENTS_DISPLAY_NAME[i], color=COLORS[i+3], linewidth=2.5)
    handles, labels = ax[2].get_legend_handles_labels()
    ax[2].legend(handles, labels, bbox_to_anchor=(1.08, 0.5), loc='center', title="Training \nenviron. \nfor \nSARL-HS")
    # Save figure
    if SAVE_FIGURES: save_figure(figure, name="return")
if PLOT_SARL_HS_TRAJECTORIES:
    plt.rcParams['font.size'] = 25
    r_policies = ["sarl","sarl"]
    r_model_dirs = ["robot_models/trained_on_hybrid_scenario/sarl_on_sfm_guo", "robot_models/trained_on_hybrid_scenario/sarl_on_hsfm_new_guo"]
    h_models = ["hsfm_new_guo","sfm_guo","orca"]
    for human_model in h_models:
        figureee, axs = plt.subplots(1, len(r_policies), figsize=(20,10))
        figureee.tight_layout()
        figureee.subplots_adjust(hspace=0.1, wspace=0.15, bottom=0.08, top=0.90, left=0.07, right=0.98)
        figureee.suptitle(human_model.split("_")[0].upper() + "-driven humans", fontsize=30, weight='bold')
        for j, robot_policy in enumerate(r_policies):
            np.random.seed(2)
            social_nav = SocialNavSim(config_data = {"insert_robot": True, "human_policy": human_model, "headless": True,
                                                    "runge_kutta": False, "robot_visible": True, "robot_radius": 0.3,
                                                    "circle_radius": 7, "n_actors": 5, "randomize_human_positions": True, "randomize_human_attributes": False},
                                    scenario="circular_crossing", parallelize_robot = True, parallelize_humans = True)
            social_nav.set_time_step(1/100)
            social_nav.set_robot_time_step(1/4)
            social_nav.set_robot_policy(policy_name=robot_policy, crowdnav_policy=True, model_dir=os.path.join(os.path.dirname(__file__),r_model_dirs[j]), il=False)
            ax = axs[j]
            social_nav.run_and_plot_trajectories_humans_and_robot(final_time=15, plot_sample_time=3, ax=ax, show=False)
            ax.set_aspect('equal', adjustable='box')
            title = robot_policy.upper() + "-" + "".join([n[0].upper() for n in r_model_dirs[j].split("/")[1].split("_")[-2:]]) + "-" + r_model_dirs[j].split("/")[-1].split("_")[2].upper() + "-driven robot"
            ax.set_title(title, fontsize=25)
        save_figure(figureee, title=True, name="sarl_over_" + human_model.split("_")[0])
if PLOT_HSFM_HUMANS_TRAJECTORIES_PT:
    plt.rcParams['font.size'] = 25
    figureee, axs = plt.subplots(1, 1, figsize=(20,10))
    figureee.tight_layout()
    figureee.subplots_adjust(left=0.07)
    data = {'motion_model': 'hsfm_new_guo', 'headless': True, 'runge_kutta': False, 'robot_visible': False, 'grid': True, 'walls': [], 'humans': {0: {'pos': [5.2, -1.4415742755371261], 'yaw': -3.141592653589793, 'goals': [[-10.0, -1.4415742755371261]], 'des_speed': 0.85, 'radius': 0.3}, 1: {'pos': [5.924108968115273, 0.6748017875764435], 'yaw': -3.141592653589793, 'goals': [[-10.0, 0.6748017875764435]], 'des_speed': 0.85, 'radius': 0.3}, 2: {'pos': [5.9961589623155565, -0.0437187054966528], 'yaw': -3.141592653589793, 'goals': [[-10.0, -0.0437187054966528]], 'des_speed': 0.85, 'radius': 0.3}, 3: {'pos': [4.1, 1.4], 'yaw': -3.141594653589793, 'goals': [[-10.0, 1.4]], 'des_speed': 0.85, 'radius': 0.3}, 4: {'pos': [4.7, -0.9], 'yaw': -3.141592653589793, 'goals': [[-10.0, -0.9]], 'des_speed': 0.85, 'radius': 0.3}}}
    social_nav = SocialNavSim(data)
    social_nav.set_time_step(1/100)
    human_states = social_nav.run_k_steps(int(15/(1/100)), save_states_time_step=1/100)
    social_nav.plot_humans_and_robot_trajectories(axs, human_states, plot_sample_time=3, show=False, plot_goals=False)
    axs.set_aspect('equal', adjustable='box')
    axs.set(ylim=[-4.5,4.5])
    save_figure(figureee, name="parallel_traffic")
if PLOT_HSFM_HUMANS_TRAJECTORIES_CC:
    # OLD
    # plt.rcParams['font.size'] = 25
    # h_models = ["hsfm_farina","hsfm_guo","hsfm_new_guo"]
    # figureee, axs = plt.subplots(1, 2, figsize=(20,10))
    # figureee.tight_layout()
    # figureee.subplots_adjust(hspace=0.1, wspace=0.15, bottom=0.08, top=0.90, left=0.07, right=0.98)
    # figureee2, axs2 = plt.subplots(1, 1, figsize=(10,10))
    # figureee2.tight_layout()
    # for i, human_model in enumerate(h_models):
    #     if i == 0: 
    #         ax = axs[i]
    #         title = "Standard HSFM"
    #     elif i == 1: 
    #         ax = axs[i]
    #         title = "Modified HSFM"
    #     elif i == 2: 
    #         ax = axs2
    #         title = "Enhanced HSFM"
    #     ax.set_title(title, fontsize=25)
    #     social_nav = SocialNavSim(config_data = {"insert_robot": False, "human_policy": human_model, "headless": True,
    #                                             "runge_kutta": False, "robot_visible": False, "robot_radius": 0.3,
    #                                             "circle_radius": 7, "n_actors": 5, "randomize_human_positions": False, "randomize_human_attributes": False},
    #                             scenario="circular_crossing", parallelize_robot = False, parallelize_humans = True)
    #     social_nav.set_time_step(1/100)
    #     human_states = social_nav.run_k_steps(int(15/(1/100)), save_states_time_step=1/100)
    #     social_nav.plot_humans_and_robot_trajectories(ax, human_states, plot_sample_time=3, show=False)
    #     ax.set_aspect('equal', adjustable='box')
    # NEW
    plt.rcParams['font.size'] = 25
    h_models = ["hsfm_guo","hsfm_new_guo"]
    figureee, axs = plt.subplots(1, 2, figsize=(20,10))
    figureee.tight_layout()
    figureee.subplots_adjust(hspace=0.1, wspace=0.15, bottom=0.08, top=0.90, left=0.07, right=0.98)
    for i, human_model in enumerate(h_models):
        ax = axs[i]
        if i == 0: title = "HSFM"
        elif i == 1: title = "Enhanced HSFM"
        ax.set_title(title, fontsize=25)
        social_nav = SocialNavSim(config_data = {"insert_robot": False, "human_policy": human_model, "headless": True,
                                                "runge_kutta": False, "robot_visible": False, "robot_radius": 0.3,
                                                "circle_radius": 7, "n_actors": 5, "randomize_human_positions": False, "randomize_human_attributes": False},
                                scenario="circular_crossing", parallelize_robot = False, parallelize_humans = True)
        social_nav.set_time_step(1/100)
        human_states = social_nav.run_k_steps(int(15/(1/100)), save_states_time_step=1/100)
        social_nav.plot_humans_and_robot_trajectories(ax, human_states, plot_sample_time=3, show=False)
        ax.set_aspect('equal', adjustable='box')
    save_figure(figureee, name="hsfm")
if PLOT_HS_RETURN_OVER_TRAINING:
    model_dirs = ["cadrl_on_orca","cadrl_on_sfm_guo","cadrl_on_hsfm_new_guo",
              "sarl_on_orca","sarl_on_sfm_guo","sarl_on_hsfm_new_guo",
              "lstm_rl_on_orca","lstm_rl_on_sfm_guo","lstm_rl_on_hsfm_new_guo"]
    scen = "hybrid_scenario"
    scens = ["circular_crossing","parallel_traffic","hybrid_scenario"]
    ### Implementation variables (do not change)
    BASE_DIR = os.path.join(os.path.dirname(__file__),"robot_models","trained_on_" + scen)
    COLORS = list(mcolors.TABLEAU_COLORS.values())
    plt.rcParams['font.size'] = 22
    ### Return
    figure, ax = plt.subplots(1, 1, figsize=(16, 8))
    figure.tight_layout()
    figure.subplots_adjust(right=0.75, left=0.1, top=0.95, bottom=0.1)
    n = 500 # Moving average window
    ax.set(xlabel='Episode', ylabel=fr'Moving average of the discounted return', xlim=[0,10500])
    ax.grid()
    for i, model in enumerate(model_dirs):
        # Read log file
        with open(os.path.join(BASE_DIR,model,"output.log")) as f: lines = f.readlines()
        # Filter lines that start with TRAIN
        train_lines = [line for line in lines if line[27:43] == "TRAIN in episode"]
        # Take reward from train lines
        reward = np.array([float(line[-8:]) for line in train_lines], dtype = np.float64)
        # Compute moving average over N window
        moving_average_reward = np.convolve(reward, np.ones(n)/n, mode='valid')
        x = np.arange(n,len(reward)+1)
        # Plot learning curve
        label = model.split("_on_")[0].upper() + "-" + scen.split("_")[0][0].upper() + scen.split("_")[1][0].upper() + "-" + model.split("_on_")[1].split("_guo")[0].split("_new")[0].upper()
        if label[0:7] == "LSTM_RL": label = "LSTM" + label[7:]
        ax.plot(x,moving_average_reward, color=COLORS[(i+3) % len(COLORS)], label=label, linewidth=2.5)
    handles, labels = ax.get_legend_handles_labels()
    figure.legend(handles, labels, bbox_to_anchor=(0.87, 0.5), loc='center', title="Trained policy")
    save_figure(figure, name="HS_return")
plt.show()