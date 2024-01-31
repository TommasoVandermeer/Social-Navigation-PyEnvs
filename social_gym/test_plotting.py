import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axis import Axis
import pickle
import numpy as np

METRICS_FILE = "Metrics_multiple_robot_policies.xlsx"
COMPLETE_METRICS_FILE = "Metrics_multiple_robot_policies.pkl"
HUMAN_TIMES_FILE = "human_times.pkl"
BAR_PLOTS = False # If true, barplots are shown
MORE_BAR_PLOTS = False # If true, more barplots are plotted
BOX_PLOTS = False # If true, boxplot are printed
HEAT_MAP = False # If true, heatmaps are plotted
CURVE_PLOTS = False # If true, curves are plotted
HUMAN_TIMES_BOX_PLOTS = True # If true, humans' time to goal with and without robot are plotted
## IMPLEMENTATION VARIABLES - DO NOT CHANGE
TESTS = ["5_humans","7_humans","14_humans","21_humans","28_humans","35_humans"]
TESTED_ON_ORCA = ["bp_on_orca.pkl",
                  "ssp_on_orca.pkl",
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
POLICY_NAMES = ["bp",
                "ssp",
                "cadrl_on_orca",
                "cadrl_on_sfm_guo",
                "cadrl_on_hsfm_new_guo",
                "sarl_on_orca",
                "sarl_on_sfm_guo",
                "sarl_on_hsfm_new_guo",
                "lstm_rl_on_orca",
                "lstm_rl_on_sfm_guo",
                "lstm_rl_on_hsfm_new_guo"]
ENVIRONMENTS = ["ORCA","SFM_GUO","HSFM_NEW_GUO"]
ENVIRONMENTS_DISPLAY_NAME = ["ORCA","SFM","HSFM"]
COLORS = list(mcolors.TABLEAU_COLORS.values())
METRICS = ['success_rate','collisions','truncated_eps','time_to_goal','min_speed','avg_speed',
           'max_speed','min_accel.','avg_accel.','max_accel.','min_jerk','avg_jerk','max_jerk',
           'min_dist','avg_dist','space_compliance','path_length','SPL']

def extract_data_from_human_times_file(results_file_list:list[str], test:str, human_times_data:dict):
    n_humans = int(test.replace("_humans",""))
    episode_times = np.empty((len(results_file_list),100), dtype=np.float64) # Assuming trials are 100
    times = np.empty((len(results_file_list),2,100,n_humans), dtype=np.float64) # Assuming trials are 100
    humans = np.empty((len(results_file_list),2,100), dtype=int) # Assuming trials are 100
    for r, results in enumerate(results_file_list):
        episode_times[r] = np.copy(human_times_data[results][test]["episode_times"])
        times[r,0] = np.copy(human_times_data[results][test]["times_to_goal_with_robot"])
        times[r,1] = np.copy(human_times_data[results][test]["times_to_goal_without_robot"])
        humans[r,0] = np.copy(human_times_data[results][test]["n_humans_reached_goal_with_robot"])
        humans[r,1] = np.copy(human_times_data[results][test]["n_humans_reached_goal_without_robot"])
    return episode_times, times, humans

def add_labels(ax:Axis, x:list[str], y:pd.Series):
    bar_labels = []
    for i, value in y.items(): bar_labels.append(round(value, 2))
    for i, name in enumerate(x): ax.text(name, y.iloc[i]/2, bar_labels[i], ha = 'center', bbox = dict(facecolor = 'white', alpha = .5))

def plot_single_heatmap(matrix:np.array, ax, metric_name:str):
    ax.imshow(matrix)
    ax.set_xlabel("Test environment")
    ax.set_ylabel("Train environment")
    ax.set_xticks(np.arange(len(ENVIRONMENTS)), labels=ENVIRONMENTS_DISPLAY_NAME)
    ax.set_yticks(np.arange(len(ENVIRONMENTS)), labels=ENVIRONMENTS_DISPLAY_NAME)
    for i in range(len(ENVIRONMENTS)):
        for j in range(len(ENVIRONMENTS)): ax.text(j, i, matrix[i, j], ha="center", va="center", color="w", weight='bold')
    ax.set_title(metric_name)

def plot_single_test_metrics(test:str, environment:str, dataframe:pd.DataFrame, more_plots:bool):
    if not more_plots:
        figure, ax = plt.subplots(2,2)
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
        figure, ax = plt.subplots(3,3)
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

def plot_single_test_complete_metrics(test:str, environment:str, data:np.array):
    figure, ax = plt.subplots(2,2)
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

def plot_heatmaps(data:pd.DataFrame, test:str):
    ## Initialize metrics matrices
    time_to_goal_matrix = np.empty((3,3), dtype=np.float64)
    path_length_matrix = np.empty((3,3), dtype=np.float64)
    space_compliance_matrix = np.empty((3,3), dtype=np.float64)
    spl_matrix = np.empty((3,3), dtype=np.float64)
    ## Extract data
    train_tests = [TRAINED_ON_ORCA, TRAINED_ON_SFM_GUO, TRAINED_ON_HSFM_NEW_GUO]
    test_tests = [TESTED_ON_ORCA, TESTED_ON_SFM_GUO, TESTED_ON_HSFM_NEW_GUO]
    for i in range(len(ENVIRONMENTS)):
        for j in range(len(ENVIRONMENTS)):
            test_set = list(set(train_tests[i]) & set(test_tests[j]))
            time_to_goal_matrix[i,j] = round(data.loc[test_set]["time_to_goal"].sum() / len(test_set),2)
            path_length_matrix[i,j] = round(data.loc[test_set]["path_length"].sum() / len(test_set),2)
            space_compliance_matrix[i,j] = round(data.loc[test_set]["space_compliance"].sum() / len(test_set),2)
            spl_matrix[i,j] = round(data.loc[test_set]["SPL"].sum() / len(test_set),2)
    ## Plot heatmaps
    figure, ax = plt.subplots(2,2)
    figure.suptitle("Average metrics over all trained robot policies - " + test)
    plot_single_heatmap(time_to_goal_matrix, ax[0,0], "Time to Goal")
    plot_single_heatmap(path_length_matrix, ax[0,1], "Path Length")
    plot_single_heatmap(space_compliance_matrix, ax[1,0], "Space Compliance")
    plot_single_heatmap(spl_matrix, ax[1,1], "SPL")
    figure.tight_layout()

def plot_curves(data:list[pd.DataFrame], test_env:str):
    # Extract data
    time_to_goal_data = np.empty((len(data[0].index.values),len(TESTS)), dtype=np.float64)
    tests = []
    for i, test in enumerate(data[0].index.values):
        tests.append(test)
        for j, df in enumerate(data): time_to_goal_data[i,j] = df.loc[test]["time_to_goal"]
    # Plot curves
    figure, ax = plt.subplots(1,1)
    figure.subplots_adjust(right=0.80)
    figure.suptitle("Average time to goal over tests with increasing crowd density - Environment: " + test_env)
    ax.set(xlabel="N° humans", ylabel="Average time to Goal", xticklabels=["0","5 humans","7 humans","14 humans","21 humans","28 humans","35 humans"])
    for i, test in enumerate(tests): ax.plot(time_to_goal_data[i], label=test, color=COLORS[i % 10], linewidth=2)
    ax.grid()
    handles, labels = ax.get_legend_handles_labels()
    figure.legend(handles, POLICY_NAMES, bbox_to_anchor=(0.90, 0.5), loc='center')

def plot_human_times_boxplots(test:str, environment:str, ep_times:np.array, hu_times:np.array, n_humans:np.array):
    figure, ax = plt.subplots(2,2)
    figure.subplots_adjust(right=0.80)
    figure.suptitle(f"Test environment: {environment} - {test}")
    # Compute average time to goal among humans who reached the goal for each trial - Reduce shape from (11,2,100,n_humans) to (11,2,100)
    avg_hu_times = np.empty(n_humans.shape, dtype=np.float64)
    for i, results in enumerate(hu_times):
        for j, test in enumerate(results):
            for k, human_times in enumerate(test):
                avg_hu_times[i,j,k] = np.nansum(human_times) / n_humans[i,j,k]
    # Divide data with robot and without robot
    avg_hu_times_w_robot = np.copy(avg_hu_times[:,0,:])
    avg_hu_times_wout_robot = np.copy(avg_hu_times[:,1,:])
    # Filter NaNs from avg_hu_times (otherwise box plot does not work) - avg_hu_times.shape = (11,2,100), NaNs are in the last dimension
    nan_mask_w_robot = ~np.isnan(avg_hu_times_w_robot)
    filtered_avg_hu_times_w_robot = [row[nan_mask_row] for row, nan_mask_row in zip(avg_hu_times_w_robot, nan_mask_w_robot)]
    nan_mask_wout_robot = ~np.isnan(avg_hu_times_wout_robot)
    filtered_avg_hu_times_wout_robot = [row[nan_mask_row] for row, nan_mask_row in zip(avg_hu_times_wout_robot, nan_mask_wout_robot)]
    # Time to goal with robot
    bplot1 = ax[0,0].boxplot(filtered_avg_hu_times_w_robot, showmeans=True, labels=POLICY_NAMES, patch_artist=True)
    ax[0,0].set(xlabel='Policy', ylabel="Average humans' Time to goal", xticklabels=[], title="Experiments with robot")
    # N° humans that reached the goal with robot
    bplot2 = ax[0,1].boxplot(n_humans.T[:,0,:], showmeans=True, labels=POLICY_NAMES, patch_artist=True)
    ax[0,1].set(xlabel='Policy', ylabel="Number of humans who reached the goal", xticklabels=[], title="Experiments with robot")
    # Time to goal without robot
    bplot3 = ax[1,0].boxplot(filtered_avg_hu_times_wout_robot, showmeans=True, labels=POLICY_NAMES, patch_artist=True)
    ax[1,0].set(xlabel='', ylabel="Average humans' Time to goal", xticklabels=[], title="Experiments without robot")
    # N° humans that reached the goal without robot
    bplot4 = ax[1,1].boxplot(n_humans.T[:,1,:], showmeans=True, labels=POLICY_NAMES, patch_artist=True)
    ax[1,1].set(xlabel='', ylabel="Number of humans who reached the goal", xticklabels=[], title="Experiments without robot")
    # Set color of boxplots
    for bplot in (bplot1, bplot2, bplot3, bplot4):
        for patch, color in zip(bplot['boxes'], COLORS):
            patch.set_facecolor(color)
    # Legend
    figure.legend(bplot1["boxes"], POLICY_NAMES, bbox_to_anchor=(0.90, 0.5), loc='center')

metrics_dir = os.path.join(os.path.dirname(__file__),'tests','metrics')
file_name = os.path.join(metrics_dir,METRICS_FILE)
# Complete data is in the following shape (test, n_humans_test, trials, metrics)
with open(os.path.join(metrics_dir,COMPLETE_METRICS_FILE), "rb") as f: complete_data = pickle.load(f)
for k, test in enumerate(TESTS):
    ## Load average metrics dataframe
    dataframe = pd.read_excel(file_name, sheet_name=test, index_col=0)
    if BOX_PLOTS:
        # Find numerical indexes of testing environments in the dataframe
        indexes = {i: dataframe.index.get_loc(i) for i, row in dataframe.iterrows()}
        # Complete data has dimensions (n_results_files, n_humans_tests, n_trials, n_metrics)
        for i, environment in enumerate(ENVIRONMENTS):
            if i == 0: env_indexes = [indexes[test] for test in TESTED_ON_ORCA]
            elif i == 1: env_indexes = [indexes[test] for test in TESTED_ON_SFM_GUO]
            else: env_indexes = [indexes[test] for test in TESTED_ON_HSFM_NEW_GUO]
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
        # Heatmap for each n_humans test
        plot_heatmaps(dataframe.loc[TRAINED_POLICIES_TESTS, :], test)
        # Heatmap for average above all n_humans tests
        if k == 0: average_df = dataframe.loc[TRAINED_POLICIES_TESTS, :]
        else: 
            average_df += dataframe.loc[TRAINED_POLICIES_TESTS, :]
            if k == len(TESTS) - 1: average_df /= len(TESTS); plot_heatmaps(average_df, "Average of all tests")
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
plt.show()