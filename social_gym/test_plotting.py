import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axis import Axis
import pickle
import numpy as np

METRICS_FILE = "Metrics_multiple_robot_policies.xlsx"
COMPLETE_METRICS_FILE = "Metrics_multiple_robot_policies.pkl"
MORE_PLOTS = True # If false, only success_rate, SPL, time_to_goal, space_compliance
PLOT_WITH_COMPLETE_DATA = True # If false, only average metrics are plotted
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
COLORS = list(mcolors.TABLEAU_COLORS.values())
METRICS = ['success_rate','collisions','truncated_eps','time_to_goal','min_speed','avg_speed',
           'max_speed','min_accel.','avg_accel.','max_accel.','min_jerk','avg_jerk','max_jerk',
           'min_dist','avg_dist','space_compliance','path_length','SPL']

def add_labels(ax:Axis, x:list[str], y:pd.Series):
    bar_labels = []
    for i, value in y.items(): bar_labels.append(round(value, 2))
    for i, name in enumerate(x): ax.text(name, y.iloc[i]/2, bar_labels[i], ha = 'center', bbox = dict(facecolor = 'white', alpha = .5))

def plot_single_test_metrics(test:str, environment:str, dataframe:pd.DataFrame):
    if not MORE_PLOTS:
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

metrics_dir = os.path.join(os.path.dirname(__file__),'tests','metrics')
file_name = os.path.join(metrics_dir,METRICS_FILE)
if PLOT_WITH_COMPLETE_DATA:
    # Complete data is in the following shape (test, n_humans_test, trials, metrics)
    with open(os.path.join(metrics_dir,COMPLETE_METRICS_FILE), "rb") as f: complete_data = pickle.load(f)
for k, test in enumerate(TESTS):
    ## Load average metrics dataframe
    dataframe = pd.read_excel(file_name, sheet_name=test, index_col=0)
    if PLOT_WITH_COMPLETE_DATA:
        # Find numerical indexes of testing environments in the dataframe
        indexes = {i: dataframe.index.get_loc(i) for i, row in dataframe.iterrows()}
        # Complete data has dimensions (n_results_files, n_humans_tests, n_trials, n_metrics)
        for i, environment in enumerate(ENVIRONMENTS):
            if i == 0: env_indexes = [indexes[test] for test in TESTED_ON_ORCA]
            if i == 1: env_indexes = [indexes[test] for test in TESTED_ON_SFM_GUO]
            if i == 2: env_indexes = [indexes[test] for test in TESTED_ON_HSFM_NEW_GUO]
            # Extracting data
            data = complete_data[env_indexes,k]
            # Plotting
            plot_single_test_complete_metrics(test, environment, data)
    else:
        for i, environment in enumerate(ENVIRONMENTS):
            # Extracting data
            if i == 0: df_env = dataframe.loc[TESTED_ON_ORCA, :]
            if i == 1: df_env = dataframe.loc[TESTED_ON_SFM_GUO, :]
            if i == 2: df_env = dataframe.loc[TESTED_ON_HSFM_NEW_GUO, :]
            # Plotting
            plot_single_test_metrics(test, environment, df_env)
plt.show()