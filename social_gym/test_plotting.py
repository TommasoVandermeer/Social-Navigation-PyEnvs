import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axis import Axis
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.colors import ListedColormap
from itertools import zip_longest
from scipy.stats import ttest_ind, f_oneway
import pickle
import numpy as np

METRICS_FILE = "Metrics_multiple_robot_policies.xlsx"
COMPLETE_METRICS_FILE = "Metrics_multiple_robot_policies.pkl"
HUMAN_TIMES_FILE = "human_times.pkl"
BAR_PLOTS = False # If true, barplots are shown
MORE_BAR_PLOTS = False # If true, more barplots are plotted
BOX_PLOTS = False # If true, boxplot are printed
HEAT_MAP = False # If true, heatmaps are plotted
SARL_ONLY_HEATMAPS = False # If true, heatmaps are plotted considering only sarl policies
SARL_ONLY_BOXPLOTS = False # If true, boxplots showing performances based on training and testing env are plotted considering only sarl policies
CURVE_PLOTS = False # If true, curves are plotted
HUMAN_TIMES_BOX_PLOTS = False # If true, humans' time to goal with and without robot are plotted
T_TEST_P_VALUE_THRESHOLD = 0.05
## IMPLEMENTATION VARIABLES - DO NOT CHANGE
TESTS = ["5_humans","15_humans","25_humans","35_humans"] # ["5_humans","7_humans","14_humans","21_humans","28_humans","35_humans"]
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
ENVIRONMENTS = ["ORCA","SFM_GUO","HSFM_NEW_GUO"]
ENVIRONMENTS_DISPLAY_NAME = ["ORCA","SFM","HSFM"]
COLORS = list(mcolors.TABLEAU_COLORS.values())
METRICS = ['success_rate','collisions','truncated_eps','time_to_goal','min_speed','avg_speed',
           'max_speed','min_accel.','avg_accel.','max_accel.','min_jerk','avg_jerk','max_jerk',
           'min_dist','avg_dist','space_compliance','path_length','SPL']

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
    ax.set_xlabel("Train environment")
    ax.set_ylabel("Test environment")
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
    figure = plt.figure()
    if only_sarl: figure.suptitle("Average metrics for SARL robot policies - " + test)
    else: figure.suptitle("Average metrics over all trained robot policies - " + test)
    outer = GridSpec(int(n_metrics/2), int(n_metrics/2), figure=figure, wspace=0.2, hspace=0.2)
    for i in range(n_metrics): # For each metric
        inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        ax1 = figure.add_subplot(inner[0])
        ax2 = figure.add_subplot(inner[1])
        plot_single_heatmap(average_metrics[i], ax1, metrics[i], anova_results[i]) # Heatmap with average values
        plot_single_ttest_map(ttest_data[i,:,:,1], ax2) # P-value of T-tests
        
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
    ax.set(xlabel="N° humans", ylabel="Average time to Goal", xticks=np.arange(len(TESTS)), xticklabels=TESTS)
    for i, test in enumerate(tests): ax.plot(time_to_goal_data[i], label=test, color=COLORS[i % 10], linewidth=2.5)
    ax.grid()
    handles, _ = ax.get_legend_handles_labels()
    figure.legend(handles, POLICY_NAMES, bbox_to_anchor=(0.90, 0.5), loc='center')

def plot_human_times_boxplots(test:str, environment:str, ep_times:list, hu_times:list, n_humans:list):
    # ep_times(list(np.array)) - (11,successful_trials) - first dimension is list
    # hu_times(list(np.array)) - (11,2,successful_trials,n_humans) - first dimension is list
    # n_humans(list(np.array)) - (11,2,successful_trials,n_humans) - first dimension is list
    figure = plt.figure()
    figure.suptitle(f"Test environment: {environment} - {test}")
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

def plot_boxplots_for_enviroments(test:str, data:list[np.array], metrics:list[str], train=True, only_sarl=False):
    figure, ax = plt.subplots(2,2)
    # figure.tight_layout()
    figure.subplots_adjust(right=0.80)
    if only_sarl: figure.suptitle("Metrics for SARL robot policies - " + test)
    else: figure.suptitle("Metrics over all trained robot policies - " + test)
    if train: x_ax_label = "Train environment"
    else: x_ax_label = "Test environment"
    if train: legend_title = "Train environment"
    else: legend_title = "Test environment"
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
plt.show()