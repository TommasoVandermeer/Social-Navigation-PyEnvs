import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axis import Axis

METRICS_FILE = "Metrics_multiple_robot_policies.xlsx"
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

def add_labels(ax:Axis, x:list[str], y:pd.Series):
    bar_labels = []
    for i, value in y.items(): bar_labels.append(round(value, 2))
    for i, name in enumerate(x): ax.text(name, y.iloc[i]/2, bar_labels[i], ha = 'center', bbox = dict(facecolor = 'white', alpha = .5))

def plot_single_test_metrics(test:str, environment:str, dataframe:pd.DataFrame):
    figure, ax = plt.subplots(2,2)
    figure.subplots_adjust(right=0.80)
    figure.suptitle(f"Metrics for {environment} environment - {test}")
    ax[0,0].bar(POLICY_NAMES,dataframe["success_rate"], color = COLORS, label=POLICY_NAMES)
    add_labels(ax[0,0], POLICY_NAMES, dataframe["success_rate"])
    ax[0,0].set(xlabel='Policy', ylabel='Success rate', xticklabels=[], ylim=[0,1])
    ax[0,1].bar(POLICY_NAMES,dataframe["SPL"], color = COLORS, label=POLICY_NAMES)
    add_labels(ax[0,1], POLICY_NAMES, dataframe["SPL"])
    ax[0,1].set(xlabel='Policy', ylabel='SPL', xticklabels=[], ylim=[0,1])
    ax[1,0].bar(POLICY_NAMES,dataframe["space_compliance"], color = COLORS, label=POLICY_NAMES)
    add_labels(ax[1,0], POLICY_NAMES, dataframe["space_compliance"])
    ax[1,0].set(xlabel='Policy', ylabel='Space compliance', xticklabels=[], ylim=[0,1])
    ax[1,1].bar(POLICY_NAMES,dataframe["time_to_goal"], color = COLORS, label=POLICY_NAMES)
    add_labels(ax[1,1], POLICY_NAMES, dataframe["time_to_goal"])
    ax[1,1].set(xlabel='Policy', ylabel='Time to goal', xticklabels=[])
    handles, labels = ax[0,0].get_legend_handles_labels()
    figure.legend(handles, labels, bbox_to_anchor=(0.90, 0.5), loc='center')

metrics_dir = os.path.join(os.path.dirname(__file__),'tests','metrics')
file_name = os.path.join(metrics_dir,METRICS_FILE)
for test in TESTS:
    ## Load metrics dataframe
    dataframe = pd.read_excel(file_name, sheet_name=test, index_col=0)
    for i, environment in enumerate(ENVIRONMENTS):
        # Extracting data
        if i == 0: df_env = dataframe.loc[TESTED_ON_ORCA, :]
        if i == 1: df_env = dataframe.loc[TESTED_ON_SFM_GUO, :]
        if i == 2: df_env = dataframe.loc[TESTED_ON_HSFM_NEW_GUO, :]
        # Plotting
        plot_single_test_metrics(test, environment, df_env)
plt.show()