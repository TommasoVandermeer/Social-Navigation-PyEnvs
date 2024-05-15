import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

MODEL_DIR = "cadrl_on_hsfm_new_guo"
MODEL_DIRS = ["cadrl_on_orca","cadrl_on_sfm_guo","cadrl_on_hsfm_new_guo",
              "sarl_on_orca","sarl_on_sfm_guo","sarl_on_hsfm_new_guo",
              "lstm_rl_on_orca","lstm_rl_on_sfm_guo","lstm_rl_on_hsfm_new_guo"]
TITLES = False
SCENARIO = "hybrid_scenario"
### Implementation variables (do not change)
BASE_DIR = base_dir = os.path.join(os.path.dirname(__file__),"trained_on_" + SCENARIO)
COLORS = list(mcolors.TABLEAU_COLORS.values())

plt.rcParams['font.size'] = 14

figure, ax = plt.subplots(1, 1)
figure.subplots_adjust(right=0.80)
n = 500 # Moving average window
if TITLES: figure.suptitle(f"Simple moving average over {n} episodes window of the return during robot policies training")
ax.set(xlabel='Episode', ylabel=fr'Moving average over {n} episodes of the discounted return ($\gamma = 0.9$)', xlim=[0,10500])
ax.grid()
for i, model in enumerate(MODEL_DIRS):
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
    label = model.split("_on_")[0].upper() + "_" + SCENARIO.split("_")[0][0].upper() + SCENARIO.split("_")[1][0].upper() + "_" + model.split("_on_")[1].split("_guo")[0].split("_new")[0].upper()
    if label[0:7] == "LSTM_RL": label = "LSTM-RL" + label[7:]
    ax.plot(x,moving_average_reward, color=COLORS[(i+3) % len(COLORS)], label=label, linewidth=2.5)
handles, labels = ax.get_legend_handles_labels()
figure.legend(handles, labels, bbox_to_anchor=(0.90, 0.5), loc='center', title="Trained policy")

figure, ax = plt.subplots(1, 1)
figure.subplots_adjust(right=0.80)
n = 500 # Moving average window
if TITLES: figure.suptitle(f"Simple moving average over {n} episodes window of the success rate during robot policies training")
ax.set(xlabel='Episode', ylabel='Success rate', xlim=[0,10500])
for i, model in enumerate(MODEL_DIRS):
    # Read log file
    with open(os.path.join(BASE_DIR,model,"output.log")) as f: lines = f.readlines()
    # Filter lines that start with TRAIN
    train_lines = [line for line in lines if line[27:43] == "TRAIN in episode"]
    # Take success from train lines
    success = np.array([float(line[-67:-63]) for line in train_lines], dtype = np.float64)
    # Compute moving average over N window
    moving_average_success = np.convolve(success, np.ones(n)/n, mode='valid')
    x = np.arange(n,len(success)+1)
    # Plot learning curve
    ax.plot(x,moving_average_success, color=COLORS[(i+3) % len(COLORS)], label=model, linewidth=2.5)
handles, labels = ax.get_legend_handles_labels()
figure.legend(handles, labels, bbox_to_anchor=(0.90, 0.5), loc='center')

plt.show()
