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
SCENARIOS = ["circular_crossing","parallel_traffic","hybrid_scenario"]
SCENARIOS_DISPLAY_NAME = [s.split("_")[0][0].upper() + s.split("_")[1][0].upper() for s in SCENARIOS]
### Implementation variables (do not change)
BASE_DIR = os.path.join(os.path.dirname(__file__),"trained_on_" + SCENARIO)
COLORS = list(mcolors.TABLEAU_COLORS.values())

plt.rcParams['font.size'] = 25

### Return
figure, ax = plt.subplots(1, 1, figsize=(16, 8))
figure.tight_layout()
figure.subplots_adjust(right=0.75, left=0.1, top=0.95, bottom=0.1)
n = 500 # Moving average window
if TITLES: figure.suptitle(f"Simple moving average over {n} episodes window of the return during robot policies training")
ax.set(xlabel='Episode', ylabel=fr'Moving average of the discounted return', xlim=[0,10500])
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
    label = model.split("_on_")[0].upper() + "-" + SCENARIO.split("_")[0][0].upper() + SCENARIO.split("_")[1][0].upper() + "-" + model.split("_on_")[1].split("_guo")[0].split("_new")[0].upper()
    if label[0:7] == "LSTM_RL": label = "LSTM" + label[7:]
    ax.plot(x,moving_average_reward, color=COLORS[(i+3) % len(COLORS)], label=label, linewidth=2.5)
handles, labels = ax.get_legend_handles_labels()
figure.legend(handles, labels, bbox_to_anchor=(0.87, 0.5), loc='center', title="Trained policy")


### Success rate
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


### Average training time
# Timestamp format
from datetime import datetime, timedelta
fmt = '%Y-%m-%d %H:%M:%S'
times = []
times_in_seconds = np.zeros((len(SCENARIOS),len(MODEL_DIRS)), np.int64)
for i, scenario in enumerate(SCENARIOS):
    base = os.path.join(os.path.dirname(__file__),"trained_on_" + scenario)
    for j, model in enumerate(MODEL_DIRS):
        # Read log file
        with open(os.path.join(base,model,"output.log")) as f: lines = f.readlines()
        # Filter lines that start with TRAIN
        rl_train_lines = [line for line in lines if line[27:43] == "TRAIN in episode"]
        initial_timestamp = rl_train_lines[0].split(", ")[0]
        final_timestamp = [l for l in rl_train_lines if int(l.split(" ")[6]) == 9999][0].split(", ")[0]
        # print(initial_timestamp, final_timestamp)
        t1 = datetime.strptime(initial_timestamp, fmt)
        t2 = datetime.strptime(final_timestamp, fmt)
        training_time = t2 - t1
        times.append(training_time)
        times_in_seconds[i,j] = training_time.total_seconds()
        print(f"Training time for {model} on {scenario} is {training_time}")
average = np.sum(times_in_seconds, axis=(0,1)) / (len(SCENARIOS) * len(MODEL_DIRS))
print(f"\nAverage training time is {str(timedelta(seconds = average))}, or {average} seconds\n")
figure, ax = plt.subplots(1, 1, figsize=(16, 8))
figure.tight_layout()
figure.subplots_adjust(right=0.75, left=0.1, top=0.95, bottom=0.1)
for j in range(len(MODEL_DIRS)): 
    label = MODEL_DIRS[j].split("_on_")[0].upper() + "-" + MODEL_DIRS[j].split("_on_")[1].split("_guo")[0].split("_new")[0].upper()
    if label[0:7] == "LSTM_RL": label = "LSTM" + label[7:]
    ax.plot([0,1,2], times_in_seconds[:,j], label=label, color=COLORS[j], linewidth=2.5)
ax.set(xlabel='Scenario', ylabel='Time for 10.000 training episodes ($s$)', xticks=np.arange(len(SCENARIOS)), xticklabels=SCENARIOS_DISPLAY_NAME)
ax.grid()
handles, labels = ax.get_legend_handles_labels()
figure.legend(handles, labels, bbox_to_anchor=(0.87, 0.5), loc='center', title="Trained policy")

plt.show()