import os
from crowd_nav.train import main as train

ROBOT_POLICIES = ["cadrl","sarl","lstm_rl"]
HUMAN_POLICIES = ["orca","sfm_guo","hsfm_new_guo"]
ROBOT_TIMESTEP = 0.25
ENV_TIMESTEP = 0.0125
TIME_LIMIT = 50
ROBOT_RADIUS = 0.3
HUMANS_RADIUS = 0.3
ROBOT_PREF_SPEED = 1.0
HUMANS_PREF_SPEED = 1.0
### IMPLEMENTATION VARIABLES, DO NOT CHANGE
ENV_CONFIG_DIR = os.path.join(os.path.dirname(__file__),'configs/env.config')
POLICY_CONFIG_DIR = os.path.join(os.path.dirname(__file__),'configs/policy.config')
TRAIN_CONFIG_DIR = os.path.join(os.path.dirname(__file__),'configs/train.config')
OUTPUT_BASE_DIR = os.path.join(os.path.dirname(__file__),'data')

def write_env_config_file(human_policy:str, time_limit=TIME_LIMIT, robot_pref_speed=ROBOT_PREF_SPEED, humans_pref_speed=HUMANS_PREF_SPEED, robot_radius=ROBOT_RADIUS, humans_radius=HUMANS_RADIUS, env_timestep=ENV_TIMESTEP, robot_timestep=ROBOT_TIMESTEP):
    with open(ENV_CONFIG_DIR, "w") as f:
        f.write("[env] \n" +
                "time_limit = " + str(time_limit) + "\n" +
                "time_step = " + str(env_timestep) + "\n" +
                "robot_time_step = " + str(robot_timestep) + "\n" +
                "val_size = 100 \n" +
                "test_size = 500 \n" +
                "randomize_attributes = false \n" +
                "\n" +
                "\n" +
                "[reward] \n" +
                "success_reward = 1 \n" +
                "collision_penalty = -0.25 \n" +
                "discomfort_dist = 0.2 \n" +
                "discomfort_penalty_factor = 0.5 \n" +
                "\n" +
                "\n" +
                "[sim] \n" +
                "train_val_sim = circle_crossing \n" +
                "test_sim = circle_crossing \n" +
                "square_width = 10 \n" +
                "circle_radius = 7 \n" +
                "human_num = 5 \n" +
                "\n" +
                "\n" +
                "[humans] \n" +
                "visible = true \n" +
                "policy = " + human_policy + "\n" +
                "radius = " + str(humans_radius) + "\n" +
                "v_pref = " + str(humans_pref_speed) + "\n" +
                "sensor = coordinates \n" +
                "\n" +
                "\n" +
                "[robot] \n" +
                "visible = false \n" +
                "policy = none \n" +
                "radius = " + str(robot_radius) + "\n" +
                "v_pref = " + str(robot_pref_speed) + "\n" +
                "sensor = coordinates")

def write_policy_config_file():
    with open(POLICY_CONFIG_DIR, "w") as f:
        f.write("[rl] \n" +
                "gamma = 0.9 \n" +
                "\n" + 
                "\n" + 
                "[om] \n" +
                "cell_num = 4 \n" +
                "cell_size = 1 \n" +
                "om_channel_size = 3 \n" +
                "\n" + 
                "\n" + 
                "[action_space] \n" +
                "kinematics = holonomic \n" +
                "speed_samples = 5 \n" +
                "rotation_samples = 16 \n" +
                "sampling = exponential \n" +
                "query_env = true \n" +
                "\n" + 
                "\n" + 
                "[cadrl] \n" +
                "mlp_dims = 150, 100, 100, 1 \n" +
                "multiagent_training = false \n" +
                "\n" + 
                "\n" + 
                "[lstm_rl] \n" +
                "global_state_dim = 50 \n" +
                "mlp1_dims = 150, 100, 100, 50 \n" +
                "mlp2_dims = 150, 100, 100, 1 \n" +
                "multiagent_training = true \n" +
                "with_om = false \n" +
                "with_interaction_module = false \n" +
                "\n" + 
                "\n" + 
                "[srl] \n" +
                "mlp1_dims = 150, 100, 100, 50 \n" +
                "mlp2_dims = 150, 100, 100, 1 \n" +
                "multiagent_training = true \n" +
                "with_om = false \n" +
                "\n" + 
                "\n" + 
                "[sarl] \n" +
                "mlp1_dims = 150, 100 \n" +
                "mlp2_dims = 100, 50 \n" +
                "attention_dims = 100, 100, 1 \n" +
                "mlp3_dims = 150, 100, 100, 1 \n" +
                "multiagent_training = true \n" +
                "with_om = false \n" +
                "with_global_state = true")

def write_train_config_file(robot_il_policy:str):
    with open(TRAIN_CONFIG_DIR, "w") as f:
        f.write("[trainer] \n" +
                "batch_size = 100 \n" +
                "\n" + 
                "\n" + 
                "[imitation_learning] \n" +
                "il_episodes = 3000 \n" +
                "il_policy = " + robot_il_policy + "\n" +
                "il_epochs = 50 \n" +
                "il_learning_rate = 0.01 \n" +
                "safety_space = 0.15 \n" +
                "\n" + 
                "\n" + 
                "[train] \n" +
                "rl_learning_rate = 0.001 \n" +
                "train_batches = 100 \n" +
                "train_episodes = 10000 \n" +
                "sample_episodes = 1 \n" +
                "target_update_interval = 50 \n" +
                "evaluation_interval = 1000 \n" +
                "capacity = 100000 \n" +
                "epsilon_start = 0.5 \n" +
                "epsilon_end = 0.1 \n" +
                "epsilon_decay = 4000 \n" +
                "checkpoint_interval = 1000")

for robot_policy in ROBOT_POLICIES:
    for human_policy in HUMAN_POLICIES:
        # Write the config files
        write_env_config_file(human_policy)
        write_policy_config_file()
        write_train_config_file(human_policy)
        # Start the training
        train(ENV_CONFIG_DIR, robot_policy, POLICY_CONFIG_DIR, TRAIN_CONFIG_DIR, os.path.join(OUTPUT_BASE_DIR, str(robot_policy + "_on_" + human_policy)), '', False, False, False)