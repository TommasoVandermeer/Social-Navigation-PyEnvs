import os
import math
import numpy as np
import time
from social_gym.social_nav_sim import SocialNavSim
# from custom_config.config_example import data
from custom_config.config_socialjym_cc import data
# from custom_config.config_corridor import data
import matplotlib.pyplot as plt


## Motion models: sfm_helbing, sfm_guo, sfm_moussaid, hsfm_farina, hsfm_guo, hsfm_moussaid, 
## hsfm_new, hsfm_new_guo, hsfm_new_moussaid, orca

### SIMULATOR INITIALIZATION
## Create instance of simulator and load paramas from config file
# social_nav = SocialNavSim(data)
## Circular crossing - config_data: {radius, n_actors, randomize_human_positions, motion_model, headless, runge_kutta, insert_robot, circle_radius, randomize_human_attributes, robot_visible}
np.random.seed(1002)
social_nav = SocialNavSim(config_data = {"insert_robot": True, "human_policy": "hsfm_new_guo", "headless": False,
                                         "runge_kutta": False, "robot_visible": True, "robot_radius": 0.3,
                                         "circle_radius": 7, "n_actors": 5, "randomize_human_positions": True, "randomize_human_attributes": False},
                          scenario="circular_crossing", parallelize_robot = False, parallelize_humans = False)
## Parallel traffic scenario - config_data: {radius, n_actors, motion_model, headless, runge_kutta, insert_robot, traffic_length, traffic_height, randomize_human_attributes, robot_visible}
# np.random.seed(1000)
# social_nav = SocialNavSim(config_data = {"insert_robot": True, "human_policy": "orca", "headless": False,
#                                          "runge_kutta": False, "robot_visible": True, "robot_radius": 0.3,
#                                          "traffic_length": 14, "traffic_height": 3, "n_actors": 10, "randomize_human_attributes": False},
#                           scenario = "parallel_traffic", parallelize_robot = False, parallelize_humans = True)
## Circular crossing with static obstacles - config_data: {radius, n_actors, randomize_human_positions, motion_model, headless, runge_kutta, insert_robot, circle_radius, randomize_human_attributes, robot_visible}
# np.random.seed(1002)
# social_nav = SocialNavSim(config_data = {"insert_robot": True, "human_policy": "hsfm_new_guo", "headless": False,
#                                          "runge_kutta": False, "robot_visible": True, "robot_radius": 0.3,
#                                          "circle_radius": 7, "n_actors": 8, "randomize_human_positions": True, "randomize_human_attributes": False},
#                           scenario="circular_crossing_with_static_obstacles", parallelize_robot = False, parallelize_humans = False)

### SIMULATION UTILS
## Set environment sampling time (default is 1/60) *** WARNING: Express in fraction ***
TIME_STEP = 1/100
social_nav.set_time_step(TIME_STEP)
## Set robot sampling time (inverse of its update frequency) (default is 1/4) *** WARNING: Express in fraction ***
social_nav.set_robot_time_step(1/4)
## Set robot policy - CrowdNav trainable policy
social_nav.set_robot_policy(policy_name="sarl", crowdnav_policy=True, model_dir=os.path.join(os.path.dirname(__file__),'robot_models/CC/sarl_on_hsfm_new_guo'), il=False)
## The RL robot policies query the environment to compute next humans state by default, to propagate the observed velocity set query_env to False
# social_nav.robot.policy.query_env = False
## Set robot policy - CrowdNav non trainable policy
# social_nav.set_robot_policy(policy_name="bp", crowdnav_policy=True)
## Set robot policy - SocialNav non trainable policy
# social_nav.set_robot_policy(policy_name="orca", runge_kutta=False)
## Set a safety space both for robot and humans
# social_nav.motion_model_manager.set_safety_space(0.15)
## Change robot radius
# social_nav.robot.set_radius_and_update_graphics(0.2)
## Add a laser sensor to the robot
# social_nav.robot.add_laser_sensor(math.pi, 61, 5, uncertainty=0.01, render=True)

### SIMULATOR RUN
## Infinite loop interactive live run (controlled speed)
## Can be paused (SPACE), resetted (R), rewinded (Z) fast and speeded up (S), hide/show stats (H), origin view (O)
social_nav.run_live()
## Run only k steps at max speed
# social_nav.run_k_steps(1000, save_states_time_step=TIME_STEP)
## Run fixed time simulation of humans and robot and plot trajectories
# social_nav.run_and_plot_trajectories_humans_and_robot(final_time=15, plot_sample_time=3)
## Run multiple models test at max speed
# models = ["hsfm_new_guo"]
# social_nav.run_multiple_models_test(final_time=15, models=models, plot_sample_time=3, two_integrations=False)
## Run complete simulation with RK45 integration, get human states a posteriori and print them
# social_nav.run_complete_rk45_simulation(final_time=32, sampling_time=1/60, plot_sample_time=1.5)
## Run integration test at max speed
# social_nav.run_integration_test(final_time=30)
## Run from previously computed states (controlled speed) - RKF45 Only humans
# social_nav.run_complete_rk45_simulation(final_time=5, sampling_time=1/60, plot_sample_time=3) # simulates only human motion
# social_nav.run_from_precomputed_states(social_nav.human_states)
## Run from previously computed states (controlled speed) - Euler both humans and robot
# SIMULATION_SECONDS = 50
# human_states, robot_states = social_nav.run_k_steps(int(SIMULATION_SECONDS/TIME_STEP), save_states_time_step=TIME_STEP)
# social_nav.run_from_precomputed_states(human_states, robot_poses=robot_states)
## Run from previously computed states (controlled speed) - Euler only humans
# SIMULATION_SECONDS = 25
# human_states = social_nav.run_k_steps(int(SIMULATION_SECONDS/TIME_STEP), save_states_time_step=TIME_STEP)
# social_nav.run_from_precomputed_states(human_states)

### PLOT TRAJECTORY 
# figure, ax = plt.subplots(figsize=(10,10))
# social_nav.plot_humans_and_robot_trajectories(ax, human_states, plot_sample_time=3)
# plt.show()

### POST-SIMULATION UTILS
## Save and load states
# np.save(os.path.join(os.path.dirname(__file__),'human_states.npy'), human_states)
# np.save(os.path.join(os.path.dirname(__file__),'robot_states.npy'), robot_states)
# human_states = np.load(os.path.join(os.path.dirname(__file__),'human_states.npy'))
# robot_states = np.load(os.path.join(os.path.dirname(__file__),'robot_states.npy'))