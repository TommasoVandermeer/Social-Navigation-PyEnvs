import os
import math
import numpy as np
from social_gym.social_nav_sim import SocialNavSim
from custom_config.config_example import data
# from custom_config.config_corridor import data

## Motion models: sfm_roboticsupo, sfm_helbing, sfm_guo, sfm_moussaid, hsfm_farina, hsfm_guo, hsfm_moussaid, 
## hsfm_new, hsfm_new_guo, hsfm_new_moussaid, orca

### SIMULATOR INITIALIZATION
## Create instance of simulator and load paramas from config file
# social_nav = SocialNavSim(data)
# Circular crossing - config_data: [radius, n_actors, random, motion_model, headless, runge_kutta,s insert_robot, randomize_human_attributes, robot_visible]
HEADLESS = False
INSERT_ROBOT = True
ROBOT_VISIBLE = True
RANDOMIZE_HUMAN_POSITIONS = False
RANDOMIZE_HUMAN_ATTRIBUTES = False
RUNGE_KUTTA = False
social_nav = SocialNavSim([7,30,RANDOMIZE_HUMAN_POSITIONS,"sfm_guo",HEADLESS,RUNGE_KUTTA,INSERT_ROBOT,RANDOMIZE_HUMAN_ATTRIBUTES,ROBOT_VISIBLE],scenario="circular_crossing")

### SIMULATION UTILS
## Set environment sampling time (default is 1/60) *** WARNING: Express in fraction ***
TIME_STEP = 1/60
social_nav.set_time_step(1/60)
## Set robot sampling time (inverse of its update frequency) (default is 1/4) *** WARNING: Express in fraction ***
social_nav.set_robot_time_step(1/4)
## Set robot policy - CrowdNav trainable policy
# social_nav.set_robot_policy(policy_name="sarl", crowdnav_policy=True, model_dir=os.path.join(os.path.dirname(__file__),'robot_models/sarl_on_sfm_guo'), il=False)
## Set robot policy - CrowdNav non trainable policy
social_nav.set_robot_policy(policy_name="bp", crowdnav_policy=True)
## Set robot policy - SocialNav non trainable policy
# social_nav.set_robot_policy(policy_name="sfm_guo", runge_kutta=False)
## Set a safety space both for robot and humans
# social_nav.motion_model_manager.set_safety_space(0.00)
## Change robot radius
# social_nav.robot.set_radius_and_update_graphics(0.2)
## Add a laser sensor to the robot
# social_nav.robot.add_laser_sensor(math.pi, 61, 5, uncertainty=0.01, render=True)

### SIMULATOR RUN
## Infinite loop interactive live run (controlled speed)
## Can be paused (SPACE), resetted (R), rewinded (Z) fast and speeded up (S), hide/show stats (H), origin view (O)
social_nav.run_live()
## Run only k steps at max speed
# social_nav.run_k_steps(1000)
## Run multiple models test at max speed
# models = ["sfm_helbing", "sfm_guo", "sfm_moussaid","hsfm_farina", "hsfm_guo", "hsfm_moussaid", "hsfm_new", "hsfm_new_guo", "hsfm_new_moussaid", "orca"]
# social_nav.run_multiple_models_test(final_time=15, models=models, plot_sample_time=3, two_integrations=True)
## Run complete simulation with RK45 integration, get human states a posteriori and print them
# social_nav.run_complete_rk45_simulation(final_time=32, sampling_time=1/60, plot_sample_time=1.5)
## Run integration test at max speed
# social_nav.run_integration_test(final_time=30)
## Run from previously computed states (controlled speed) - RKF45 Only humans
# social_nav.run_complete_rk45_simulation(final_time=5, sampling_time=1/60, plot_sample_time=3) # simulates only human motion
# social_nav.run_from_precomputed_states(social_nav.human_states)
## Run from previously computed states (controlled speed) - Euler both humans and robot
# SIMULATION_SECONDS = 30
# human_states, robot_states = social_nav.run_k_steps(int(SIMULATION_SECONDS/TIME_STEP))
# social_nav.run_from_precomputed_states(human_states, robot_poses=robot_states)