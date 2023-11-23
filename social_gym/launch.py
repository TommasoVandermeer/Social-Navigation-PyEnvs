import os
from social_gym.social_nav_sim import SocialNavSim
from custom_config.config_example import data
# from custom_config.config_corridor import data

# Motion models: sfm_roboticsupo, sfm_helbing, sfm_guo, sfm_moussaid, hsfm_farina, hsfm_guo, hsfm_moussaid, 
# hsfm_new, hsfm_new_guo, hsfm_new_moussaid

## SIMULATOR INITIALIZATION
# Create instance of simulator and load paramas from config file
# social_nav = SocialNavSim(data)
# Circular crossing - config_data: [radius, n_actors, random, motion_model, headless, runge_kutta, insert_robot, randomize_human_attributes, robot_visible]
social_nav = SocialNavSim([7,5,False,"sfm_guo",False,False,True,False,True],scenario="circular_crossing")

## SIMULATOR RUN
# Infinite loop interactive live run (controlled speed)
# Can be paused (SPACE), resetted (R), rewinded (Z) fast and speeded up (S), hide/show stats (H), origin view (O)
# social_nav.set_robot_policy(os.path.join(os.path.dirname(__file__),'robot_model'), model='il') # Set robot motion model
social_nav.run_live()
# Run only k steps at max speed
# social_nav.run_k_steps(1000)
# Run multiple models test at max speed
# models = ["sfm_helbing", "sfm_guo", "sfm_moussaid","hsfm_farina", "hsfm_guo", "hsfm_moussaid", "hsfm_new", "hsfm_new_guo", "hsfm_new_moussaid"]
# social_nav.run_multiple_models_test(final_time=15, models=models, plot_sample_time=3, two_integrations=True)
# Run complete simulation with RK45 integration, get human states a posteriori and print them
# social_nav.run_complete_rk45_simulation(final_time=32, sampling_time=1/60, plot_sample_time=1.5)
# Run integration test at max speed
# social_nav.run_integration_test(final_time=30)
# Run from previously computed states (controlled speed)
# social_nav.run_complete_rk45_simulation(final_time=5, sampling_time=1/60, plot_sample_time=3)
# social_nav.run_from_precomputed_states(social_nav.human_states)