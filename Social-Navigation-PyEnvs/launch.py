from social_nav_sim import SocialNav
# from config.config_example import data
from config.config_corridor import data

# Motion models: sfm_roboticsupo, sfm_helbing, sfm_guo, sfm_moussaid, hsfm_farina, hsfm_guo, hsfm_moussaid, 
# hsfm_new, hsfm_new_guo, hsfm_new_moussaid

## SIMULATOR INITIALIZATION
# Create instance of simulator and load paramas from config file
# social_nav = SocialNav(data)
# Circular crossing - config_data: [radius, n_actors, random, motion_model, headless, runge_kutta, insert_robot]
social_nav = SocialNav([7,6,False,"hsfm_guo",False,False,False],mode="circular_crossing")

## SIMULATOR RUN
# Infinite loop run
social_nav.run()
# Run only k steps
# social_nav.run_k_steps(1000)
# Run test
# social_nav.run_multiple_models_test()
# Run integration test
# social_nav.run_integration_test()