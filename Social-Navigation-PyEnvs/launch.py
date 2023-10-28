from social_nav_sim import SocialNav
from config.config import data

# Motion models: sfm_roboticsupo, sfm_helbing, sfm_guo, sfm_moussaid, hsfm_farina, hsfm_guo, hsfm_moussaid, hsfm_new

# Infinite loop run
SocialNav(data).run()

# Circular crossing - config_data: [radius, n_actors, random, motion_model, headless, runge_kutta, insert_robot]
# SocialNav([7,6,False,"hsfm_new",False,False,False],mode="circular_crossing").run()