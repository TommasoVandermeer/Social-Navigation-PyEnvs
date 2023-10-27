from social_nav_sim import SocialNav
from config.config import data

# SocialNav(data).run_k_steps(1000)
# Circular crossing - config_data: [radius, n_actors, random, motion_model, headless, runge_kutta, insert_robot]
SocialNav([7,70,False,"sfm_moussaid",False,False,False],mode="circular_crossing").run()