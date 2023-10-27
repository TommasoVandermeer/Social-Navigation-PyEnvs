from social_nav_sim import SocialNav
from config.config import data

SocialNav(data).run()
# Circular crossing - config_data: [radius, n_actors, random, motion_model, headless, runge_kutta, insert_robot]
# SocialNav([7,6,False,"sfm_roboticsupo",False,False,False],mode="circular_crossing").run()