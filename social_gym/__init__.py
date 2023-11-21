from gymnasium.envs.registration import register

register(
    id='SocialGym-v0',
    entry_point='social_gym.social_nav_gym:SocialNavGym',
)