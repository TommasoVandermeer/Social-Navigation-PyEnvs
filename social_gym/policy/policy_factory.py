from social_gym.policy.sfm_helbing import SFMHelbing
from social_gym.policy.sfm_guo import SFMGuo
from social_gym.policy.sfm_moussaid import SFMMoussaid

def none_policy():
    return None

policy_factory = dict()
policy_factory['none'] = none_policy
policy_factory['sfm_helbing'] = SFMHelbing
policy_factory['sfm_guo'] = SFMGuo
policy_factory['sfm_moussaid'] = SFMMoussaid