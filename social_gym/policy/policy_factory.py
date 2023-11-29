from social_gym.policy.linear import Linear
from social_gym.policy.orca import ORCA
from social_gym.policy.sfm_helbing import SFMHelbing
from social_gym.policy.sfm_guo import SFMGuo
from social_gym.policy.sfm_moussaid import SFMMoussaid
from social_gym.policy.hsfm_farina import HSFMFarina

def none_policy():
    return None

policy_factory = dict()
policy_factory['none'] = none_policy
policy_factory['linear'] = Linear
policy_factory['orca'] = ORCA
policy_factory['sfm_helbing'] = SFMHelbing
policy_factory['sfm_guo'] = SFMGuo
policy_factory['sfm_moussaid'] = SFMMoussaid
policy_factory['hsfm_farina'] = HSFMFarina
## TO BE IMPLEMENTED
policy_factory['hsfm_guo'] = None
policy_factory['hsfm_moussaid'] = None
policy_factory['hsfm_new'] = None
policy_factory['hsfm_new_guo'] = None
policy_factory['hsfm_new_moussaid'] = None
