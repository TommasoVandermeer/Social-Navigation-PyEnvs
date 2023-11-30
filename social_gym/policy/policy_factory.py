from social_gym.policy.linear import Linear
from social_gym.policy.orca import ORCA
from social_gym.policy.sfm_helbing import SFMHelbing
from social_gym.policy.sfm_guo import SFMGuo
from social_gym.policy.sfm_moussaid import SFMMoussaid
from social_gym.policy.hsfm_farina import HSFMFarina
from social_gym.policy.hsfm_guo import HSFMGuo
from social_gym.policy.hsfm_moussaid import HSFMMoussaid
from social_gym.policy.hsfm_new import HSFMNew
from social_gym.policy.hsfm_new_guo import HSFMNewGuo
from social_gym.policy.hsfm_new_moussaid import HSFMNewMoussaid

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
policy_factory['hsfm_guo'] = HSFMGuo
policy_factory['hsfm_moussaid'] = HSFMMoussaid
policy_factory['hsfm_new'] = HSFMNew
policy_factory['hsfm_new_guo'] = HSFMNewGuo
policy_factory['hsfm_new_moussaid'] = HSFMNewMoussaid