from crowd_nav.policy_no_train.linear import Linear
from crowd_nav.policy_no_train.orca import ORCA
from crowd_nav.policy_no_train.sfm_helbing import SFMHelbing
from crowd_nav.policy_no_train.sfm_guo import SFMGuo
from crowd_nav.policy_no_train.sfm_moussaid import SFMMoussaid
from crowd_nav.policy_no_train.hsfm_farina import HSFMFarina
from crowd_nav.policy_no_train.hsfm_guo import HSFMGuo
from crowd_nav.policy_no_train.hsfm_moussaid import HSFMMoussaid
from crowd_nav.policy_no_train.hsfm_new import HSFMNew
from crowd_nav.policy_no_train.hsfm_new_guo import HSFMNewGuo
from crowd_nav.policy_no_train.hsfm_new_moussaid import HSFMNewMoussaid

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