from collections import namedtuple

ActionXY = namedtuple('ActionXY', ['vx', 'vy']) # Holonomic
ActionRot = namedtuple('ActionRot', ['v', 'r']) # Unicycle (Non-holonomic, constraints on rotation)
# TO BE REMOVED
ActionXYW = namedtuple('ActionXYW', ['bvx', 'bvy', 'w']) # Holonomic3 - Bodyframe velocity [bvx, bvy], Angular velocity w.
NewState = namedtuple('NewState', ['px', 'py', 'vx', 'vy']) # For SFM when integrating with Runge-Kutta-45
NewHeadedState = namedtuple('NewHeadedState', ['px', 'py', 'theta', 'bvx', 'bvy', 'w']) # For HSFM when integrating with Runge-Kutta-45