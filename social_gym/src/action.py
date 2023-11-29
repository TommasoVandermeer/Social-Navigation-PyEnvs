from collections import namedtuple

ActionXY = namedtuple('ActionXY', ['vx', 'vy']) # Holonomic
ActionRot = namedtuple('ActionRot', ['v', 'r']) # Unicycle (Non-holonomic, constraints on rotation)
ActionXYW = namedtuple('ActionXYW', ['bvx', 'bvy', 'w']) # Holonomic3 - Bodyframe velocity [bvx, bvy], Angular velocity w.