import math

motion_model = "sfm"

walls = [[[1,1], [1.5,1], [1.5,3], [1,3]],
         [[3,9], [6,9], [6,9.5], [3,9.5]]]

humans = {0: {"pos": [6.66,6.66], "yaw": -math.pi, "goals": [[5,5],[8,2]]},
          1: {"pos": [2.5,2.5], "yaw": 0.0, "goals": [[5,5],[8,2]]},
          2: {"pos": [2.0,5.0], "yaw": -math.pi, "goals": [[3,7],[8,8]], "group_id": 1},
          3: {"pos": [2.0,3.33], "yaw": 0.0, "goals": [[3,7],[8,8]], "group_id": 1}}

def initialize():
    return walls, humans, motion_model