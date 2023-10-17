motion_model = 'sfm_helbing'

insert_robot = False

grid = True

walls = []

humans = {0: {'pos': [14.0,7.5], 'yaw': -3.1416, 'goals': [[1.0,7.5],[14.0,7.5]]},
          1: {'pos': [7.5,14.0], 'yaw': -1.5708, 'goals': [[7.5,1.0],[7.5,14.0]]},
          2: {'pos': [1.0,7.5], 'yaw': 0.0, 'goals': [[14.0,7.5],[1.0,7.5]]},
          3: {'pos': [7.5,1.0], 'yaw': 1.5708, 'goals': [[7.5,14.0],[7.5,1.0]]}}

def initialize():
  return walls, humans, motion_model, insert_robot, grid