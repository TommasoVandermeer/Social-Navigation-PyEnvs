motion_model = "sfm_helbing"

runge_kutta = False

insert_robot = False

grid = True

test = False

walls = []

humans = {0: {'pos': [14.0,7.5], 'yaw': -3.1416, 'goals': [[1.0,7.5],[14.0,7.5]]},
          1: {'pos': [9.5086,13.6819], 'yaw': -1.885, 'goals': [[5.4914,1.3181],[9.5086,13.6819]]},
          2: {'pos': [2.2414,11.3206], 'yaw': -0.6283, 'goals': [[12.7586,3.6794],[2.2414,11.3206]]},
          3: {'pos': [2.2414,3.6794], 'yaw': 0.6283, 'goals': [[12.7586,11.3206],[2.2414,3.6794]]},
          4: {'pos': [9.5086,1.3181], 'yaw': 1.885, 'goals': [[5.4914,13.6819],[9.5086,1.3181]]}}

data = {"motion_model": motion_model, "runge_kutta": runge_kutta, "insert_robot": insert_robot, "grid": grid, "test": test, "humans": humans, "walls": walls}

def initialize():
  return data