import math

## Chose between sfm_roboticsupo, sfm_helbing
motion_model = "sfm_helbing"

## Decide wether to integrate with RKF45(True) or Euler(False)
runge_kutta = True

## Decide wether to insert the robot in the simulation
insert_robot = True

## Decide wether to print a unitary metric grid in the background
grid = True

## Decide wether to run a test (True) or the normal simulator
test = False

## Add walls by specifing its vertices (at least 3 vertices)
walls = [[[1,1], [1.5,1], [1.5,3], [1,3], [0.5,2]],
         [[3,9], [5,7], [6,9], [6,9.5], [3,9.5]],
         [[7,5], [7,5.5], [9,7], [9,5]]]

## Humans can be included by specifing various parameters, check src/human_agent.py
humans = {0: {"pos": [6.66,6.66], "yaw": -math.pi, "goals": [[5,5],[8,2]]},
          1: {"pos": [2.5,2.5], "yaw": 0.0, "goals": [[5,5],[8,2]]},
          2: {"pos": [2.0,5.0], "yaw": -math.pi, "goals": [[3,7],[8,8]], "group_id": 1},
          3: {"pos": [2.0,3.33], "yaw": 0.0, "goals": [[3,7],[8,8]], "group_id": 1, "radius": 0.4}}

data = {"motion_model": motion_model, "runge_kutta": runge_kutta, "insert_robot": insert_robot, "grid": grid, "test": test, "humans": humans, "walls": walls}

def initialize():
    return data