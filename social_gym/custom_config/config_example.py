import math

## Run pygame without GUI - WARNING: No event handling if true
headless = False

## Chose between sfm_roboticsupo, sfm_helbing, sfm_guo, sfm_moussaid, hsfm_farina, hsfm_guo, hsfm_moussaid, 
# hsfm_new, hsfm_new_guo, hsfm_new_moussaid
motion_model = "orca"

## Decide wether to integrate with RKF45(True) or Euler(False)
runge_kutta = False

## Decide wether the robot must be considered by humans
robot_visible = True

## Decide wether to print a unitary metric grid in the background
grid = True

## Add walls by specifing its vertices (at least 3 vertices) - WRITE VERTICES IN COUNTER-CLOCKWISE ORDER
walls = [[[1,1], [1.5,1], [1.5,3], [1,3], [0.5,2]],
         [[3,9], [5,7], [6,9], [6,9.5], [3,9.5]],
         [[7,5], [9,5], [9,7], [7,5.5]]]

## Humans can be included by specifing various parameters, check src/human_agent.py
humans = {0: {"pos": [6.66,6.66], "yaw": -math.pi, "goals": [[5,5],[8,2]]},
          1: {"pos": [2.5,2.5], "yaw": 0.0, "goals": [[5,5],[8,2]]},
          2: {"pos": [2.0,5.0], "yaw": -math.pi, "goals": [[3,7],[8,8]], "group_id": 1},
          3: {"pos": [2.0,3.33], "yaw": 0.0, "goals": [[3,7],[8,8]], "group_id": 1, "radius": 0.4}}

robot = {"pos": [7.5,7.5], "yaw": 0.0, "radius": 0.25, "goals": [[0,0]]}

data = {"headless": headless, "motion_model": motion_model, "runge_kutta": runge_kutta, "robot_visible": robot_visible, "grid": grid, "humans": humans, "walls": walls, "robot": robot}