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
walls = [[[1-7.5,1-7.5], [1.5-7.5,1-7.5], [1.5-7.5,3-7.5], [1-7.5,3-7.5], [0.5-7.5,2-7.5]],
         [[3-7.5,9-7.5], [5-7.5,7-7.5], [6-7.5,9-7.5], [6-7.5,9.5-7.5], [3-7.5,9.5-7.5]],
         [[7-7.5,5-7.5], [9-7.5,5-7.5], [9-7.5,7-7.5], [7-7.5,5.5-7.5]]]

## Humans can be included by specifing various parameters, check src/human_agent.py
humans = {0: {"pos": [6.66-7.5,6.66-7.5], "yaw": -math.pi, "goals": [[5-7.5,5-7.5],[8-7.5,2-7.5]]},
          1: {"pos": [2.5-7.5,2.5-7.5], "yaw": 0.0, "goals": [[5-7.5,5-7.5],[8-7.5,2-7.5]]},
          2: {"pos": [2.0-7.5,5.0-7.5], "yaw": -math.pi, "goals": [[3-7.5,7-7.5],[8-7.5,8-7.5]], "group_id": 1},
          3: {"pos": [2.0-7.5,3.33-7.5], "yaw": 0.0, "goals": [[3-7.5,7-7.5],[8-7.5,8-7.5]], "group_id": 1, "radius": 0.4}}

robot = {"pos": [7.5-7.5,7.5-7.5], "yaw": 0.0, "radius": 0.25, "goals": [[0-7.5,0-7.5]]}

data = {"headless": headless, "motion_model": motion_model, "runge_kutta": runge_kutta, "robot_visible": robot_visible, "grid": grid, "humans": humans, "walls": walls, "robot": robot}