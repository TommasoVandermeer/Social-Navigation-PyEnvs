import math

## Run pygame without GUI - WARNING: No event handling if true
headless = False

## Chose between sfm_roboticsupo, sfm_helbing, sfm_guo, sfm_moussaid, hsfm_farina, hsfm_guo, hsfm_moussaid, 
# hsfm_new, hsfm_new_guo, hsfm_new_moussaid
motion_model = "sfm_guo"

## Decide wether to integrate with RKF45(True) or Euler(False)
runge_kutta = False

## Decide wether the robot must be considered by humans
robot_visible = False

## Decide wether to print a unitary metric grid in the background
grid = True

## Add walls by specifing its vertices (at least 3 vertices)  - WRITE VERTICES IN COUNTER-CLOCKWISE ORDER
walls = [[[0.9,-2.5], [1.1,-2.5], [1.1,2.5], [0.9,2.5]],
         [[-0.9,-2.5], [-0.9,2.5], [-1.1,2.5], [-1.1,-2.5]],
         [[0,0], [0.9,0], [0.9,1], [0,1]]]

## Humans can be included by specifing various parameters, check src/human_agent.py
humans = {0: {"pos": [-0.2,5], "yaw": -math.pi, "goals": [[-0.2,-5],[-2,-1.25],[-2,1.25],[-0.2,5]]}}
# humans = {0: {"pos": [12.0,2.0], "yaw": -math.pi, "goals": [[3.0,13.0],[12.0,2.0]]}}
# humans = {0: {"pos": [3.0,2.0], "yaw": -math.pi, "goals": [[12.0,13.0],[3.0,2.0]]}}

data = {"headless": headless, "motion_model": motion_model, "runge_kutta": runge_kutta, "robot_visible": robot_visible, "grid": grid, "humans": humans, "walls": walls}