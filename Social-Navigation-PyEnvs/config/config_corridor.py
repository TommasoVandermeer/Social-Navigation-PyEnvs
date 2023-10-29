import math

## Run pygame without GUI - WARNING: No event handling if true
headless = False

## Chose between sfm_roboticsupo, sfm_helbing, sfm_guo, sfm_moussaid, hsfm_farina, hsfm_guo, hsfm_moussaid, 
# hsfm_new, hsfm_new_guo, hsfm_new_moussaid
motion_model = "hsfm_farina"

## Decide wether to integrate with RKF45(True) or Euler(False)
runge_kutta = False

## Decide wether to insert the robot in the simulation
insert_robot = False

## Decide wether to print a unitary metric grid in the background
grid = True

## Add walls by specifing its vertices (at least 3 vertices)
walls = [[[0.9+7.5,-2.5+7.5], [1.1+7.5,-2.5+7.5], [1.1+7.5,2.5+7.5], [0.9+7.5,2.5+7.5]],
         [[-0.9+7.5,-2.5+7.5], [-1.1+7.5,-2.5+7.5], [-1.1+7.5,2.5+7.5], [-0.9+7.5,2.5+7.5]],
         [[0+7.5,0+7.5], [0.9+7.5,0+7.5], [0.9+7.5,1+7.5], [0+7.5,1+7.5]]]

## Humans can be included by specifing various parameters, check src/human_agent.py
humans = {0: {"pos": [-0.2+7.5,5+7.5], "yaw": -math.pi, "goals": [[-0.2+7.5,-5+7.5],[-2+7.5,-1.25+7.5],[-2+7.5,1.25+7.5],[-0.2+7.5,5+7.5]]}}

data = {"headless": headless, "motion_model": motion_model, "runge_kutta": runge_kutta, "insert_robot": insert_robot, "grid": grid, "humans": humans, "walls": walls}