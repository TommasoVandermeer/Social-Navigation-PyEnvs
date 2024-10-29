
## Run pygame without GUI - WARNING: No event handling if true
headless = False

## Chose between sfm_roboticsupo, sfm_helbing, sfm_guo, sfm_moussaid, hsfm_farina, hsfm_guo, hsfm_moussaid, 
# hsfm_new, hsfm_new_guo, hsfm_new_moussaid
motion_model = "hsfm_new_guo"

## Decide wether to integrate with RKF45(True) or Euler(False)
runge_kutta = False

## Decide wether the robot must be considered by humans
robot_visible = True

## Decide wether to print a unitary metric grid in the background
grid = True

## Add walls by specifing its vertices (at least 3 vertices) - WRITE VERTICES IN COUNTER-CLOCKWISE ORDER
walls = []

## Humans can be included by specifing various parameters, check src/human_agent.py
humans = {0: {"pos": [-1.7743506, 7.1795807], "yaw": 4.954673, "goals": [[1.7743506, -7.1795807]]},
          1: {"pos": [-6.1466966, -3.6587288], "yaw": 6.820094, "goals": [[6.1466966, 3.6587288]]},
          2: {"pos": [-4.7714133, -5.1887245], "yaw": 7.1104574, "goals": [[4.7714133, 5.1887245]]},
          3: {"pos": [-2.3724763, -7.0859675], "yaw": 7.5309, "goals": [[2.3724763, 7.0859675]]},
          4: {"pos": [7.1258082, -0.7462779], "yaw": 9.32043, "goals": [[-7.1258082, 0.7462779]]}}

robot = {"pos": [0., -7.], "yaw": 1.5707964, "radius": 0.25, "goals": [[0., 7.]]}

data = {"headless": headless, "motion_model": motion_model, "runge_kutta": runge_kutta, "robot_visible": robot_visible, "grid": grid, "humans": humans, "walls": walls, "robot": robot}

# [[-1.7743506  7.1795807  0.         0.         4.954673   0.       ]
#  [-6.1466966 -3.6587288  0.         0.         6.820094   0.       ]
#  [-4.7714133 -5.1887245  0.         0.         7.1104574  0.       ]
#  [-2.3724763 -7.0859675  0.         0.         7.5309     0.       ]
#  [ 7.1258082 -0.7462779  0.         0.         9.32043    0.       ]
#  [ 0.        -7.         0.         0.         1.5707964  0.       ]]