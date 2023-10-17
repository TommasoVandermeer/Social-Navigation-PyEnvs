import math
import os
from random import randint

PKG_DIR = os.path.join(os.path.expanduser('~'),'Repos/Social-Navigation-PyEnvs/Social-Navigation-PyEnvs')
CONFIG_DIR = os.path.join(PKG_DIR,'config/')

def bound_angle(angle):
    if (angle > math.pi): angle -= 2 * math.pi
    if (angle < -math.pi): angle += 2 * math.pi
    return angle

def write_config(model_title:str, init_pos:list, init_yaw:list, goal:list, n_actors:int, title:str):
    data = (f"motion_model = '{model_title}'\n" +
            "\n" +
            "insert_robot = False\n" +
            "\n" +
            "grid = True\n" +
            "\n" +
            "walls = []\n" +
            "\n" +
            "humans = {")
    ocb = "{"
    ccb = "}"
    for i in range(n_actors):
        if i == 0: data += f"{i}: {ocb}'pos': [{init_pos[i][0]},{init_pos[i][1]}], 'yaw': {init_yaw[i]}, 'goals': [[{goal[i][0]},{goal[i][1]}],[{init_pos[i][0]},{init_pos[i][1]}]]{ccb},\n"; continue
        if i < n_actors - 1: data += f"          {i}: {ocb}'pos': [{init_pos[i][0]},{init_pos[i][1]}], 'yaw': {init_yaw[i]}, 'goals': [[{goal[i][0]},{goal[i][1]}],[{init_pos[i][0]},{init_pos[i][1]}]]{ccb},\n"
        else: data += f"          {i}: {ocb}'pos': [{init_pos[i][0]},{init_pos[i][1]}], 'yaw': {init_yaw[i]}, 'goals': [[{goal[i][0]},{goal[i][1]}],[{init_pos[i][0]},{init_pos[i][1]}]]{ccb}{ccb}\n"
    data += ("\n" + 
             "def initialize():\n" +
             "  return walls, humans, motion_model, insert_robot, grid")
    
    with open(f'{CONFIG_DIR}{title}.py', 'w') as file:
            file.write(data)

def main():
    radius = int(input("Specify the desired radius of the circular workspace (3, 4, 5, 6, or 7):\n"))
    n_actors = int(input("Specify the number of actors to insert in the experiment:\n"))
    model = int(input("Specify the model that guides human motion (0: sfm_helbing, 1: sfm_robotics_upo):\n"))
    rand = bool(int(input("Specify if you want the actors randomly positioned (0: False, 1: True):\n")))

    ### COMPUTATIONS
    if (model == 0): model_title = "sfm_helbing"
    elif (model == 1): model_title = "sfm_roboticsupo"
    if (not rand): title = f"config_circular_crossing_{model_title}_{n_actors}_{radius}m"
    else: title = f"config_circular_crossing_{model_title}_{n_actors}_{radius}m"
    dist_center = radius - 0.5

    init_pos = []
    goal = []
    init_yaw = []

    center = [7.5,7.5]

    if (not rand):
        arch = (2 * math.pi) / (n_actors)

        for i in range(n_actors):
            init_pos.append([round(dist_center * math.cos(arch * i),4), round(dist_center * math.sin(arch * i),4)])
            init_yaw.append(round(bound_angle(-math.pi + arch * i),4))
            goal.append([-init_pos[i][0],-init_pos[i][1]])
        
        for i in range(n_actors):
            init_pos[i][0] += center[0]
            init_pos[i][1] += center[1]
            goal[i][0] += center[0]
            goal[i][1] += center[1]
    else:
        arch = (2 * math.pi) * 0.6 / (2 * radius * math.pi)
        rand_nums = []

        for i in range(n_actors):
            if (i == 0):
                init_pos.append([round(dist_center * math.cos(0.0),4), round(dist_center * math.sin(0.0),4)])
                init_yaw.append(round(bound_angle(-math.pi),4))
                goal.append([-init_pos[0][0],-init_pos[0][1]])
            else:
                check = False
                num = 0
                while (not check):
                    num = randint(2,round(((2 * radius * math.pi) - 2)/ 0.6))
                    if (num in rand_nums): continue
                    else: check = True
                rand_nums.append(num)
                init_yaw.append(round(bound_angle(-math.pi + arch * num),4))
                init_pos.append([round(dist_center * math.cos(arch * num),4), round(dist_center * math.sin(arch * num),4)])
                goal.append([-init_pos[i][0],-init_pos[i][1]])

        for i in range(n_actors):
            init_pos[i][0] += center[0]
            init_pos[i][1] += center[1]
            goal[i][0] += center[0]
            goal[i][1] += center[1]

    ## GENERATE CONFIG
    write_config(model_title, init_pos, init_yaw, goal, n_actors, title)

    print(f"Generation successfully completed!\n")

if __name__ == "__main__":
    main()