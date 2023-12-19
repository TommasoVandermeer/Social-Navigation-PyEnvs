import math
import numpy as np

def bound_angle(angle):
    if angle > math.pi: angle -= 2 * math.pi
    if angle < -math.pi: angle += 2 * math.pi
    return angle

def round_time(time):
    if time < 10.0: time = round(time, 3)
    elif ((time >= 10.0) and (time < 100.0)): time = round(time, 2)
    elif ((time >= 100.0) and (time < 1000.0)): time = round(time, 1)
    else: time = round(time)
    return time