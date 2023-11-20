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

def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))