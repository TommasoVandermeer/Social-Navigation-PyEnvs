import math
import numpy as np
from numba import njit

def bound_angle(angle):
    two_pi = 2 * math.pi
    if angle >= two_pi: angle %= two_pi # Wrap angle in [-360°,360°]
    if angle <= -two_pi: angle %= -two_pi # Wrap angle in [-360°,360°]
    if angle > math.pi: angle -= two_pi
    if angle < -math.pi: angle += two_pi
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

def is_multiple(number, dividend, tolerance=1e-7):
    mod = number % dividend
    return (abs(mod) <= tolerance) or (abs(dividend - mod) <= tolerance)

@njit(nogil=True)
def two_dim_norm(array:np.ndarray):
    return math.sqrt(array[0]**2 + array[1]**2)

@njit(nogil=True)
def two_dim_dot_product(array1:np.ndarray, array2:np.ndarray):
    return array1[0]*array2[0] + array1[1]*array2[1]

@njit(nogil=True)
def two_by_two_matrix_mul_two_dim_array(matrix:np.ndarray, array:np.ndarray):
    return np.array([matrix[0,0] * array[0] + matrix[0,1] * array[1], matrix[1,0] * array[0] + matrix[1,1] * array[1]], np.float64)

@njit(nogil=True)
def bound_two_dim_array_norm(array:np.ndarray, limit:np.float64):
    array_norm = two_dim_norm(array)
    if array_norm > limit: array = (array / array_norm) * limit
    return array

@njit(nogil=True)
def jitted_point_to_segment_distance(x1:np.float64, y1:np.float64, x2:np.float64, y2:np.float64, x3:np.float64, y3:np.float64):
    px = x2 - x1
    py = y2 - y1
    if px == 0 and py == 0: return two_dim_norm(np.array([x3-x1, y3-y1], np.float64))
    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)
    if u > 1: u = 1
    elif u < 0: u = 0
    x = x1 + u * px
    y = y1 + u * py
    return two_dim_norm(np.array([x - x3, y-y3], np.float64))

jitted_bound_angle = njit(nogil=True)(bound_angle)