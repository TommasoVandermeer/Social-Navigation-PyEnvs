import pygame
import math

def bound_angle(angle):
    if angle > math.pi: angle -= 2 * math.pi
    if angle < -math.pi: angle += 2 * math.pi
    return angle

def points_distance(point1:list[float], point2:list[float]):
    return math.sqrt(((point2[0] - point1[0]) ** 2) + ((point2[1] - point1[1]) ** 2))

def points_unit_vector(point1:list[float], point2:list[float]):
    dist = points_distance(point1, point2)
    diff = [point1[0]-point2[0],point1[1]-point2[1]]
    return [diff[0]/dist, diff[1]/dist]

def vector_angle(vector:list[float]):
    return math.atan2(vector[1],vector[0])

def vector_difference(vector1:list[float], vector2:list[float]):
    return [vector2[0] - vector1[0], vector2[1] - vector1[1]]

def normalized(vector:list[float]):
    norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    vector_normalized = vector
    vector_normalized[0] /= norm
    vector_normalized[1] /= norm
    return vector_normalized