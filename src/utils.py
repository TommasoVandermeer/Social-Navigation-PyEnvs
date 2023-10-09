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