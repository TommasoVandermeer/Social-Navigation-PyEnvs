import pygame
import math

def bound_angle(angle):
    if angle > math.pi: angle -= 2 * math.pi
    if angle < -math.pi: angle += 2 * math.pi
    return angle