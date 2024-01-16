import numpy as np
import math
from social_gym.src.human_agent import HumanAgent

class LaserSensor:
    """
    This class is used to simulate a laser range finder.
    """
    def __init__(self, init_pos:np.array, init_yaw:float, range:float, samples:int, max_distance:float, uncertainty=None):
        self.range = range
        self.samples = samples
        if max_distance > 10: raise ValueError("Maxium distance for laser is 10 meters")
        else: self.max_distance = max_distance
        if uncertainty is not None: self.uncertainty = uncertainty
        # Pose of the sensor
        self.update_pose(init_pos,init_yaw)

    def update_pose(self, position:np.array, yaw:float):
        if yaw > math.pi or yaw < - math.pi: raise ValueError("Angle passed ust be wrapped between [-pi,pi]")
        self.position = position
        self.yaw = yaw

    def sphere_ray_intersect(self, ray_direction:np.array, sphere_center:np.array, sphere_radius:float):
        s = self.position - sphere_center
        b = np.dot(s, ray_direction)
        c = np.dot(s,s) - (sphere_radius * sphere_radius)
        h = b * b - c
        if h < 0.0: return self.max_distance
        h = np.sqrt(h)
        t = - b - h
        if t < 0.0: return self.max_distance
        return np.min([t,self.max_distance])
    
    def segment_ray_intersect(self, ray_direction:np.array, segment_vertices:list[list[float]]):
        x1 = segment_vertices[0][0]
        y1 = segment_vertices[0][1]
        x2 = segment_vertices[1][0]
        y2 = segment_vertices[1][1]
        x3 = self.position[0]
        y3 = self.position[1]
        x4 = self.position[0] + ray_direction[0]
        y4 = self.position[1] + ray_direction[1]
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator <= 0.0: return self.max_distance
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = - ((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
        if 0 < t < 1 and u > 0: 
            intersection_point = np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dtype=np.float64)
            return np.min([np.linalg.norm(self.position - intersection_point),self.max_distance])
        else: return self.max_distance

    def get_laser_measurements(self, humans:list[HumanAgent], walls):
        # Warning: angles are not wrapped between -pi and pi
        angles = np.linspace( self.yaw - (self.range/2), self.yaw + (self.range/2), self.samples)
        measurements = []
        for angle in angles:
            measurement = self.max_distance
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
            for human in humans: 
                raycast = self.sphere_ray_intersect(direction, human.position, human.radius)
                if raycast < measurement: measurement = raycast
            for wall in walls:
                for segment in wall.segments.values():
                    raycast = self.segment_ray_intersect(direction, segment)
                    if raycast < measurement: measurement = raycast
            if self.uncertainty is not None: measurement = self.add_uncertainty(measurement)
            measurements.append(measurement)
        return dict(zip(angles, measurements))

    def add_uncertainty(self, measurement:float):
        measurement = np.random.normal(measurement, self.uncertainty)
        measurement = max(min(measurement, self.max_distance), 0)
        return measurement
