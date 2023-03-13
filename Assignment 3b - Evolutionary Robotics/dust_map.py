import numpy as np
import random
import pygame
from bisect import bisect
from time import time

def within_circle(x1, y1, x2, y2, r):
    dx, dy = x1 - x2, y1 - y2
    return dx*dx + dy*dy <= r*r

class DustMap:
    def __init__(self, bounds=(0,0,1200,1200), regen_rate=0.0001, density=0.01, random_state=None):
        if type(random_state) is str:
            if random_state == 'None': random_state = None
            else: random_state = int(random_state)
        self.density = density
        self.regen_rate = regen_rate
        self.min_x, self.min_y, self.max_x, self.max_y = bounds
        self.particles = []
        self.removed = []
        self.reset_particles(random_state)
    
    def set_density(self, density=0.25):
        self.density = density

    def set_bounds(self, bounds=(0,0,1000,1000)):
        self.min_x, self.min_y, self.max_x, self.max_y = bounds
    
    def __sort(self):
        self.sort_x = sorted(self.particles.items(), key=lambda v: v[1][0])
        self.sort_y = sorted(self.particles.items(), key=lambda v: v[1][1])
        self.just_x, self.just_y = [v[1][0] for v in self.sort_x], [v[1][1] for v in self.sort_y]

    def reset_particles(self, random_state=None):
        np.random.seed(random_state)
        self.max_particles = int(round((self.max_x - self.min_x) * (self.max_y - self.min_y) * self.density))
        particle_x = np.random.uniform(self.min_x, self.max_x, (self.max_particles,))
        particle_y = np.random.uniform(self.min_y, self.max_y, (self.max_particles,))
        self.particles = {i:(particle_x[i], particle_y[i]) for i in range(self.max_particles)}
        self.removed = []
        self.__sort()

    def get_intersect(self, robot_x, robot_y, robot_r):
        # returns a list of intersected particles
        # each particle is a tuple (i, (x, y)) where i is the index of the particle in self.particles
        bsxL, bsxR = bisect(self.just_x, robot_x-robot_r), bisect(self.just_x, robot_x+robot_r)
        bsyL, bsyR = bisect(self.just_y, robot_y-robot_r), bisect(self.just_y, robot_y+robot_r)
        inx, iny = self.sort_x[bsxL:bsxR], self.sort_y[bsyL:bsyR]
        inxy = set.intersection(set(inx), iny)
        hits_i = {v[0] for v in inxy if within_circle(v[1][0], v[1][1], robot_x, robot_y, robot_r)}
        hits = set.difference(hits_i, self.removed)
        return [(i, self.particles[i]) for i in hits]
    
    def remove_particles(self, particles):
        # 'particles' is expected to be in the same format as the output of self.get_intersect
        self.removed = self.removed + [p[0] for p in particles]
    
    def regenerate(self, random_state=None):
        # 'exclude' is expected to be in the same format as the output of self.get_intersect
        np.random.seed(random_state)
        random.seed(random_state)
        gen = np.random.poisson(self.regen_rate * len(self.removed), 1)[0]
        regen = min(gen, len(self.removed))
        regenerate = random.sample(range(len(self.removed)), regen)
        self.removed = [self.removed[i] for i in set.difference(set(range(len(self.removed))), regenerate)]

    def draw_dust(self, surface: pygame.Surface, particle_size=1):
        draw_index = set.difference({k for k in self.particles.keys()}, self.removed)
        points = [self.particles[p] for p in draw_index]
        for p in points:
            x, y = p
            pygame.draw.circle(surface=surface, color=pygame.Color(120,120,60), center=(x, y), radius=particle_size)

if __name__ == '__main__':
    dm = DustMap()
    N = 1000
    r = 30
    robot_xs = np.random.uniform(20,980, (N,))
    robot_ys = np.random.uniform(20,980, (N,))
    print("starting...")
    start = time()
    for i in range(N):
        ps = dm.get_intersect(robot_xs[i], robot_ys[i], r)
        dm.regenerate()
        dm.remove_particles(ps)
    print(round(((time() - start) / N)*1000, 2), "ms")