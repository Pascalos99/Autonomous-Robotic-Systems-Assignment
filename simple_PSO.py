import math
import numpy as np
from numpy.random import uniform as U
from random import random as rnd

# Try a simple iterative approach first:
class Particle:
    def __init__(self, pos, vel, prf, abc):
        self.pos = pos
        self.vel = vel
        self.prf = prf
        self.a, self.b, self.c = abc
        self.pbest = pos
        self.gbest = pos
    
    def update_velocity(self) -> None:
        # v := a * v + b * Rb * (pos_pbest - pos) + c * Rc * (pos_gbest - pos)
        self.vel = self.a * self.vel + self.b * rnd() * (self.pbest - self.pos) + self.c * rnd() * (self.gbest - self.pos)

    def update_position(self, dt: float) -> None:
        # simple euclidean position update
        self.pos = self.pos + self.vel * dt

class Problem:
    def __init__(self, function: function, x_shape: tuple):
        self.function = function
        self.x_shape = x_shape

def init_swarm(num_particles: int, x_range: tuple, prb: Problem) -> list:
    particles = []
    x_min, x_max = x_range
    for i in range(num_particles):
        pos = U([x_min, x_max, prb.x_shape])
        particles.append(Particle(pos,
            U([0, 1, prb.x_shape]),
            prb.function(pos)))
    return particles

def swarm_iteration(swarm: list, prb: Problem):
    pass

def PSO(num_particles: int, x_range, func):
    pass