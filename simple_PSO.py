import numpy as np
from numpy.random import uniform as U
from random import random as rnd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Try a simple iterative approach first

# Particle stores all necessary information to represent a single agent in the swarm
class Particle:
    def __init__(self, pos, vel, function: callable, abc: tuple):
        self.update_listeners = []

        self.pos = pos
        self.vel = vel
        self.prf = function(pos)
        self.a, self.b, self.c = abc
        self.pbest = self.pos
        self.pprf = self.prf
        self.gbest = None
        self.gprf = None
        self.function = function
    
    def update_global_best(self, gpos, gprf) -> None:
        self.gbest = gpos
        self.gprf = gprf

    def update_velocity(self) -> None:
        # v := a * v + b * Rb * (pos_pbest - pos) + c * Rc * (pos_gbest - pos)
        self.vel = self.a * self.vel + self.b * rnd() * (self.pbest - self.pos) + self.c * rnd() * (self.gbest - self.pos)

    def update_position(self, dt: float) -> None:
        # simple euclidean position update
        self.pos = self.pos + self.vel * dt
        self.prf = self.function(self.pos)
        if self.prf < self.pprf:
            self.pbest = self.pos
            self.pprf = self.prf

def init_swarm(num_particles: int, function: callable, x_range: tuple, x_shape: tuple, abc: tuple) -> list:
    swarm = []
    x_min, x_max = x_range
    for i in range(num_particles):
        pos = U(x_min, x_max, x_shape)
        swarm.append(Particle(pos,
            U(x_min - x_max, x_max - x_min, x_shape),
            function, abc))
    return swarm

def swarm_iteration(swarm: list, neighbor_function: callable, dt: float) -> None:
    for particle in swarm:
        best_neighbor = neighbor_function(particle)
        particle.update_global_best(best_neighbor.pos, best_neighbor.prf)
        particle.update_velocity()

    for particle in swarm:
        particle.update_position(dt)

def get_best_of_swarm(swarm: list) -> Particle:
    best_particle = swarm[0]
    for particle in swarm[1:]:
        if particle.prf < best_particle.prf:
            best_particle = particle
    return best_particle

def get_2nd_best_of_swarm(swarm: list, best_particle: Particle) -> Particle:
    almost_best_particle = None
    for particle in swarm:
        if particle is best_particle: continue
        if almost_best_particle is None or particle.prf < almost_best_particle.prf:
            almost_best_particle = particle
    return almost_best_particle

def PSO(grapher: callable, swarm: list, num_iters: int, dt: float):

    ########################################################################
    #                           SETUP NEIGHBORS                            #
    ########################################################################

    best_particle, almost_best_particle = None, None

    def inclusive_global_neighbor_function(_) -> Particle:
        return best_particle
    def exclusive_global_neighbor_function(particle: Particle) -> Particle:
        if particle is best_particle: return almost_best_particle
        return best_particle
    
    def get_fixed_neighborhoods(num_neighborhoods: int) -> dict:
        # this function encompass a fixed neighborhood relation between particles (no matter their distance)
        neighborhoods = [[] for i in range(num_neighborhoods)]
        for i in range(len(swarm)):
            neighborhoods[i%num_neighborhoods].append(swarm[i])
        neighbormatrix = {}
        for ngb in neighborhoods:
            for part in ngb:
                neighbormatrix[part] = ngb
        return neighbormatrix
    
    def get_inclusive_fixed_neighbor_function(neighbormatrix: dict) -> callable:
        def inclusive_fixed_neighbor_function(particle: Particle) -> Particle:
            return get_best_of_swarm(neighbormatrix[particle])
        return inclusive_fixed_neighbor_function
    
    def get_exclusive_fixed_neighbor_function(neighbormatrix: dict) -> callable:
        def exclusive_fixed_neighbor_function(particle: Particle) -> Particle:
            best = get_best_of_swarm(neighbormatrix[particle])
            if particle is best: return get_2nd_best_of_swarm(neighbormatrix[particle], best)
        return exclusive_fixed_neighbor_function
    
    neighbor_function = exclusive_global_neighbor_function
    
    ########################################################################
    #                          PERFORM ITERATIONS                          #
    ########################################################################

    for iter in range(num_iters):
        # update global best:
        best_particle = get_best_of_swarm(swarm)
        almost_best_particle = get_2nd_best_of_swarm(swarm, best_particle)

        swarm_iteration(swarm, neighbor_function, dt)
        grapher(xs = [p.pos for p in swarm], ys = [p.prf for p in swarm])

if __name__ == '__main__':
    xss = []
    def grapher(xs, ys):
        xss.append(xs)

    # DEFINE FUNCTION
    A, B = 0, 100
    rosenbrock_func = lambda x: (A - x[0])**2 + B * (x[1] - x[0]**2)**2
    rastrigin_func = lambda x: 2 * 10 + ((x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1])))
    # both functions seem to work perfectly fine; getting to (0,0) in about ~50 steps

    x_range = (-3, 3)
    x_shape = (2,)
    abc = (0.9, 2., 2.)
    num_particles = 20
    num_iters = 100
    dt = 0.1
    func = rastrigin_func # only change this
    swarm = init_swarm(num_particles, func, x_range, x_shape, abc)
    PSO(grapher, swarm, num_iters, dt)


    # X, Y needed for the benchmark function
    x = np.arange(*x_range, 0.025)
    y = np.arange(*x_range, 0.025)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots()

    contour = ax.contourf(X, Y, func((X, Y)), 250, vmin=0, vmax=60, cmap=mpl.colormaps['jet'])
    ax.set_xlim(x_range)
    ax.set_ylim(x_range)
    ax.set_title("Benchmark function: Rastrigin")
    ax.set_xlabel("x values")
    ax.set_ylabel("y values")
    scat = ax.scatter([x[0] for x in xss[0]], [x[1] for x in xss[0]], c='w', marker="*")

    def animate(i):
        scat.set_offsets(xss[i])
        return scat, 

    ani = animation.FuncAnimation(fig, animate, frames=len(xss), interval=100, blit=True)
    FFwriter = animation.FFMpegWriter(fps=10)
    ani.save('rastrigin.mp4', writer=FFwriter)
    
    plt.show()