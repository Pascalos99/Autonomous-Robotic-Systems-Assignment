import numpy as np
from numpy.random import uniform as U
from random import random as rnd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Try a simple iterative approach first

# Particle stores all necessary information to represent a single agent in the swarm
class Particle:
    def __init__(self, pos, vel, function: callable, abc: tuple, remember_global_best_of_all_time=False):
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
        self.remember_global_best_of_all_time = remember_global_best_of_all_time
    
    def update_global_best(self, gpos, gprf) -> None:
        if self.remember_global_best_of_all_time and self.gprf is not None:
            if self.gprf < gprf: return
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

def init_swarm(num_particles: int, function: callable, x_range: tuple, x_shape: tuple, abc: tuple, **kwargs) -> list:
    swarm = []
    x_min, x_max = x_range
    for i in range(num_particles):
        pos = U(x_min, x_max, x_shape)
        swarm.append(Particle(pos,
            U(x_min - x_max, x_max - x_min, x_shape),
            function, abc, **kwargs))
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

def get_fixed_neighborhoods(swarm: list, num_neighborhoods: int) -> dict:
    if num_neighborhoods is None: num_neighborhoods = 5
    # this function encompass a fixed neighborhood relation between particles (no matter their distance)
    neighborhoods = [[] for i in range(num_neighborhoods)]
    for i in range(len(swarm)):
        neighborhoods[i%num_neighborhoods].append(swarm[i])
    neighbormatrix = {}
    for ngb in neighborhoods:
        for part in ngb:
            neighbormatrix[part] = ngb
    return neighbormatrix

def PSO(grapher: callable, swarm: list, num_iters: int, dt: float, neighbor_setting="exclusive_global", neighbor_param=None):
    # neighbor_param is 'number of fixed neighborhoods' OR 'neighbor_range', depending on neighbor_setting

    ########################################################################
    #                           SETUP NEIGHBORS                            #
    ########################################################################

    best_particle, almost_best_particle = None, None

    def inclusive_global_neighbor_function(_) -> Particle:
        return best_particle
    def exclusive_global_neighbor_function(particle: Particle) -> Particle:
        if particle is best_particle: return almost_best_particle
        return best_particle
    
    def get_inclusive_fixed_neighbor_function(neighbormatrix: dict) -> callable:
        def inclusive_fixed_neighbor_function(particle: Particle) -> Particle:
            return get_best_of_swarm(neighbormatrix[particle])
        return inclusive_fixed_neighbor_function
    
    def get_exclusive_fixed_neighbor_function(neighbormatrix: dict) -> callable:
        def exclusive_fixed_neighbor_function(particle: Particle) -> Particle:
            best = get_best_of_swarm(neighbormatrix[particle])
            if particle is best: return get_2nd_best_of_swarm(neighbormatrix[particle], best)
            return best
        return exclusive_fixed_neighbor_function
    
    setting_map = {
        "inclusive_global": inclusive_global_neighbor_function,
        "exclusive_global": exclusive_global_neighbor_function,
        "inclusive_fixed": get_inclusive_fixed_neighbor_function(get_fixed_neighborhoods(swarm, neighbor_param)),
        "exclusive_fixed": get_exclusive_fixed_neighbor_function(get_fixed_neighborhoods(swarm, neighbor_param))
    }
    neighbor_function = setting_map[neighbor_setting]
    
    ########################################################################
    #                          PERFORM ITERATIONS                          #
    ########################################################################

    for iter in range(num_iters):
        # update global best:
        best_particle = get_best_of_swarm(swarm)
        almost_best_particle = get_2nd_best_of_swarm(swarm, best_particle)

        swarm_iteration(swarm, neighbor_function, dt)
        grapher(xs = [p.pos for p in swarm], ys = [p.prf for p in swarm])
        
        for p in swarm:
            p.a -= ((0.9 - 0.4) / num_iters)
        # print(swarm[0].a)

if __name__ == '__main__':
    xss = []
    yss = []
    def grapher(xs, ys):
        xss.append(xs)
        yss.append(ys)

    # DEFINE FUNCTION
    A, B = 0, 100
    rosenbrock_func = lambda x: (A - x[0])**2 + B * (x[1] - x[0]**2)**2
    rastrigin_func = lambda x: 2 * 10 + ((x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1])))
    # both functions seem to work perfectly fine; getting to (0,0) in about ~50 steps

    # PARAMETERS -  you can change these!
    x_range = (-3, 3)
    init_x_range = x_range
    x_shape = (2,)
    abc = (0.9, 2., 2.)
    num_particles = 20
    num_iters = 100
    dt = 0.1
    neighbor_protocol = "exclusive_global"
    # choose from: "inclusive_global", "exclusive_global", "inclusive_fixed", "exclusive_fixed"
    num_neighborhoods = None
    func = rosenbrock_func
    particle_kwargs = {"remember_global_best_of_all_time": False} #True} # not sure if this is good?

    # RUN SIMULATION
    swarm = init_swarm(num_particles, func, init_x_range, x_shape, abc, **particle_kwargs)
    PSO(grapher, swarm, num_iters, dt, neighbor_protocol, num_neighborhoods)

    # X, Y needed for the benchmark function
    x = np.arange(*x_range, 0.025)
    y = np.arange(*x_range, 0.025)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots()

    # Rosenbrock vmax=1000, Rastrigin vmax=60
    contour = ax.contourf(X, Y, func((X, Y)), 250, vmin=0, vmax=1000, cmap='jet') 
    ax.set_xlim(x_range)
    ax.set_ylim(x_range)
    ax.set_title("Benchmark function: Rosenbrock")
    ax.set_xlabel("x values")
    ax.set_ylabel("y values")
    scat = ax.scatter([x[0] for x in xss[0]], [x[1] for x in xss[0]], c='w', marker="*")

    def animate(i):
        scat.set_offsets(xss[i])
        return scat, 

    ani = animation.FuncAnimation(fig, animate, frames=len(xss), interval=100, blit=True)
    # FFwriter = animation.FFMpegWriter(fps=10)
    # ani.save('rastrigin.mp4', writer=FFwriter)
    
    plt.show()