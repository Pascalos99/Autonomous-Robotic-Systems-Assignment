import numpy as np
from numpy.random import uniform as U
import random
from random import random as rnd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

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

if __name__ == '__main__':
    xss, yss = [], []
    def grapher(xs, ys):
        xss.append(xs)
        yss.append(ys)
    
    # DEFINE FUNCTION
    A, B = 0, 100
    rosenbrock_func = lambda x: (A - x[0])**2 + B * (x[1] - x[0]**2)**2
    rastrigin_func = lambda x: 2 * 10 + ((x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1])))

    # PARAMETERS -  you can change these!
    x_range = (-3, 3)
    init_x_range = x_range
    x_shape = (2,)
    abc = (0.9, 2., 2.)
    num_particles = 20
    num_iters = 100
    dt = 0.1
    neighbor_protocol = "exclusive_global" # choose from: "inclusive_global", "exclusive_global", "inclusive_fixed", "exclusive_fixed"
    num_neighborhoods = None
    particle_kwargs = {"remember_global_best_of_all_time": False} #True} # not sure if this is good?
    
    # Set Benchmark Function
    func = rosenbrock_func

    # RUN SIMULATION
    swarm = init_swarm(num_particles, func, init_x_range, x_shape, abc, **particle_kwargs)
    PSO(grapher, swarm, num_iters, dt, neighbor_protocol, num_neighborhoods)



    # ANIMATION
    # Create Benchmark Function
    x = np.arange(*x_range, 0.025)
    y = np.arange(*x_range, 0.025)
    X, Y = np.meshgrid(x, y)

    # Create Figure, Grid and subplots
    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(2, 3)
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    ax_global_best = fig.add_subplot(gs[0, 2])
    ax_all_perf = fig.add_subplot(gs[1, 2])

    # Rosenbrock vmax=1000, Rastrigin vmax=60
    if func == rosenbrock_func:
        contour = ax_main.contourf(X, Y, func((X, Y)), 300, vmin=0, vmax=1000, cmap='jet') 
        ax_main.set_title("Benchmark function: Rosenbrock")
    else:
        contour = ax_main.contourf(X, Y, func((X, Y)), 200, vmin=0, vmax=60, cmap='jet')
        ax_main.set_title("Benchmark function: Rastrigin")
        
    ax_main.set_xlim(x_range)
    ax_main.set_ylim(x_range)
    ax_main.set_xlabel("x values")
    ax_main.set_ylabel("y values")
    scat = ax_main.scatter([x[0] for x in xss[0]], [x[1] for x in xss[0]], c='w', marker="*")

    ax_global_best.set_xlim(0, num_iters)
    ax_global_best.set_ylim(-0.1, min(yss[0]))
    ax_global_best.set_title("Global Best Performance")
    ax_global_best.set_xlabel("Iteration")
    ax_global_best.set_ylabel("Performance Score")
    line_global_data = []
    line_global_best,  = ax_global_best.plot([], [], c='#00ee00', lw=2)

    ax_all_perf.set_xlim(0, num_iters)
    ax_all_perf.set_ylim(-0.1, 120)
    ax_all_perf.set_title("All Particle Performance")
    ax_all_perf.set_xlabel("Iteration")
    ax_all_perf.set_ylabel("Performance Score")
    line_all_data = [[] for _ in range(num_particles)]
    # color gen from https://stackoverflow.com/questions/13998901/generating-a-random-hex-color-in-python
    all_lines = [ax_all_perf.plot([], [], c='#'+'%06x' % random.randint(0, 0xFFFFFF), alpha=0.5)[0] for _ in range(num_particles)]
    
    plt.tight_layout()

    def animate_main(i):
        scat.set_offsets(xss[i])
        return scat,
    
    def animate_global_best(i):
        line_global_data.append(min(yss[i]))
        line_global_best.set_data([index for index, _ in enumerate(line_global_data)], line_global_data)
        if max(line_global_data) > ax_global_best.get_ylim()[1]:
            ax_global_best.set_ylim(-0.1, max(line_global_data))
        return line_global_best, 
    
    def animate_all_perf(i, lines):
        for j in range(len(lines)):
            line_all_data[j].append(yss[i][j])
            lines[j].set_data([index for index, _ in enumerate(line_all_data[j])], line_all_data[j])
        return lines
    
    # ANIMATION SPEED
    interval = 100
    ani_main = animation.FuncAnimation(fig, animate_main, frames=len(xss), interval=interval)
    ani_global_best = animation.FuncAnimation(fig, animate_global_best, frames=len(xss), interval=interval)
    ani_all_perf = animation.FuncAnimation(fig, animate_all_perf, fargs=([all_lines]), frames=len(xss), interval=interval)
    plt.show()