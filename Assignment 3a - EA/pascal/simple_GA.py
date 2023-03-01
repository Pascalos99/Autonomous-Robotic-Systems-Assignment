import numpy as np
from numpy.random import uniform as U
import random
from random import random as rnd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from genetic_algorithm import Parameter, Genotype, Fitness, GeneticAlgorithm
        
if __name__ == '__main__':
    print('printing pop 1')
    def init_numeric(l=100):
        return int(l * random.random())
    numeric = Parameter(init_numeric, lambda x: x + int(random.random()*10), lambda x, y: (x + y)//2)
    simplegeno = Genotype(x=numeric, y=numeric)
    simplegeno.alter_init(x={'l':200})
    pop1 = simplegeno.get_population(10)
    pop2 = simplegeno.get_population(5)
    print(pop1)
    print("now pop 2")
    print(pop2)
    print("now joined")
    print(simplegeno.join_pops(pop1, pop2))
    print("now crossover pop 1")
    pop3 = simplegeno.crossover(pop1, [(0,1), (1,2), (2,3)])
    print(pop3)
    simplegeno.mutate_all(pop3)
    print("now mutated:")
    print(pop3)

if __name__ == '__main__' and False:
    xss, yss = [], []
    
    # DEFINE FUNCTION
    A, B = 0, 100
    rosenbrock_func = lambda x: (A - x[0])**2 + B * (x[1] - x[0]**2)**2
    rastrigin_func = lambda x: 2 * 10 + ((x[0]**2 - 10 * np.cos(2 * np.pi * x[0])) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1])))

    # PARAMETERS -  you can change these!
    x_range = (-3, 3)
    # init_x_range = x_range
    x_shape = (2,)
    popsize = 100
    num_gens = 50
    
    # SET FUNCTION TO OPTIMIZE
    func = rastrigin_func

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