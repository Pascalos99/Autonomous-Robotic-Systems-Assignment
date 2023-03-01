import numpy as np
from numpy.random import uniform as U
import random
from random import random as rnd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from genetic_algorithm import *

# simple n=2
def init_numeric(min=-3, max=3):
    return min + (max - min) * random.random()
def mut_numeric(val, sigma=0.5):
    return val + random.gauss(0, sigma)
def cross_numeric(val1, val2):
    return (val1 + val2) / 2.
numeric = Parameter(init_numeric, mut_numeric, cross_numeric)
var2geno = Genotype(x=numeric, y=numeric)

def basic_n2_GA(fitness, popsize=100, init_x=(-3,3), init_y=(-3,3), avoid_asex=True, keep_old_population=True, elitist_percent=0.3,
             lucky_chance=0.1, sigma_x = 0.5, sigma_y = 0.5):
    GA = GeneticAlgorithm(fitness, var2geno, random_pairing, elitist_selection, population_size=popsize)
    GA.initialize(x={'min':init_x[0], 'max':init_x[1]}, y={'min':init_y[0], 'max':init_y[1]})
    GA.alter_pairing(avoid_asex=avoid_asex)
    GA.alter_crossover(keep_old_population=keep_old_population)
    GA.alter_selection(elitist_percent=elitist_percent, lucky_chance=lucky_chance)
    GA.alter_mutation(x={'sigma':sigma_x}, y={'sigma':sigma_y})
    return GA

if __name__ == '__main__':
    A, B = 0, 100
    rosenbrock_fitness = Fitness(lambda pop: (A - pop['x'])**2 + B * (pop['y'] - pop['x']**2)**2, minimize=True)
    rastrigin2_fitness = Fitness(lambda pop: 2 * 10 + ((pop['x']**2 - 10 * np.cos(2 * np.pi * pop['x'])) + (pop['y']**2 - 10 * np.cos(2 * np.pi * pop['y']))), minimize=True)
    GA_rosenbrock = basic_n2_GA(rosenbrock_fitness, popsize=100)
    GA_rastrigin2 = basic_n2_GA(rastrigin2_fitness, popsize=100)

    # Rosenbrock
    fitness = GA_rosenbrock.iterate(50)
    plt.plot([sum(f)/len(f) for f in fitness], label='average fitness')
    plt.plot([f[0] for f in fitness], label='top fitness')
    plt.legend()
    plt.title('Rosenbrock')
    plt.show()

    # Rastrigin
    fitness = GA_rastrigin2.iterate(50)
    plt.plot([sum(f)/len(f) for f in fitness], label='average fitness')
    plt.plot([f[0] for f in fitness], label='top fitness')
    plt.legend()
    plt.title('Rastrigin 2D')
    plt.show()

if __name__ == '__main__' and False:
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