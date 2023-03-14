import configparser
import pathlib

from matplotlib import pyplot as plt

from genetic_algorithm import Fitness
from neural_GA import ANN, sigmoid, tanh, get_ANN_GA
from simulation import default_ann_bridge, get_recurrent_ann_bridge, default_fitness_func
from barebones_simulation import get_average_fitness
import genetic_algorithm as ga
import neural_GA as nga
import random

working_directory = pathlib.Path(__file__).parent.absolute()
config = configparser.ConfigParser()
config.read(f"{working_directory}/config.ini")
training_maps = [f"train/map_{i}.json" for i in range(1,19)]
testing_maps = [f"map_{i}.json" for i in range(19,26)]

# mu_init=numeric, sigma_init=numeric, avoid_asex=boolean, fitness_weighting=boolean, elite_sel=numeric, lucky_sel=numeric,
# elite_sur=numeric, lucky_sur=numeric, mut_rate=numeric, mut_chance=numeric, mut_sigma=numeric
def meta_geno():
    def init_numeric(mu=0, sigma=0.5, can_be_negative=False):
        val = random.normalvariate(mu, sigma)
        if not can_be_negative: return max(0, val)
        return val
    def muta_numeric(val, sigma=0.5, can_be_negative=False):
        valn = val + random.normalvariate(0, sigma)
        if not can_be_negative: return max(0, valn)
        return valn
    def cros_numeric(val1, val2, fitness1=1., fitness2=1.):
        f1 = sigmoid(fitness1 - fitness2)
        return f1 * val1 + (1 - f1) * val2
    numeric = ga.Parameter(init_numeric, muta_numeric, cros_numeric)

    def muta_boolean(val, mut_rate=0.25):
        if random.random() < mut_rate:
            return not val
        return val
    def cros_boolean(val1, val2, fitness1=1., fitness2=1.):
        if val1 == val2: return val1
        f1 = sigmoid(fitness1 - fitness2)
        if random.random() < f1: return val1
        return val2
    boolean = ga.Parameter(lambda: random.random() < 0.5, muta_boolean, cros_boolean)
    return ga.Genotype(mu_init=numeric, sigma_init=numeric, avoid_asex=boolean, fitness_weighting=boolean, elite_sel=numeric,
                       lucky_sel=numeric, elite_sur=numeric, lucky_sur=numeric, mut_rate=numeric, mut_chance=numeric, mut_sigma=numeric)

def get_GA_from_meta(meta_ind, fitness, geno, popsize=20):
    GA = ga.GeneticAlgorithm(fitness, geno, ga.random_pairing, ga.elitist_selection, ga.elitist_selection,
                             population_size=popsize)
    GA.initialize(ann={'mu': meta_ind['mu_init'], 'sigma': meta_ind['sigma_init']})
    GA.alter_pairing(avoid_asex=meta_ind['avoid_asex'])
    fitweigh = nga.sigmoid_fitness_weighting
    if not meta_ind['fitness_weighting']:
        fitweigh = lambda p1, p2: 0.5
    GA.alter_crossover(fitness_weighting=fitweigh)
    GA.alter_selection(elitist_percent=meta_ind['elite_sel'], lucky_chance=meta_ind['lucky_sel'])
    GA.alter_survival(elitist_percent=meta_ind['elite_sur'], lucky_chance=meta_ind['lucky_sur'])
    GA.alter_mutation(master_mutation_rate=meta_ind['mut_rate'],
                      ann={'mut_chance': meta_ind['mut_chance'], 'sigma': meta_ind['mut_sigma']})
    return GA

def evaluate_GA(meta_ind, fitness, geno, popsize=20, num_iters=4):
    GA = get_GA_from_meta(meta_ind, fitness, geno, popsize)
    GA.iterate(num_iters)
    return GA.population['fitness'][0] + 0.7 * GA.population['fitness'][1] + 0.5 * GA.population['fitness'][2]

def get_meta_GA(fitness, geno, population_size, minipopsize=20, miniiters=4):
    metageno = meta_geno()
    metafitness = Fitness(lambda ind: evaluate_GA(ind, fitness, geno, minipopsize, miniiters))
    GA = ga.GeneticAlgorithm(metafitness, metageno, ga.random_pairing, ga.elitist_selection, ga.elitist_selection,
                             population_size=population_size)
    GA.initialize(mu_init=      {'mu': 0.00, 'sigma': 0.70, 'can_be_negative': True},
                  sigma_init=   {'mu': 0.70, 'sigma': 0.50},
                  elite_sel=    {'mu': 0.35, 'sigma': 0.15},
                  lucky_sel=    {'mu': 0.20, 'sigma': 0.15},
                  elite_sur=    {'mu': 0.20, 'sigma': 0.15},
                  lucky_sur=    {'mu': 0.20, 'sigma': 0.15},
                  mut_rate=     {'mu': 0.70, 'sigma': 0.40},
                  mut_chance=   {'mu': 0.70, 'sigma': 0.40},
                  mut_sigma=    {'mu': 0.50, 'sigma': 0.25})
    GA.alter_pairing(avoid_asex=False)
    GA.alter_selection(elitist_percent=0.35, lucky_chance=0.1)
    GA.alter_survival(elitist_percent=0.1, lucky_chance=0.1)
    GA.alter_mutation(master_mutation_rate=0.5,
                      mu_init=              {'sigma': 0.50, 'can_be_negative': True},
                      sigma_init=           {'sigma': 0.25},
                      avoid_asex=           {'mut_rate': 0.25},
                      fitness_weighting=    {'mut_rate': 0.15},
                      elite_sel=            {'sigma': 0.10},
                      lucky_sel=            {'sigma': 0.05},
                      elite_sur=            {'sigma': 0.10},
                      lucky_sur=            {'sigma': 0.05},
                      mut_rate=             {'sigma': 0.15},
                      mut_chance=           {'sigma': 0.15},
                      mut_sigma=            {'sigma': 0.10})
    return GA

def meta_learning():
    latent_size = 4
    num_inputs, num_outputs = int(config['BOT']['num_sensors']) + latent_size, 2
    hidden_layers = [latent_size]
    activation_functions = [sigmoid, tanh]
    ann_bridge = get_recurrent_ann_bridge(latent_size)
    fitness = Fitness(get_average_fitness(
        fitness_func=default_fitness_func,
        ann_bridge=ann_bridge,
        iterations_per_map=50,
        maps_to_load = [training_maps[13]]
    ))
    anngeno = ga.Genotype(ann=nga.FixedTopologyANN(num_inputs, num_outputs, hidden_layers, activation_functions))
    metaGA = get_meta_GA(fitness, anngeno, population_size=20, minipopsize=20, miniiters=4)
    meta_fitrecord = metaGA.iterate(10)
    plt.plot(meta_fitrecord)
    plt.show()

    best_ind = {k: metaGA.population[k][0] for k in metaGA.population.keys()}
    print("BEST INDIVIDUAL:")
    print(best_ind)
    print("fitness =",best_ind['fitness'])
    best_GA = get_GA_from_meta(best_ind, fitness, anngeno, popsize=50)
    fitrecord = best_GA.iterate(10)
    plt.plot(fitrecord)
    plt.show()

if __name__=='__main__':
    meta_learning()