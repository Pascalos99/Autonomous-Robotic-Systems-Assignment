import math
import random
from inspect import getfullargspec as fargs


class Parameter:
    def __init__(self, init: callable, mutate: callable, crossover: callable):
        self.init = init  # init() -> some_type
        self.mutate = mutate  # mutate(value: some_type) -> some_type
        self.crossover = crossover  # crossover(value1: some_type, value2: some_type) -> some_type
        # crossover has two optional parameters "fitness1: float" and "fitness2: float", which (if present)
        #  represent the fitness of either parent when generating their child
        #  [!] the parent's fitness * -1 is given instead if fitness is to be minimized


class Genotype:
    def __init__(self, **parameters):
        # note that all patameters must be instances of the Parameter class
        self.params = list(parameters.keys())
        for par in self.params:
            self.__dict__[par] = parameters[par]

        self.__init_kwargs = {par: {} for par in self.params}
        self.__mutate_kwargs = {par: {} for par in self.params}
        self.__crossover_kwargs = {par: {} for par in self.params}
        self.__fit_params = {
            par: [fit for fit in ['fitness1', 'fitness2'] if fit in fargs(self.__dict__[par].crossover).args] for par in
            self.params}

    def get_param(self, param_name) -> Parameter:
        if not param_name in self.params:
            raise Warning("param %s not in available params for %s" % (param_name, self))
        return self.__dict__[param_name]

    def alter_init(self, **kwargs_per_param):
        self.__init_kwargs = {par: kwargs_per_param[par] if par in kwargs_per_param.keys() else {} for par in
                              self.params}

    def alter_mutate(self, **kwargs_per_param):
        self.__mutate_kwargs = {par: kwargs_per_param[par] if par in kwargs_per_param.keys() else {} for par in
                                self.params}

    def alter_crossover(self, **kwargs_per_param):
        self.__crossover_kwargs = {par: kwargs_per_param[par] if par in kwargs_per_param.keys() else {} for par in
                                   self.params}

    def get_individual(self) -> dict:
        return {param: self.__dict__[param].init(**self.__init_kwargs[param]) for param in self.params}

    def get_population(self, population_size) -> dict:
        return {param: [self.__dict__[param].init(**self.__init_kwargs[param]) for _ in range(population_size)] for
                param in self.params}

    def mutate_all(self, population, indices=None) -> None:
        if indices is None:
            indices = range(len(population[list(population.keys())[0]]))
        for param in self.params:
            for i in indices:
                population[param][i] = self.__dict__[param].mutate(population[param][i], **self.__mutate_kwargs[param])

    def crossover(self, population: dict, pairs: list, FM=1) -> dict:
        # pairs is a list of 2-tuples of indices of individuals to be crossed over
        # the returned dictionary is in order of the given pairs and contains their unmutated offspring
        # FM is an internal parameter: "fitness multiplier" which is +1 for maximizing fitness, and -1 otherwise
        ex1 = lambda p1, p2: {'fitness1': FM * population['fitness'][p1], 'fitness2': FM * population['fitness'][p2]}
        extr = lambda par, p1, p2: {fit: ex1(p1, p2)[fit] for fit in self.__fit_params[par]}
        return {param: [self.__dict__[param].crossover(population[param][p1], population[param][p2],
                                                       **extr(param, p1, p2), **self.__crossover_kwargs[param]) for
                        p1, p2 in pairs] for param in self.params}

    def join_pops(self, *pops) -> dict:
        all_params = self.params
        for pop in pops: all_params = all_params + [par for par in pop.keys() if not par in all_params]
        population = {param: [] for param in all_params}
        for pop in pops:
            for param in all_params:
                if param in pop.keys():
                    population[param] = population[param] + pop[param]
                else:
                    population[param] = population[param] + [None for _ in range(len(pop[list(pop.keys())[0]]))]
        return population


class Fitness:
    def __init__(self, compute: callable, minimize=False):
        # compute({individual}) -> double
        self.__compute = compute
        self.minimize = minimize

    def individual(self, individual: dict, recompute=False) -> float:
        if (not recompute) and ('fitness' in individual.keys()):
            fit = individual['fitness']
            if fit is not None: return fit
        individual['fitness'] = self.__compute(individual)
        return individual['fitness']

    def compute(self, population: dict, recompute_all=False) -> None:
        if (not 'fitness' in population.keys()) or recompute_all:
            population['fitness'] = [None for i in range(len(population[list(population.keys())[0]]))]
        population['fitness'] = [population['fitness'][i] if population['fitness'][i] is not None else self.__compute(
            {param: population[param][i] for param in population.keys()}) for i in range(len(population['fitness']))]

    def population(self, population: dict, recompute=False) -> list:
        self.compute(population, recompute)
        return population['fitness']

    def sort_population(self, population: dict, recompute=False) -> None:
        self.compute(population, recompute)
        sorted_index = sorted(range(len(population['fitness'])), key=lambda i: population['fitness'][i])
        if not self.minimize: sorted_index.reverse()
        for par in population.keys():
            population[par] = [population[par][i] for i in sorted_index]


class GeneticAlgorithm:
    def __init__(self, fitness: Fitness, genotype: Genotype, pairing: callable, selection: callable, survival: callable,
                 population_size: int):
        # selection(population: dict) -> allowed_to_reproduce: dict
        # pairing(population: dict, size: int) -> [(p1_index, p2_index), ...]: list(tuple) w/ len = size
        # survival(allowed_to_reproduce: dict) -> survivors: dict
        self.fitness = fitness
        self.genotype = genotype
        self.pairing = pairing
        self.selection = selection
        self.survival = survival
        self.popsize = population_size
        self.pairingpars = {}
        self.selectionpars = {}
        self.survivorpars = {}
        self.population = None
        self.iter = 0
        self.mut_rate = 1.

    def initialize(self, **init_kwargs):
        # kwargs need to be named as the parameter they affect, with the value being the kwargs dictionary of the method it modifies
        self.genotype.alter_init(**init_kwargs)
        self.population = self.genotype.get_population(self.popsize)

    def alter_mutation(self, master_mutation_rate=1, **mutation_kwargs):
        # kwargs need to be named as the parameter they affect, with the value being the kwargs dictionary of the method it modifies
        self.mut_rate = master_mutation_rate
        self.genotype.alter_mutate(**mutation_kwargs)

    def alter_crossover(self, **crossover_kwargs):
        # kwargs need to be named as the parameter they affect, with the value being the kwargs dictionary of the method it modifies
        self.genotype.alter_crossover(**crossover_kwargs)

    def alter_selection(self, **selection_kwargs):
        # kwargs are fed directly into the selection method defined at initialization
        self.selectionpars = selection_kwargs

    def alter_survival(self, **survival_kwargs):
        # kwargs are fed directly into the survival method defined at initialization
        self.survivorpars = survival_kwargs

    def alter_pairing(self, **pairing_kwargs):
        # kwargs are fed directly into the pairing method defined at initialization
        self.pairingpars = pairing_kwargs

    def iterate(self, num_iters, record_fitness=True, record_population=False, display_iterations=False):
        fitness_record = []
        populat_record = []

        for i in range(num_iters):
            if display_iterations:
                print(f"Iteration { i + 1 }/{ num_iters }")
            # determine population fitness:
            self.fitness.sort_population(self.population, i == 0)
            if record_fitness: fitness_record.append(list(self.population['fitness']))
            if record_population: populat_record.append(dict(self.population))
            # create offspring:
            breedpop = self.selection(self.population, **self.selectionpars)
            survivepop = self.survival(breedpop, **self.survivorpars)
            numoffspring = self.popsize - len(survivepop[list(survivepop.keys())[0]])
            FM = 1. - 2. * int(self.fitness.minimize)
            offspring = self.genotype.crossover(breedpop, self.pairing(breedpop, numoffspring, **self.pairingpars),
                                                FM=FM)
            # mutate offspring:
            mutate_indices = None
            if self.mut_rate < 1.:
                mutate_indices = random.sample(range(numoffspring), int(round(self.mut_rate * numoffspring)))
            self.genotype.mutate_all(offspring, mutate_indices)
            # set new population:
            self.population = self.genotype.join_pops(survivepop, offspring)
            self.iter += 1

        self.fitness.sort_population(self.population, False)
        if record_fitness: fitness_record.append(self.population['fitness'])
        if record_population: populat_record.append(self.population)
        if record_fitness and not record_population: return fitness_record
        if not record_fitness and record_population: return populat_record
        if record_fitness and record_population: return fitness_record, populat_record


# NOTE: we can use selection methods and survival methods interchangably!
# ------------- #
#   SELECTION   #
# ------------- #

def elitist_selection(population, elitist_percent=0.3, lucky_chance=0.1):
    popsize = len(population[list(population.keys())[0]])
    elitists = int(max(math.ceil(elitist_percent), round(elitist_percent * popsize)))
    pop_elitist = {par: population[par][:elitists] for par in population.keys()}
    luckies = int(round(lucky_chance * (popsize - elitists)))
    index_lucky = random.sample(range(elitists, popsize), luckies)
    return {par: pop_elitist[par] + [population[par][i] for i in index_lucky] for par in population.keys()}


def tournament_selection(population, tournament_size=3):
    maximize = True
    if population["fitness"][0] < population["fitness"][-1]:
        maximize = False
    
    population_size = len(population[list(population.keys())[0]])
    winners = { par: [] for par in population.keys() }
    for _ in range(population_size):
        competitors = { par: population[par] for par in population.keys() }
        tournament_indexes = random.sample(range(0, len(competitors[list(competitors.keys())[0]])), tournament_size)
        tournament = { key: [values[i] for i in tournament_indexes] for key, values in competitors.items() }
        if maximize:
            winner = { key: [values[i] for i in [tournament["fitness"].index(max(tournament["fitness"]))]] for key, values in tournament.items() }
        else:
            winner = { key: [values[i] for i in [tournament["fitness"].index(min(tournament["fitness"]))]] for key, values in tournament.items() }
            
        for key in winner:
            winners[key].append(winner[key][0])

    return winners

# ------------- #
#    SURVIVAL   #
# ------------- #

def binary_survival(population, keep_old_pops=False):
    if keep_old_pops: return population
    return {par: [] for par in population.keys()}


# ------------- #
#    PAIRING    #
# ------------- #
def random_asex(population, size):
    popsize = len(population[list(population.keys())[0]])
    return [(c, c) for c in [random.choice(range(popsize)) for _ in range(size)]]


def random_pairing(population, size, avoid_asex=True):
    popsize = len(population[list(population.keys())[0]])
    res = [(random.choice(range(popsize)), random.choice(range(popsize))) for _ in range(size)]
    if not avoid_asex: return res
    for i in range(size):
        p1, p2 = res[i]
        while p1 == p2:
            p1, p2 = random.choice(range(popsize)), random.choice(range(popsize))
        res[i] = (p1, p2)
    return res
