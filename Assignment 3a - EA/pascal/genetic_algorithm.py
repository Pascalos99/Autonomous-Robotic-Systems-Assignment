import random
import inspect
import multiprocessing as mp

import time

def my_function(i, key):
        print("starting %s [%d]"%(key, i))
        time.sleep(2)
        print("finished %s [%d]"%(key, i))
        return (key, int(round(random.random() * 100)))

if __name__ == '__main__':
    results = {'a': 1, 'b': None, 'c': None, 'd':7, 'e': None, 'f': 1, 'g': None}
    params = list(results.keys())
    def store_result(result):
        key, val = result
        results[key] = val
    pool = mp.Pool(mp.cpu_count())
    ts = time.time()
    for i in range(len(params)):
        if results[params[i]] is None:
            pool.apply_async(my_function, args=(i, params[i]), callback=store_result)
    pool.close()
    pool.join()
    print('computation took:', time.time() - ts)
    print(results)

class Parameter:
    def __init__(self, init: callable, mutate: callable, crossover: callable):
        self.init = init # init() -> some_type
        self.mutate = mutate # mutate(value: some_type) -> some_type
        self.crossover = crossover # crossover(value1: some_type, value2: some_type) -> some_type
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
        self.__fit_params = {par: [fit for fit in ['fitness1', 'fitness2'] if fit in inspect.getfullargspec(self.__dict__[par].crossover).args] for par in self.params}

    def get_param(self, param_name) -> Parameter:
        if not param_name in self.params:
            raise Warning("param %s not in available params for %s"%(param_name, self))
        return self.__dict__[param_name]
    
    def alter_init(self, **kwargs_per_param):
        self.__init_kwargs = {par: kwargs_per_param[par] if par in kwargs_per_param.keys() else {} for par in self.params}
    
    def alter_mutate(self, **kwargs_per_param):
        self.__mutate_kwargs = {par: kwargs_per_param[par] if par in kwargs_per_param.keys() else {} for par in self.params}

    def alter_crossover(self, **kwargs_per_param):
        self.__crossover_kwargs = {par: kwargs_per_param[par] if par in kwargs_per_param.keys() else {} for par in self.params}
    
    def get_individual(self) -> dict:
        return {param: self.__dict__[param].init(**self.__init_kwargs[param]) for param in self.params}
    
    def get_population(self, population_size) -> dict:
        return {param: [self.__dict__[param].init(**self.__init_kwargs[param]) for _ in range(population_size)] for param in self.params}
    
    def mutate_all(self, population, indices=None) -> None:
        if indices is None:
            indices = range(len(population))
        for param in self.params:
            for i in indices:
                population[param][i] = self.__dict__[param].mutate(population[param][i], **self.__mutate_kwargs[param])
    
    def crossover(self, population: dict, pairs: list, FM=1) -> dict:
        # pairs is a list of 2-tuples of indices of individuals to be crossed over
        # the returned dictionary is in order of the given pairs and contains their unmutated offspring
        # FM is an internal parameter: "fitness multiplier" which is +1 for maximizing fitness, and -1 otherwise
        ex1 = lambda p1, p2: {'fitness1': FM * population['fitness'][p1], 'fitness2': FM * population['fitness'][p2]}
        extr = lambda par, p1, p2: {fit: ex1(p1,p2)[fit] for fit in self.__fit_params[par]}
        return {param: [self.__dict__[param].crossover(population[param][p1], population[param][p2],
            **extr(param, p1, p2), **self.__crossover_kwargs[param]) for p1, p2 in pairs] for param in self.params}
    
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
    
    # missing:
    # * selection protocol
    # * genetic algorithm framework
    # * fitness sorting

class Fitness:
    def __init__(self, compute: callable, minimize=False):
        # compute({individual}) -> double
        self.__compute_fitness = compute
        self.minimize = minimize

    def individual(self, individual: dict, recompute=False):
        if (not recompute) and ('fitness' in individual.keys()):
            fit = individual['fitness']
            if fit is not None: return fit
        individual['fitness'] = self.__compute(individual)
        return individual['fitness']
    
    def __compute_parallel(self, i, individual):
        print("computing in parallel!")
        return (i, self.__compute_fitness(individual))
    
    def __store_parallel(self, result):
        i, fitness = result
        print("results:", i, fitness)
        self.__temp_fitness[i] = fitness

    def __compute(self, i, individual, parallel_pool: mp.Pool=None):
        if parallel_pool is None:
            return self.__compute_fitness(individual)
        parallel_pool.apply_async(self.__compute_parallel, args=(i, individual), callback=self.__store_parallel)
        # self.__temp_fitness[i] = parallel_pool.apply(self.__compute_parallel, args=(i, individual))

    def compute(self, population: dict, recompute_all=False, parallel=False) -> None:
        pool = None
        if parallel: pool = mp.Pool(mp.cpu_count())

        if (not 'fitness' in population.keys()) or recompute_all:
            population['fitness'] = [None for i in range(len(population[list(population.keys())[0]]))]
        self.__temp_fitness = [population['fitness'][i] if population['fitness'][i] is not None else self.__compute(i, {param: population[param][i] for param in population.keys()}, pool) for i in range(len(population['fitness']))]
        
        if parallel:
            pool.close()
            pool.join()
        for x in range(len(self.__temp_fitness)):
            if self.__temp_fitness[x] is None:
                raise Warning("This should not be happening")
        population['fitness'] = self.__temp_fitness

    def population(self, population: dict, recompute=False, parallel=False):
        self.compute(population, recompute, parallel)
        return population['fitness']
    
    def sort_population(self, population: dict, recompute=False, parallel=False) -> None:
        self.compute(population, recompute, parallel)
        sorted_index = sorted(range(len(population['fitness'])), key=lambda i: population['fitness'][i])
        for par in population.keys():
            population[par] = [population[par][i] for i in sorted_index]

class GeneticAlgorithm:
    def __init__(self, fitness: Fitness, genotype: Genotype, pairing: callable, selection: callable, population_size: int):
        # selection(population: dict) -> survivors: dict
        # pairing(population: dict, size: int) -> [(p1_index, p2_index), ...]: list(tuple) w/ len = size
        self.fitness = fitness
        self.genotype = genotype
        self.pairing = pairing
        self.selection = selection
        self.popsize = population_size
        self.pairingpars = {}
        self.selectionpars = {}
        self.population = None
        self.iter = 0
        self.mut_rate = 1.
        self.keep_pops = True

    def initialize(self, **init_kwargs):
        # kwargs need to be named as the parameter they affect, with the value being the kwargs dictionary of the method it modifies
        self.genotype.alter_init(**init_kwargs)
        self.population = self.genotype.get_population(self.popsize)
    
    def alter_mutation(self, master_mutation_rate=1, **mutation_kwargs):
        # kwargs need to be named as the parameter they affect, with the value being the kwargs dictionary of the method it modifies
        self.mut_rate = master_mutation_rate
        self.genotype.alter_mutate(**mutation_kwargs)
    
    def alter_crossover(self, keep_old_population=True, **crossover_kwargs):
        # kwargs need to be named as the parameter they affect, with the value being the kwargs dictionary of the method it modifies
        self.keep_pops = keep_old_population
        self.genotype.alter_crossover(**crossover_kwargs)
    
    def alter_selection(self, **selection_kwargs):
        # kwargs are fed directly into the selection method defined at initialization
        self.selectionpars = selection_kwargs
    
    def alter_pairing(self, **pairing_kwargs):
        # kwargs are fed directly into the pairing method defined at initialization
        self.pairingpars = pairing_kwargs
    
    def iterate(self, num_iters, record_fitness=True, record_population=False, parallel=False):
        fitness_record = []
        populat_record = []

        for i in range(num_iters):
            # determine population fitness:
            self.fitness.sort_population(self.population, recompute=i==0, parallel=parallel)
            if record_fitness: fitness_record.append(list(self.population['fitness']))
            if record_population: populat_record.append(dict(self.population))
            # create offspring:
            survivepop = self.selection(self.population, **self.selectionpars)
            numoffspring = self.popsize
            if self.keep_pops: numoffspring -= len(survivepop[list(survivepop.keys())[0]])
            FM = 1. - 2. * int(self.fitness.minimize)
            offspring = self.genotype.crossover(survivepop, self.pairing(survivepop, numoffspring, **self.pairingpars), FM=FM)
            # mutate offspring:
            mutate_indices = None
            if self.mut_rate < 1.:
                mutate_indices = random.sample(range(numoffspring), int(round(self.mut_rate * numoffspring)))
            self.genotype.mutate_all(offspring, mutate_indices)
            # set new population:
            if self.keep_pops:
                self.population = self.genotype.join_pops(survivepop, offspring)
            else: self.population = offspring
            self.iter += 1

        self.fitness.sort_population(self.population, parallel=parallel)
        if record_fitness: fitness_record.append(self.population['fitness'])
        if record_population: populat_record.append(self.population)
        if record_fitness and not record_population: return fitness_record
        if not record_fitness and record_population: return populat_record
        if record_fitness and record_population: return fitness_record, populat_record

def elitist_selection(population, elitist_percent=0.3, lucky_chance=0.1):
    popsize = len(population[list(population.keys())[0]])
    elitists = int(round(elitist_percent * popsize))
    pop_elitist = {par: population[par][:elitists] for par in population.keys()}
    index_lucky = random.sample(range(elitists, popsize), int(round(lucky_chance * (popsize - elitists))))
    return {par: pop_elitist[par] + [population[par][i] for i in index_lucky] for par in population.keys()}
    
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