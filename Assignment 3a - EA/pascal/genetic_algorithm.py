class Parameter:
    def __init__(self, init: callable, mutate: callable, crossover: callable):
        self.init = init # init() -> some_type
        self.mutate = mutate # mutate(value: some_type) -> some_type
        self.crossover = crossover # crossover(value1: some_type, value2: some_type) -> some_type

class Genotype:
    def __init__(self, **parameters):
        # note that all patameters must be instances of the Parameter class
        self.params = list(parameters.keys())
        for par in self.params:
            self.__dict__[par] = parameters[par]

        self.__init_kwargs = {par: {} for par in self.params}
        self.__mutate_kwargs = {par: {} for par in self.params}
        self.__crossover_kwargs = {par: {} for par in self.params}

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
    
    def crossover(self, population: dict, pairs: list) -> dict:
        # pairs is a list of 2-tuples of indices of individuals to be crossed over
        # the returned dictionary is in order of the given pairs and contains their unmutated offspring
        return {param: [self.__dict__[param].crossover(population[param][p1], population[param][p2], **self.__crossover_kwargs[param]) for p1, p2 in pairs] for param in self.params}
    
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
        self.__compute = compute
        self.minimize = minimize

    def individual(self, individual: dict, recompute=False):
        if (not recompute) and ('fitness' in individual.keys()):
            fit = individual['fitness']
            if fit is not None: return fit
        individual['fitness'] = self.__compute(individual)
        return individual['fitness']
    
    def compute(self, population: dict, recompute_all=False) -> None:
        if (not 'fitness' in population.keys()) or recompute_all:
            population['fitness'] = [None for i in range(len(population[list(population.keys())[0]]))]
        population['fitness'] = [population['fitness'][i] if population['fitness'][i] is not None else self.__compute({param: population[param][i] for param in population.keys()}) for i in range(len(population['fitness']))]

    def population(self, population: dict, recompute=False):
        self.compute(population, recompute)
        return population['fitness']
    
    def sort_population(self, population: dict, recompute=False) -> None:
        self.compute(population, recompute)
        sorted_index = sorted(range(len(population['fitness'])), key=lambda i: population['fitness'][i])
        for par in population.keys():
            population[par] = [population[par][i] for i in sorted_index]

class GeneticAlgorithm:
    def __init__(self, fitness: Fitness, genotype: Genotype, pairing: callable, selection: callable, population_size: int):
        # selection(population) -> survivors
        # pairing(population, size) -> [(parent1, parent2), ...] (len = size)
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
    
    def iterate(self, num_iters, record_fitness=True, record_population=False):
        fitness_record = [[] for _ in range(num_iters)]
        populat_record = [{} for _ in range(num_iters)]

        for i in range(num_iters):
            # determine population fitness:
            self.fitness.sort_population(self.population, i==0)
            if record_fitness: fitness_record.append(self.population['fitness'])
            if record_population: populat_record.append(self.population)
            # create offspring:
            survivepop = self.selection(self.population, **self.selectionpars)
            numoffspring = self.popsize
            if self.keep_pops: numoffspring -= len(survivepop)
            offspring = self.genotype.crossover(survivepop, self.pairing(survivepop, numoffspring, **self.pairingpars))
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

        if record_fitness: fitness_record.append(self.population['fitness'])
        if record_population: populat_record.append(self.population)
        if record_fitness and not record_population: return fitness_record
        if not record_fitness and record_population: return populat_record
        if record_fitness and record_population: return fitness_record, populat_record