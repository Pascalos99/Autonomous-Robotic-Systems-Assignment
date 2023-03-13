import numpy as np

import genetic_algorithm as ga


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


class ANN:
    def __init__(self, num_inputs, num_outputs, hidden_layers=[], activation_functions=sigmoid):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layers = hidden_layers
        self.layers = [num_inputs] + hidden_layers + [num_outputs]
        self.network = [np.zeros((self.layers[i] + 1, self.layers[i + 1])) for i in range(len(self.layers) - 1)]
        self.activations = None
        if type(activation_functions) is not list:
            self.activation_functions = [activation_functions for _ in range(len(self.network))]
        else:
            self.activation_functions = [activation_functions[i] for i in range(len(self.network))]

    def forward(self, input):
        if not np.shape(input) == (self.num_inputs,):
            raise Warning("method expects a single input array of shape (%d,)" % self.num_inputs)
        self.activations = [input]
        for l in range(len(self.network)):
            self.activations.append(
                self.activation_functions[l](np.dot(np.concatenate((self.activations[-1], [1.])), self.network[l])))
        return self.activations[-1]

    def __repr__(self):
        return "ANN(%s)" % (','.join([str(x) for x in self.layers]))


def network_crossover_simple(parent1: list, parent2: list, p1_chance=0.5) -> list:
    p1_picks = [(p1_chance + np.random.rand(*np.shape(p1))).astype(int) for p1 in parent1]
    return [p1_picks[i] * parent1[i] + (1 - p1_picks[i]) * parent2[i] for i in range(len(parent1))]


class FixedTopologyANN(ga.Parameter):
    def __init__(self, num_inputs, num_outputs, hidden_layers=[], activation_functions=sigmoid):
        def init(mu=0, sigma=0.5) -> ANN:
            ann = ANN(num_inputs, num_outputs, hidden_layers, activation_functions)
            ann.network = [np.random.normal(mu, sigma, np.shape(ann.network[i])) for i in range(len(ann.network))]
            return ann

        def mutate(old: ANN, mut_chance=1., sigma=0.05) -> ANN:
            new = ANN(num_inputs, num_outputs, hidden_layers, activation_functions)
            new.network = [
                a + np.random.normal(0, sigma, np.shape(a)) * (mut_chance + np.random.rand(*np.shape(a))).astype(int)
                for a in old.network]
            return new

        def crossover(parent1: ANN, parent2: ANN, fitness_weighting=lambda p1, p2: 0.5, fitness1=1, fitness2=1) -> ANN:
            # fitness_weighting(fitness1: float, fitness2: float) -> float | is the probability of selecting from parent1
            child = ANN(num_inputs, num_outputs, hidden_layers, activation_functions)
            child.network = network_crossover_simple(parent1.network, parent2.network,
                                                     fitness_weighting(fitness1, fitness2))
            return child

        ga.Parameter.__init__(self, init, mutate, crossover)


def get_ANN_GA(fitness, num_inputs, num_outputs, hidden_layers=[], activation_functions=sigmoid, popsize=50, init_mu=0,
               init_sigma=0.5,
               avoid_asex=True, keep_old_population=True, elitist_percent=0.4, lucky_chance=0.2, master_mut_chance=1.,
               weight_mut_chance=1., mut_sigma=0.1, crossover_fit_weighting=lambda p1, p2: 0.5):
    """
    Parameters Explained: \n
    * hidden_layers: a list containing integers representing the size of the networks' hidden layers
        -- ex.: [3]: a single hidden layer of size 3, []: no hidden layers (we get a single-layer-perceptron)

    * activation_functions: a list containing the activation functions of each layer len >= len(hidden_layers) + 1
        -- alternatively, a single activation function can be given to set the activation the same for each layer

    * init_mu: bias for initial weight values
    * init_sigma: stdev for initial weight values
    * avoid_asex: whether or not to avoid asexual reproduction
    * keep_old_population: whether or not to keep the population that was selected to reproduce for the next generation
    * elitist_percent: the top percentage of the fitness-sorted population that will be allowed to reproduce
    * lucky_chance: the chance for an individual in the lower fitness range to be allowed to reproduce
    * master_mut_chance: the chance that a newborn individual is mutated
    * weight_mut_chance: the chance that a weight of an individual is mutated (given that the individual is selected to mutate)
    * mut_sigma: stdev of weight change attributed to a mutated weight

    * crossover_fit_weighting: a function which determines the ratio at which the weights of two individuals are combined at crossover
        -- the ratio depends on the fitness of each parent (does not need to be symmetric, but it is adviced to make it symmetric)
    """

    anngeno = ga.Genotype(ann=FixedTopologyANN(num_inputs, num_outputs, hidden_layers, activation_functions))
    GA = ga.GeneticAlgorithm(fitness, anngeno, ga.random_pairing, ga.elitist_selection, ga.binary_survival,
                             population_size=popsize)
    GA.initialize(ann={'mu': init_mu, 'sigma': init_sigma})
    GA.alter_pairing(avoid_asex=avoid_asex)
    GA.alter_crossover(fitness_weighting=crossover_fit_weighting)
    GA.alter_selection(elitist_percent=elitist_percent, lucky_chance=lucky_chance)
    GA.alter_survival(keep_old_pops=keep_old_population)
    GA.alter_mutation(master_mutation_rate=master_mut_chance, ann={'mut_chance': weight_mut_chance, 'sigma': mut_sigma})
    return GA
