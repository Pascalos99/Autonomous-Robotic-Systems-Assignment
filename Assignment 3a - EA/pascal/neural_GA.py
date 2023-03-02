import genetic_algorithm as ga
import numpy as np

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

class ANN:
    def __init__(self, num_inputs, num_outputs, hidden_layers=[], activation_functions=sigmoid):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layers = hidden_layers
        self.layers = [num_inputs] + hidden_layers + [num_outputs]
        self.network = [np.zeros((self.layers[i]+1, self.layers[i+1])) for i in range(len(self.layers)-1)]
        if type(activation_functions) is not list:
            self.activation_functions = [activation_functions for _ in range(len(self.layers))]
        else: self.activation_functions = [activation_functions[i] for i in range(len(self.layers))]

    def forward(self, input):
        if not np.shape(input) == (self.num_inputs, ):
            raise Warning("method expects a single input array of shape (%d,)"%self.num_inputs)
        self.activations = [input]
        for l in range(len(self.network)):
            self.activations.append(self.activation_functions[l](np.dot(np.concatenate((self.activations[-1], [1.])), self.network[l])))
        return self.activations[-1]
    
    def __repr__(self):
        return "ANN(%s)"%(','.join([str(x) for x in self.layers]))

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
            new.network = [a + np.random.normal(0, sigma, np.shape(a)) * (mut_chance + np.random.rand(*np.shape(a))).astype(int) for a in old.network]
            return new
        def crossover(parent1: ANN, parent2: ANN, fitness_weighting=lambda p1, p2: 0.5, fitness1=1, fitness2=1) -> ANN:
            # fitness_weighting(fitness1: float, fitness2: float) -> float | is the probability of selecting from parent1
            child = ANN(num_inputs, num_outputs, hidden_layers, activation_functions)
            child.network = network_crossover_simple(parent1.network, parent2.network, fitness_weighting(fitness1, fitness2))
            return child
        ga.Parameter.__init__(self, init, mutate, crossover)

def get_ANN_GA(fitness, num_inputs, num_outputs, hidden_layers=[], activation_functions=sigmoid, popsize=50, init_mu=0, init_sigma=0.5,
               avoid_asex=True, keep_old_population=True, elitist_percent=0.4, lucky_chance=0.2, master_mut_chance=1.,
               weight_mut_chance=1., mut_sigma=0.1, crossover_fit_weighting=lambda p1, p2: 0.5):
    anngeno = ga.Genotype(ann=FixedTopologyANN(num_inputs, num_outputs, hidden_layers, activation_functions))
    GA = ga.GeneticAlgorithm(fitness, anngeno, ga.random_pairing, ga.elitist_selection, population_size=popsize)
    GA.initialize(ann={'mu':init_mu, 'sigma':init_sigma})
    GA.alter_pairing(avoid_asex=avoid_asex)
    GA.alter_crossover(keep_old_population=keep_old_population, fitness_weighting=crossover_fit_weighting)
    GA.alter_selection(elitist_percent=elitist_percent, lucky_chance=lucky_chance)
    GA.alter_mutation(master_mutation_rate=master_mut_chance, ann={'mut_chance':weight_mut_chance, 'sigma':mut_sigma})
    return GA

if __name__ == '__main__':
    def ann_cost(ann: ANN, data) -> float:
        cost, num = 0, 0
        for dat in data:
            train_x, train_y = dat
            cost += np.sum((ann.forward(train_x) - train_y)**2)
            num += 1
        return cost / num
    # This particular dataset should be familiar to anyone who did ACML assignment 1 :)
    dat_len, comp_len = 8, 3
    train_data = [(np.eye(dat_len)[i], np.eye(dat_len)[i]) for i in range(dat_len)]
    fitness = ga.Fitness(lambda pop: ann_cost(pop['ann'], train_data), minimize=True)
    # We are testing the autoencoder capabilities of this GA-powered ANN
    # * note that this task is easier to accomplish with backpropagation!
    num_inputs, num_outputs, hidden_layers = dat_len, dat_len, [comp_len]
    activation_functions = sigmoid
    popsize = 50
    fit_ratio_importance = 1/3.
    params = {
        "init_mu": 0,
        "init_sigma": 0.5,
        "avoid_asex": False,
        "keep_old_population": False,
        "elitist_percent": 0.35,
        "lucky_chance": 0.15,
        "master_mut_chance": 0.5,
        "weight_mut_chance": 0.6,
        "mut_sigma": 0.35,
        "crossover_fit_weighting": lambda p1, p2: (p1 / (p1 + p2)) * fit_ratio_importance + (1 - fit_ratio_importance)
    }

    GA = get_ANN_GA(fitness, num_inputs, num_outputs, hidden_layers, activation_functions, popsize, **params)
    fits = GA.iterate(400, parallel=True)
    best: ANN = GA.population['ann'][0]
    print("best fitness:", fits[-1][0])
    print("best model output: (should be the identity matrix)")
    for dat in train_data:
        print(np.round(best.forward(dat[0]), 2))
    
    from matplotlib import pyplot as plt
    plt.plot(fits)
    plt.xlabel("generations")
    plt.ylabel("cost")
    plt.show()
