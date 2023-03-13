import numpy as np

from neural_GA import get_ANN_GA, sigmoid, tanh


def euclidian_dist(a_weights: list, b_weights: list) -> float:
    return (sum([(a - b) ** 2 for a, b in zip(a_weights, b_weights)])) ** 0.5

def normalize(x):
    S = float(np.linalg.norm(x))
    if S == 0: 
        return x
    return x/S

def get_diversity_from_dist_matrix(m):
    acc = []
    for row in range(len(m)):
        for col in range(row + 1, len(m[0])):
            acc.append(m[row][col])
    return sum(acc) / len(acc)


def calc_diversity(population):
    anns = population['ann']

    dist_matrix = np.zeros(shape=(len(anns), len(anns)))
    for i, ann in enumerate(anns):
        for j, p in enumerate(anns):
            acc_dist = 0
            for layer in range(len(ann.network)):
                acc_dist += euclidian_dist(ann.network[0][layer], p.network[0][layer])
            dist_matrix[i][j] = acc_dist

    print(dist_matrix)
    diversity = get_diversity_from_dist_matrix(dist_matrix)
    return diversity


if __name__ == '__main__':
    ga = get_ANN_GA(lambda x: 1, 14, 2, [4], popsize=5, activation_functions=[sigmoid, sigmoid, tanh])

    print(calc_diversity(ga.population))
