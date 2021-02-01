import numpy as np


def one_way_crossover(population, n_vars, popsize):
    for j in np.arange(popsize, step=2):
        group1 = population[j, :]
        group2 = population[j + 1, :]
        cross_point = np.random.randint(n_vars)
        group1[:cross_point] = group2[:cross_point]
        group2[cross_point:] = group1[cross_point:]
        population[j, :] = group1
        population[j + 1, :] = group2

    return population
