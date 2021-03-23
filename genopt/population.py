import numpy as np


def init_discrete(popsize, n_vars, var_size):
    return np.random.randint(2, size=(popsize, n_vars * var_size))


def init_real(popsize, n_vars):
    return np.random.normal(0, 1, (popsize, n_vars))


def update_population(population, top_individual, elitism):
    new_population = population.copy()
    new_population[:elitism, :] = top_individual
    return new_population
