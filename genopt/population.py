import numpy as np


def random_init(n_vars, popsize, encoding, var_size):
    if encoding == "real":
        population = np.random.normal(0, 1, (popsize, n_vars))
    else:
        population = np.random.randint(2, size=(popsize, n_vars * var_size))
    return population


def update_population(population, top_individual, elitism):
    population[:elitism, :] = top_individual
    return population
