import numpy as np


def init_discrete(popsize: int, n_vars: int, var_size: int):
    return np.random.randint(2, size=(popsize, n_vars * var_size))


def init_real(popsize: int, n_vars: int):
    return np.random.normal(0, 1, (popsize, n_vars))


def update_population(population: np.ndarray, top_individual: np.ndarray, elitism: int):
    new_population = population.copy()
    np.random.shuffle(new_population)
    new_population[:elitism, :] = top_individual
    return new_population
