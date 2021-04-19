import numpy as np


def select_mutatees(
    population: np.ndarray, chromosome_size: int, mut_p: float
) -> np.ndarray:
    popsize = population.shape[0]
    return np.random.rand(popsize, chromosome_size) < mut_p


def mutation_discrete(
    population: np.ndarray,
    n_vars: int,
    mut_p: float,
    var_size: int,
) -> np.ndarray:
    population = np.atleast_2d(population.copy())
    selected = select_mutatees(population, n_vars * var_size, mut_p)
    population[selected] = (population[selected] + 1) % 2
    return population


def mutation_real(
    population: np.ndarray,
    n_vars: int,
    mut_p: float,
    mut_var: float,
) -> np.ndarray:
    population = np.atleast_2d(population.copy())
    selected = select_mutatees(population, n_vars, mut_p)
    population[selected] = population[selected] + np.random.normal(
        0, mut_var, size=population[selected].shape
    )
    return population
