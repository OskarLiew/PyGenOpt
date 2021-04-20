import numpy as np


def init_discrete(popsize: int, n_vars: int, var_size: int) -> np.ndarray:
    """Initialize a population of binary chromosomes

    Args:
        popsize (int): Population size
        n_vars (int): Number of variables
        var_size (int): Size of each variable

    Returns:
        np.ndarray: Binary population
    """
    return np.random.randint(2, size=(popsize, n_vars * var_size))


def init_real(popsize: int, n_vars: int) -> np.ndarray:
    """Initalize a population of real valued chromosomes

    Args:
        popsize (int): Population size
        n_vars (int): Number of variables

    Returns:
        np.ndarray: Real valued population
    """
    return np.random.normal(0, 1, (popsize, n_vars))


def update_population(
    population: np.ndarray, top_individual: np.ndarray, elitism: int
) -> np.ndarray:
    """Update a population with elitism and shuffle

    Args:
        population (np.ndarray): Current population
        top_individual (np.ndarray): Chromosome of top individual
        elitism (int): Number of copies of the best individual to copy to new population

    Returns:
        np.ndarray: Population after elitism and shuffle
    """
    new_population = population.copy()
    np.random.shuffle(new_population)
    new_population[:elitism, :] = top_individual
    return new_population
