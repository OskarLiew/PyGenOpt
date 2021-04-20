import numpy as np


def one_way_crossover(population: np.ndarray) -> np.ndarray:
    """Produce child chomosomes from pairs of parent chromosomes by swapping
    the chromosomes to the right of a randomly selected index.

    Args:
        population (np.ndarray): Parent population

    Returns:
        np.ndarray: Child population
    """
    new_population = np.atleast_2d(population.copy())
    popsize, chromosome_length = new_population.shape
    for j in np.arange(popsize - 1, step=2):

        # Take two chromosomes and find a crossover point
        chromosome1 = new_population[j, :].copy()
        chromosome2 = new_population[j + 1, :].copy()
        cross_point = np.random.randint(1, chromosome_length - 1)

        # Swap the chromosomes at the crossover point and update the population
        new_population[j, :cross_point] = chromosome2[:cross_point]
        new_population[j + 1, :cross_point] = chromosome1[:cross_point]

    return new_population
