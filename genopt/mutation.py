import numpy as np


def random_selection(population: np.ndarray, mut_p: float) -> np.ndarray:
    """Randomly select genes to mutate

    Args:
        population (np.ndarray): Population of chromosomes
        chromosome_size (int): Number of genes per chromosome
        mut_p (float): Mutation probability

    Returns:
        np.ndarray: Mask of elements from population to mutate
    """
    return np.random.rand(*population.shape) < mut_p


def mutation_discrete(
    population: np.ndarray,
    mut_p: float,
) -> np.ndarray:
    """Mutate discrete chromosomes by swapping the value of selected genes

    Args:
        population (np.ndarray): Population to mutate
        mut_p (float): Mutation probability of each gene

    Returns:
        np.ndarray: Mutated population
    """
    population = np.atleast_2d(population.copy())
    selected = random_selection(population, mut_p)
    population[selected] = (population[selected] + 1) % 2
    return population


def mutation_real(
    population: np.ndarray,
    mut_p: float,
    mut_var: float,
) -> np.ndarray:
    """Mutate real values chromosomes by perturbing the value of selected genes
    by a value from a normal distribution

    Args:
        population (np.ndarray): Population to mutate
        mut_p (float): Mutation probability of each gene
        mut_var (float): Variance of normal distribution

    Returns:
        np.ndarray: Mutated population
    """
    population = np.atleast_2d(population.copy())
    selected = random_selection(population, mut_p)
    population[selected] = population[selected] + np.random.normal(
        0, mut_var, size=population[selected].shape
    )
    return population
