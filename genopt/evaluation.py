from typing import Callable, Tuple
import numpy as np


def decode_discrete(
    population: np.ndarray, n_vars: int, var_range: Tuple[float], var_size: int
) -> np.ndarray:
    """Decode a population of binary chromosomes into an array with
    discrete values in a variable range.

    Args:
        population (np.ndarray): Population to decode
        n_vars (int): Number of variables in decoded chromosome
        var_range (Tuple[float]): Variable range in decoded chromosome as a tuple
            of floats with length 2: ``(lower, upper)``
        var_size (int): Number of binary genes for each decoded variable

    Returns:
        np.ndarray: Decoded population with shape (popsize, n_vars)
    """
    # Special case where there is no need to decode
    if var_range[0] == 0 and var_range[1] == 1 and var_size == 1:
        return population

    population = np.atleast_2d(population.copy())
    popsize = population.shape[0]
    decoded = np.zeros((popsize, n_vars))
    var_min = var_range[0]
    var_max = var_range[1]

    # Has room for optimization
    for i_indiv in range(popsize):
        for i_var in range(n_vars):
            var_bits = population[
                i_indiv,
                i_var * var_size : (i_var * var_size + var_size),
            ]
            var_decimal = var_bits.dot(-(2 ** np.arange(var_bits.size)))
            var_scaled = (var_min - var_max) * var_decimal / (
                2 ** var_size - 1
            ) + var_min
            decoded[i_indiv, i_var] = var_scaled

    return decoded


def evaluate(variables: np.ndarray, objective_function: Callable) -> np.ndarray:
    """Evaluate the fitness scores of a population

    Args:
        variables (np.ndarray): Decoded chromosomes
        objective_function (Callable): Objective function that takes a
            single chromosome as input and returns a fitness score.

    Returns:
        np.ndarray: Fitness scores with shape (popsize, )
    """
    variables = np.atleast_2d(variables.copy())
    popsize = variables.shape[0]
    fitness = [objective_function(variables[i, :]) for i in range(popsize)]
    return np.array(fitness)
