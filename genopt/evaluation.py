import numpy as np
from numpy.core.fromnumeric import var


def decode_discrete(population, n_vars, var_range, var_size):
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


def evaluate(variables, objective_function):
    variables = np.atleast_2d(variables.copy())
    popsize = variables.shape[0]
    fitness = [objective_function(variables[i, :]) for i in range(popsize)]
    i_max = np.argmax(fitness)
    return np.array(fitness), i_max
