import numpy as np


def random_mutation(
    population,
    n_vars,
    popsize,
    mut_p,
    mut_var,
    encoding,
    var_size,
):
    if encoding == "real":
        selected = np.random.rand(popsize, n_vars) < mut_p
        population[selected] = population[selected] + np.random.normal(
            0, mut_var, size=population[selected].shape
        )
    else:
        selected = np.random.rand(popsize, n_vars * var_size) < mut_p
        population[selected] = (population[selected] + 1) % 2
    return population
