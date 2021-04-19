import numpy as np
from genopt.crossover import one_way_crossover


def test_one_way_crossover_single_chromosome():
    n_vars = 4
    chromosome = np.random.rand(n_vars)
    new_chromosome = one_way_crossover(chromosome)
    assert (new_chromosome == chromosome).all()


def test_one_way_crossover_even_popsize():
    n_vars = 5
    popsize = 2
    population = np.random.rand(popsize, n_vars)
    new_population = one_way_crossover(population)
    assert population[0, 0] == new_population[1, 0]
    assert population[1, 0] == new_population[0, 0]


def test_one_way_crossover_uneven_popsize():
    n_vars = 5
    popsize = 3
    population = np.random.rand(popsize, n_vars)
    new_population = one_way_crossover(population)
    assert population[0, 0] == new_population[1, 0]
    assert population[1, 0] == new_population[0, 0]
    assert (population[2, :] == new_population[2, :]).all()
