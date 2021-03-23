import numpy as np
from genopt.population import init_discrete, init_real, update_population


def test_init_discrete():
    population = init_discrete(25, 10, 4)
    assert population.shape == (25, 40)
    assert np.logical_and(population >= 0, population <= 1).all()


def test_init_discrete():
    population = init_real(25, 10)
    assert population.shape == (25, 10)
    assert -0.5 <= population.mean() <= 0.5
    assert 0.5 <= population.var() <= 1.5


def test_update_population():
    top_individual = np.ones(10)
    population = init_real(25, 10)
    new_population = update_population(population, top_individual, 2)
    assert not (population == new_population).all()
    assert (new_population[0, :] == top_individual).all()
    assert (new_population[1, :] == top_individual).all()
    assert (new_population[2, :] != top_individual).all()
