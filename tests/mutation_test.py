import numpy as np
from genopt.mutation import mutation_discrete, mutation_real


def test_mutation_real_single_chromosome():
    n_vars = 10
    chromosome = np.random.rand(n_vars)
    mutated = mutation_real(chromosome.copy(), 1, 1)
    assert not (chromosome == mutated).any()


def test_mutation_real():
    n_vars = 10
    popsize = 25
    chromosome = np.random.rand(popsize, n_vars)
    mutated = mutation_real(chromosome.copy(), 0.5, 1)
    assert not (chromosome == mutated).all()
    assert (chromosome == mutated).any()


def test_mutation_discrete_single_chromosome():
    n_vars = 10
    var_size = 4
    chromosome = np.random.randint(2, size=n_vars * var_size)
    mutated = mutation_discrete(chromosome.copy(), 1)
    assert not (chromosome == mutated).any()


def test_mutation_discrete():
    n_vars = 10
    var_size = 4
    popsize = 25
    chromosome = np.random.randint(2, size=(popsize, n_vars * var_size))
    mutated = mutation_discrete(chromosome.copy(), 0.5)
    assert not (chromosome == mutated).all()
    assert (chromosome == mutated).any()
