import numpy as np

from genopt.evaluation import evaluate, decode_discrete


def objective(arr):
    return arr.sum()


def test_decode_discrete_single_chromosome():
    n_vars = 10
    var_size = 4
    chromosome = np.random.randint(2, size=n_vars * var_size)
    x = decode_discrete(chromosome, n_vars, (-1, 1), var_size)
    assert np.logical_and(x >= -1, x <= 1).all()


def test_decode_discrete_population():
    n_vars = 10
    var_size = 4
    popsize = 10
    chromosome = np.random.randint(2, size=(popsize, n_vars * var_size))
    x = decode_discrete(chromosome, n_vars, (-1, 1), var_size)
    assert np.logical_and(x >= -1, x <= 1).all()


def test_decode_discrete_special_case():
    n_vars = 10
    var_size = 1
    chromosome = np.random.randint(2, size=n_vars * var_size)
    x = decode_discrete(chromosome, n_vars, (0, 1), var_size)
    assert np.logical_and(x >= -1, x <= 1).all()
    assert x is chromosome


def test_evaluate_single_chromosome():
    n_vars = 10
    variables = np.random.rand(n_vars)
    y, i_max = evaluate(variables, objective)
    assert y[0] == objective(variables)
    assert i_max == 0


def test_evaluate_population():
    n_vars = 10
    popsize = 25
    variables = np.random.rand(popsize, n_vars)
    variables[-1, :] = 1
    _, i_max = evaluate(variables, objective)
    assert i_max == popsize - 1
