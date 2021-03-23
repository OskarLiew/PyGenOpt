import numpy as np
from genopt import GeneticOptimizer


def objective(arr):
    return arr.sum()


def test_go_real():
    n_vars = 10
    random_result = objective(np.random.rand(n_vars))
    go = GeneticOptimizer(n_vars, 100, objective)
    result = go.optimize(10)
    assert objective(result) > random_result


def test_go_discrete():
    n_vars = 10
    var_size = 4
    random_result = objective(np.random.rand(n_vars))
    go = GeneticOptimizer(
        n_vars, 100, objective, encoding="discrete", var_size=var_size
    )
    result = go.optimize(10)
    result_decoded = go.decode(result)
    assert objective(result_decoded) > random_result
