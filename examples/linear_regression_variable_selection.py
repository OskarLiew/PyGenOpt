from functools import partial
import statsmodels.api as sm
from sklearn import datasets
import numpy as np
import pandas as pd
from genopt import GeneticOptimizer


# Helper function to get the columns defined by the chromosome
def get_x_subset(chromosome, x_data):
    mask = chromosome.astype(bool)
    return x_data.iloc[:, mask]


# Objective function to optimize
def linear_regression_minimize_bic(chromosome, x_data, targets):
    if np.all(chromosome == 0):
        return -1e9

    x_subset = get_x_subset(chromosome, x_data)
    regression_model = sm.OLS(targets, x_subset)
    results = regression_model.fit()

    return -results.bic


def main():
    # Load Boston housing dataset
    data = datasets.load_boston()
    x_data = pd.DataFrame(data=data["data"], columns=data["feature_names"])
    targets = data["target"]

    # Setup GeneticOptimizer
    optimizer = GeneticOptimizer(
        n_vars=x_data.shape[1],
        popsize=100,
        objective_function=partial(
            linear_regression_minimize_bic, x_data=x_data, targets=targets
        ),
        encoding="discrete",
        var_range=(0, 1),
        var_size=1,
    )

    best_chromosome = optimizer.optimize(10)

    # Show results of optimization
    x_subset = get_x_subset(best_chromosome, x_data)
    regression_model = sm.OLS(targets, x_subset)
    results = regression_model.fit()
    print(results.summary())


if __name__ == "__main__":
    main()
