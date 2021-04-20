# PyGenOpt

A simple package for genetic optimization in python.

## Installation

To install from source code:

`git clone https://github.com/OskarLiew/PyGenOpt.git`

`pip install PyGenOpt/`

Package will be uploaded to PyPi repository later.

## Usage

Example code to minimize [BIC](https://en.wikipedia.org/wiki/Bayesian_information_criterion) by selecting variables of a linear regression model:

```python
from functools import partial
import statsmodels.api as sm
from sklearn import datasets
import numpy as np
import pandas as pd
from genopt import GeneticOptimizer


# Helper function to fit a regression model after a chromosome
def linear_regression_fit_chromosome(chromosome, x_data, targets):
    x_subset = x_data.iloc[:, chromosome.astype(bool)]
    regression_model = sm.OLS(targets, x_subset)
    return regression_model.fit()


# Objective function to maximize
def linear_regression_minimize_bic(chromosome, x_data, targets):
    if np.all(chromosome == 0):
        return -1e9

    results = linear_regression_fit_chromosome(chromosome, x_data, targets)
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
    results = linear_regression_fit_chromosome(best_chromosome, x_data, targets)
    print(results.summary())


if __name__ == "__main__":
    main()
```
