# PyGenOpt

A simple package for genetic optimization in python.

## Installation

To install from source code:

`git clone https://github.com/OskarLiew/PyGenOpt.git`

`pip install PyGenOpt/`

Package might be uploaded to PyPi repository later.

## Usage

Example code to maximize bic by selecting variables of a linear regression model:

```python
import statsmodels.api as sm
from genopt import GeneticOptimizer
from functools import partial
import numpy as np


# Helper function to get the columns defined by the chromosome
def get_X_subset(chromosome, X):
    mask = chromosome.astype(bool)
    return X.iloc[:, mask]
    

# Objective function to optimize
def linear_regression_minimize_bic(chromosome, X, y):
    if np.all(chromosome == 0):
        return -1e9
    
    X_subset = get_X_subset(chromosome, X)
    lr = sm.OLS(y, X_subset)
    results = lr.fit()

    return -results.bic


# Load data. There are not very many variables in this dataset, so there
# is a large chanse that the best subset is found in the first generation
data = sm.datasets.longley.load_pandas()
X = sm.add_constant(data.exog)
y = data.endog
feature_names = X.columns

# Setup GeneticOptimizer
go = GeneticOptimizer(
    n_vars = X.shape[1],
    popsize=100,
    n_gen=10,
    objective_function=partial(linear_regression_minimize_bic, X=X, y=y),
    encoding="discrete",
    var_range=(0, 1),
    var_size=1,
)

best_chromosome = go.optimize()

# Show results of optimization
X_subset = get_X_subset(best_chromosome, X)
lr = sm.OLS(y, X_subset)
results = lr.fit()
print(results.summary())

```
