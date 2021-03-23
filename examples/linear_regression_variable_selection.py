from functools import partial
import statsmodels.api as sm
from sklearn import datasets
import numpy as np
import pandas as pd
from genopt import GeneticOptimizer


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


# Load Boston housing dataset
data = datasets.load_boston()
X = pd.DataFrame(data=data["data"], columns=data["feature_names"])
y = data["target"]

# Setup GeneticOptimizer
go = GeneticOptimizer(
    n_vars=X.shape[1],
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
