Linear Regression Variable Selection
====================================

One use case for genetic optimization (GO) is to find a good set of variables
for a multiple linear regression model. Once the number of variables in a
dataset grows, it becomes increasingly harder to find good combinations of
variables and GO can help with that.

One way to measure how good a linear regression model explains some dataset is to
look at the
`Bayesian Information Critereon (BIC) <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_ 
that decreases when the likelihood of the model increases and increases when
more variables are added. A regression model with low BIC means we have found a
good fit using few variables, which makes sure the model has not overfit the data.

Code Walkthough
---------------

We start by importing necessary packages. We use
`statsmodels <https://www.statsmodels.org/stable/index.html>`_ for the regression
model because it automatically computes the BIC value of each model.

.. literalinclude:: ../../../examples/linear_regression_variable_selection.py
    :lines: 1-6

Next we need to write the function to optmize. In our case we want to fit a linear
regression model on a set of variables that are defined by the chromosome. For this
we can use a binary chromosome where 1 means that a variable is included and 0
means that it is excluded. We write a small helper function to fit a regression
model on these variables

.. literalinclude:: ../../../examples/linear_regression_variable_selection.py
    :lines: 9-13

Because the :py:class:`~genopt.GeneticOptimizer` will try to maximize the fitness
of its population, and we want to minimize the BIC, we define our fitness as
``-BIC``. If all chromosomes are 0 we cannot train a model, so we create an edge
case for that, that gives very low fitness.

.. literalinclude:: ../../../examples/linear_regression_variable_selection.py
    :lines: 16-22

In our main function we load the data that we want to use, in this case the 
`Boston housing dataset <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html>`_,
which has a small dimensionality for GO, but big enough for the purposes of this example.

We initialize a :py:class:`~genopt.GeneticOptimizer` with a population size of 100
and binary genes. We use the ``partial`` function from functools to set default
values, so that ``chromosome`` is the only argument of the objective function.
Then we can start the optimization. Because the dimensionality of the Boston
dataset is small we can expect to find the global optimum within a few generations.
We can then print the results of the best model.

.. literalinclude:: ../../../examples/linear_regression_variable_selection.py
    :lines: 25-

If we run it we get this output::

    2021-04-20 22:29:32,169 - INFO - genopt.go - Generation: 0 - Max fitness: -3118.9878994365486
    2021-04-20 22:29:32,237 - INFO - genopt.go - Generation: 1 - Max fitness: -3118.9878994365486
    2021-04-20 22:29:32,304 - INFO - genopt.go - Generation: 2 - Max fitness: -3118.9878994365486
    2021-04-20 22:29:32,371 - INFO - genopt.go - Generation: 3 - Max fitness: -3118.9878994365486
    2021-04-20 22:29:32,437 - INFO - genopt.go - Generation: 4 - Max fitness: -3116.0042335230205
    2021-04-20 22:29:32,503 - INFO - genopt.go - Generation: 5 - Max fitness: -3112.7405121573247
    2021-04-20 22:29:32,570 - INFO - genopt.go - Generation: 6 - Max fitness: -3112.7405121573247
    2021-04-20 22:29:32,638 - INFO - genopt.go - Generation: 7 - Max fitness: -3112.7405121573247
    2021-04-20 22:29:32,705 - INFO - genopt.go - Generation: 8 - Max fitness: -3110.0331325837083
    2021-04-20 22:29:32,775 - INFO - genopt.go - Generation: 9 - Max fitness: -3110.0331325837083
                                    OLS Regression Results                                
    =======================================================================================
    Dep. Variable:                      y   R-squared (uncentered):                   0.958
    Model:                            OLS   Adj. R-squared (uncentered):              0.957
    Method:                 Least Squares   F-statistic:                              1425.
    Date:                Tue, 20 Apr 2021   Prob (F-statistic):                        0.00
    Time:                        22:29:32   Log-Likelihood:                         -1530.1
    No. Observations:                 506   AIC:                                      3076.
    Df Residuals:                     498   BIC:                                      3110.
    Df Model:                           8                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    CRIM          -0.0799      0.031     -2.545      0.011      -0.142      -0.018
    ZN             0.0437      0.014      3.138      0.002       0.016       0.071
    CHAS           2.8766      0.898      3.205      0.001       1.113       4.640
    RM             5.5964      0.244     22.926      0.000       5.117       6.076
    DIS           -0.7761      0.158     -4.904      0.000      -1.087      -0.465
    PTRATIO       -0.4881      0.098     -5.000      0.000      -0.680      -0.296
    B              0.0140      0.003      5.404      0.000       0.009       0.019
    LSTAT         -0.4853      0.041    -11.761      0.000      -0.566      -0.404
    ==============================================================================
    Omnibus:                      189.204   Durbin-Watson:                   1.012
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1085.081
    Skew:                           1.526   Prob(JB):                    2.39e-236
    Kurtosis:                       9.492   Cond. No.                     1.49e+03
    ==============================================================================

    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.49e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.

Full Code
---------

.. literalinclude:: ../../../examples/linear_regression_variable_selection.py
