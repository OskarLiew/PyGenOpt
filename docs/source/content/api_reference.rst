.. _api-reference:

API Referece
============

genopt
------

Contains the :py:class:`~genopt.GeneticOptimizer` class, which is the easiest
way to perform genetic optimization using PyGenOpt. Initialize the optimizer with
a multitude of settings and then use :py:meth:`~genopt.GeneticOptimizer.optimize`
to start searching for an optimum.

.. automodule:: genopt
   :members: 
   :undoc-members:
   :show-inheritance:

population
----------

Contains functions for initializing and updating the the population of
chromosomes.

.. automodule:: genopt.population
   :members: 
   :undoc-members:
   :show-inheritance:

evaluation
----------

Contains functions for evaluating the fitness of the population using the
objective function.

.. automodule:: genopt.evaluation
   :members: 
   :undoc-members:
   :show-inheritance:

selection
---------

Contains functions for selection of fit individuals in the population.

.. automodule:: genopt.selection
   :members: 
   :undoc-members:
   :show-inheritance:

crossover
---------

Contains crossover operations that are used to mix the chromosomes of the
population.

.. automodule:: genopt.crossover
   :members: 
   :undoc-members:
   :show-inheritance:

mutation
--------

Contains functions for introducing new genes into the population.

.. automodule:: genopt.mutation
   :members: 
   :undoc-members:
   :show-inheritance: