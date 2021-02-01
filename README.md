# PyGenOpt

A simple package for genetic optimization in python.

## Installation

To install from source code:

`git clone https://github.com/OskarLiew/PyGenOpt.git`
`pip install PyGenOpt`

Pip package might come later.

## Usage

Example code to maximize a very simple function:

```python
from genopt import GeneticOptimizer

def objective(arr):
    return sum(arr)

go = GeneticOptimizer(
        n_vars=2,
        popsize=100,
        n_gen=50,
        objective_function=objective,
        t_sel_p=0.7,
        t_sel_size=1,
        mut_p=None,
        mut_var=1,
        encoding="real",
        var_range=(0, 1),
        var_size=1,
        crossover_points=1,
        elitism=1,
)
result = go.optimize()
```
