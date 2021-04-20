import logging
from typing import Callable, Tuple

import numpy as np

from genopt.crossover import one_way_crossover
from genopt.evaluation import decode_discrete, evaluate
from genopt.mutation import mutation_discrete, mutation_real
from genopt.population import init_discrete, init_real, update_population
from genopt.selection import tournament_selection

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


# pylint: disable=R0902
class GeneticOptimizer:
    """Maximize the function ``objective_function`` using genetic optimization.
    Supports real and discrete encoded variables.

    Args:
        n_vars (int): Number of variables
        popsize (int): Population size, i.e number of chromosomes.
            Needs to be an even number.
        objective_function (callable): Function to optimize for. Takes
            the decoded variables as input and returns a fitness. The
            genetic optimizer will then try to maximize the fitness.
        t_sel_p (float, optional): Probability of the fittest individual
            to win a tournament. Defaults to 0.7.
        t_sel_size (int, optional): Tournament size. Defaults to 1.
        mut_p (float, optional): Mutation probability, set to
            1/(n_vars*var_size) if None. Defaults to None.
        mut_var (int, optional): Mutation variance for real encoded
            variables. Does nothing if encoding='discrete'. Defaults to 1.
        encoding (str, optional): Type of variable encoding. Can be 'real'
            or 'discrete'. Defaults to 'real'.
        var_range (tuple, optional): Range of discrete variables. Does
            nothing if encoding='real'. Defaults to (0, 1).
        var_size (int, optional): Number of genes for each discrete
            variables. Multiplicatively increases chromosome length.
            Does nothing if encoding='real'. Defaults to 1.
        elitism (int, optional): Number of copies of the best
            (maximum fitness) to transfer to the next generation.
            Defaults to 1.
    """

    def __init__(
        self,
        n_vars: int,
        popsize: int,
        objective_function: Callable,
        t_sel_p: float = 0.7,
        t_sel_size: int = 1,
        mut_p: float = None,
        mut_var: float = 1.0,
        encoding: str = "real",
        var_range: Tuple[float] = (0, 1),
        var_size: int = 1,
        elitism: int = 1,
    ):

        # Assertions
        assert encoding in [
            "discrete",
            "real",
        ], "Encoding can only be real or discrete."
        if encoding == "discrete":
            assert (
                var_range[0] < var_range[1]
            ), "First value in range must be smaller than the first"
            assert (
                isinstance(var_size, int) and var_size > 0
            ), "Variable size must be an integer larger than 0"

        self.n_vars = n_vars
        self.objective_function = objective_function
        self.t_sel_p = t_sel_p
        self.t_sel_size = t_sel_size
        self.encoding = encoding
        if encoding == "real":
            self.var_size = 1
        else:
            self.var_size = var_size
        self.var_range = var_range
        if mut_p is None:
            self.mut_p = 1 / (self.n_vars * self.var_size)
        else:
            self.mut_p = mut_p
        self.mut_var = mut_var
        self.elitism = elitism
        self.fitness = np.zeros(popsize)
        self.top_individual = None

        # Initialize population
        if encoding == "real":
            self.population = init_real(popsize, self.n_vars)
        else:
            self.population = init_discrete(popsize, self.n_vars, self.var_size)

    def optimize(self, n_gen: int) -> np.ndarray:
        """Run the genetic optimizer for n_gen generations and return the
        top individual of the population

        Args:
            n_gen (int): Number of generations to optimize for

        Returns:
            np.ndarray: Chromosome of the top individual
        """
        for i in range(n_gen):

            # Evaluate population
            self.fitness = self.evaluate(self.population)
            i_max = np.argmax(self.fitness)
            self.top_individual = self.population[i_max, :]

            # Selection
            tmp_population = self.select(self.population)

            # Crossover
            tmp_population = self.crossover(tmp_population)

            # Mutation
            tmp_population = self.mutate(tmp_population)

            # Put in top individual to make sure performance never drops
            self.population = update_population(
                tmp_population, self.top_individual, self.elitism
            )
            LOGGER.info(f"Generation: {i} - Max fitness: {self.fitness.max()}")
        return self.top_individual

    def decode(self, population: np.ndarray) -> np.ndarray:
        """Decode binary chromosomes to an array of discrete values

        Args:
            population (np.ndarray): Population of binary chromosomes as a 2d
                array of shape (popsize, n_vars * var_size)

        Returns:
            np.ndarray: Decoded population as 2d array of shape (popsize, n_vars)
        """
        if self.encoding == "real":
            return population

        return decode_discrete(
            population,
            self.n_vars,
            self.var_range,
            self.var_size,
        )

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        """Evaluate population fitness using the objective function.
        Decodes if necessary.

        Args:
            population (np.ndarray): Population as a 2d array of shape
                (popsize, n_vars * var_size)

        Returns:
            np.ndarray: Population fitness
        """
        variables = self.decode(population)
        return evaluate(
            variables,
            self.objective_function,
        )

    def select(self, population: np.ndarray) -> np.ndarray:
        """Randomly select high fitness individuals

        Args:
            population (np.ndarray): Population as a 2d array of shape
                (popsize, n_vars * var_size)

        Returns:
            np.ndarray: Population after selection
        """
        return tournament_selection(
            population, self.fitness, self.t_sel_p, self.t_sel_size
        )

    @staticmethod
    def crossover(population: np.ndarray) -> np.ndarray:
        """Mix the chromosomes of the population

        Args:
            population (np.ndarray): Population as a 2d array of shape
                (popsize, n_vars * var_size)

        Returns:
            np.ndarray: Population after crossover
        """
        return one_way_crossover(population)

    def mutate(self, population: np.ndarray) -> np.ndarray:
        """Mutate the population to introduce new chromosomes to the pool

        Args:
            population (np.ndarray): Population as a 2d array of shape
                (popsize, n_vars * var_size)

        Returns:
            np.ndarray: Population after mutation
        """
        if self.encoding == "real":
            return mutation_real(population, self.mut_p, self.mut_var)

        return mutation_discrete(
            population,
            self.mut_p,
        )
