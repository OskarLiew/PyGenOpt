import numpy as np

from genopt.crossover import one_way_crossover
from genopt.evaluation import decode, evaluate_pop
from genopt.mutation import random_mutation
from genopt.population import random_init, update_population
from genopt.selection import tournament_selection


# pylint: disable=R0902
class GeneticOptimizer:
    def __init__(
        self,
        n_vars,
        popsize,
        n_gen,
        objective_function,
        t_sel_p=0.7,
        t_sel_size=1,
        mut_p=None,
        mut_var=1,
        encoding="real",
        var_range=(0, 1),
        var_size=1,
        elitism=1,
    ):
        """Maximize the function ``objective_function`` using genetic optimization.
        Supports real and discrete encoded variables.

        Args:
            n_vars (int): Number of variables
            popsize (int): Population size, i.e number of chromosomes.
                Needs to be an even number.
            n_gen (int): Number of generations for the optimization to run
            objective_function (callable): Function to optimize for. Takes
                the decoded variables as input and returns a fitness. The
                genetic optimizer will then try to maximize the fitness.
            t_sel_p (float, optional): Probability of the fittest individual
                to win a tournament. Defaults to 0.7.
            t_sel_size (int, optional): Tournament size. Defaults to 1.
            mut_p ([type], optional): Mutation probability, set to
                1/(n_vars*var_size) if None. Defaults to None.
            mut_var (int, optional): Mutation variance for real encoded
                variables. Does nothing if encoding='discrete'. Defaults to 1.
            encoding (str, optional): Type of variable encoding. Can be 'real'
                or 'discrete'. Defaults to 'real'.
            var_range (tuple, optional): Range of discrete variables. Does
                nothing if encoding='real'. Defaults to (0,1).
            var_size (int, optional): Number of genes for each discrete
                variables. Multiplicatively increases chromosome length.
                Does nothing if encoding='real'. Defaults to 1.
            elitism (int, optional): Number of copies of the best
                (maximum fitness) to transfer to the next generation.
                Defaults to 1.
        """

        # Assertions
        assert popsize % 2 == 0, "Popsize should be an even number"
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
        self.popsize = popsize
        self.n_gen = n_gen
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
        self.fitness = np.zeros(self.popsize)
        self.top_individual = None

        # Initialize population
        self.population = random_init(
            self.n_vars, self.popsize, self.encoding, self.var_size
        )

    def optimize(self):
        for i in range(self.n_gen):

            # Evaluate population
            self.fitness, i_max = self.evaluate_pop(self.population)
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
            print("Generation:", i, " - Max fitness:", self.fitness.max())
        return self.top_individual

    def decode(self, population: np.ndarray) -> np.ndarray:
        """Decode binary chromosomes to real arrays

        Args:
            population (np.ndarray): Population as a 2d array of shape
                (popsize, n_vars * var_size)

        Returns:
            np.ndarray: Decoded population as 2d array of shape (popsize, n_vars)
        """
        return decode(
            population,
            self.n_vars,
            self.encoding,
            self.var_range,
            self.var_size,
        )

    def evaluate_pop(self, population: np.ndarray) -> np.ndarray:
        """Return population fitness using the objective function. Decodes if necessary.

        Args:
            population (np.ndarray): Population as a 2d array of shape
                (popsize, n_vars * var_size)

        Returns:
            np.ndarray: Population fitness
        """
        return evaluate_pop(
            population,
            self.n_vars,
            self.objective_function,
            self.var_range,
            self.var_size,
            self.encoding,
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
            population, self.fitness, self.popsize, self.t_sel_p, self.t_sel_size
        )

    def crossover(self, population: np.ndarray) -> np.ndarray:
        """Mix the chromosomes of the population

        Args:
            population (np.ndarray): Population as a 2d array of shape
                (popsize, n_vars * var_size)

        Returns:
            np.ndarray: Population after crossover
        """
        return one_way_crossover(population, self.n_vars, self.popsize)

    def mutate(self, population: np.ndarray) -> np.ndarray:
        """Mutate the population to introduce new chromosomes to the pool

        Args:
            population (np.ndarray): Population as a 2d array of shape
                (popsize, n_vars * var_size)

        Returns:
            np.ndarray: Population after mutation
        """
        return random_mutation(
            population,
            self.n_vars,
            self.popsize,
            self.mut_p,
            self.mut_var,
            self.encoding,
            self.var_size,
        )
