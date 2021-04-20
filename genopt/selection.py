import numpy as np


def tournament_selection(
    population: np.ndarray, fitness: np.ndarray, t_sel_p: float, t_sel_size: int
) -> np.ndarray:
    """Select fit individuals from a population by tournament selection.
    In tournament selection a group from the population are put into tournaments
    of two on two "competition". The individual with the highest fitness will have
    ``t_sel_p`` chance to win.

    Args:
        population (np.ndarray): Population of chromosomes
        fitness (np.ndarray): Fitness scores for the chromosomes
        t_sel_p (float): Probability of the fittest individual to win a
            competition. Must be >0.5 in order for fitness to increase in the
            new population
        t_sel_size (int): Tournament size by number of participants. Should be
            2 or more.

    Returns:
        np.ndarray: Population after selection of tournament winners.
    """
    # Couldn't figure out a way to make it less messy without a bunch of loops
    popsize = population.shape[0]

    # Select t_sel_size random individuals for each tournament without replacement
    selected = np.array(
        [
            np.random.choice(popsize, size=t_sel_size, replace=False)
            for _ in range(popsize)
        ]
    )
    # Save their sorted indices
    i_sel = np.argsort(-np.take(fitness, selected), axis=1)

    # Select the best in t_sel_p of cases, nr 2 in t_sel_p*(1-t_sel_p)^1 of cases etc
    prob_thresholds = np.cumsum(
        [t_sel_p * (1 - t_sel_p) ** i for i in range(t_sel_size - 1)]
    )
    # Reshape probability thresholds so they can be broadcasted
    prob_thresholds = np.tile(np.append(prob_thresholds, 1), popsize).reshape(
        (-1, t_sel_size)
    )
    # Sample a random number in [0, 1] and see the threshold it lands in
    prob_diffs = prob_thresholds - np.random.rand(popsize).reshape((-1, 1))
    i_winner = np.argmax(np.minimum(prob_diffs, 0), axis=1)

    # Return the new population
    i_sel = i_sel[np.arange(popsize), i_winner]
    selected = selected[np.arange(popsize), i_sel]
    return population[selected, :]
