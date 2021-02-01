import numpy as np


def tournament_selection(population, fitness, popsize, t_sel_p, t_sel_size):
    # Couldn't figure out a way to make it less messy without despicable for-loops
    k = t_sel_size

    # Select k random individuals for each tournament
    selected = np.random.randint(popsize, size=(popsize, k))
    # Save their sorted indices
    i_sel = np.argsort(-np.take(fitness, selected), axis=1)

    # Select the best in t_sel_p of cases, nr 2 in t_sel_p*(1-t_sel_p)^1 of cases etc
    probs = np.array([t_sel_p * (1 - t_sel_p) ** i for i in range(k - 1)])
    probs = np.tile(np.append(np.cumsum(probs), 1), popsize).reshape((-1, k))
    i_winner = np.argmax(np.random.rand(popsize).reshape((-1, 1)) < probs, axis=1)

    # Return the new population
    i_sel = i_sel[np.arange(popsize), i_winner]
    selected = selected[np.arange(popsize), i_sel]
    return population[selected, :]
