from genopt.selection import tournament_selection
import numpy as np
from genopt.crossover import one_way_crossover


def test_tournament_selection_size_2():
    popsize = 2
    t_sel_p = 0.7
    t_sel_size = 2
    fitness = np.arange(popsize)

    wins = 0
    iterations = 1000
    for _ in range(iterations):
        selected = tournament_selection(
            fitness.reshape((-1, 1)), fitness, t_sel_p, t_sel_size
        )
        wins += (selected == 1).sum()

    win_probability = wins / (popsize * iterations)
    assert t_sel_p - 0.03 < win_probability < t_sel_p + 0.03


def test_tournament_selection_size_5():
    popsize = 5
    t_sel_p = 0.7
    t_sel_size = 5
    fitness = np.arange(popsize)

    wins = np.zeros(popsize)
    iterations = 1000
    for _ in range(iterations):
        selected = tournament_selection(
            fitness.reshape((-1, 1)), fitness, t_sel_p, t_sel_size
        )

        for i in range(popsize):
            wins[i] += np.sum(selected == i)

    wins /= iterations * t_sel_size

    prob_thresholds = [t_sel_p * (1 - t_sel_p) ** i for i in range(t_sel_size - 1)]
    prob_thresholds.append(1 - sum(prob_thresholds))

    diff = 1e-2
    for i, win in enumerate(wins):
        prob = prob_thresholds[-(i + 1)]
        assert prob - diff < win < prob + diff
