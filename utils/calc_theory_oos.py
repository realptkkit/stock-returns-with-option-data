import numpy as np


def calculate_theory_performance(data, start, end):
    test_frame = data[(data.date >= start) & (data.date < end)]
    obs = len(test_frame)
    returns = test_frame["target_eret"].to_numpy()
    predictions = test_frame["theory_eret"].to_numpy()
    rss = np.sum((returns-predictions)**2)
    tss = np.sum(returns**2)

    oos = 1 - rss/tss

    return obs, oos, rss, tss
