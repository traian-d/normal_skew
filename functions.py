import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

TOTAL_COL = 'totals'
SEED_COL = 'seeds'
OUTCOME_COL = 'outcomes'


def random_seeds(size, mean, st_dev):
    return np.random.normal(mean, st_dev, size)


def initialize_seeds(size, mean, st_dev):
    return pd.DataFrame({
        SEED_COL: random_seeds(size, mean, st_dev),
        TOTAL_COL: np.zeros(size)
    })


def rands(mean, st_dev):
    return np.random.normal(mean, st_dev, 1)[0]


def sample_outcomes(data, st_dev):
    data[OUTCOME_COL] = np.vectorize(lambda x: rands(x, st_dev))(data[SEED_COL])


def simulate(data, iterations, outcome_std, reward_applicator, reward_function, show=True, save=False, **kwargs):
    plot_freq = iterations // 5
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(iterations):
        if i > 0 and (not i % plot_freq) and show:
            pts, density = plot_score_hist(data, TOTAL_COL)
            ax.plot(pts, density, label=str(i))
        sample_outcomes(data, outcome_std)
        kwargs['data'] = data
        reward_applicator(reward_function, **kwargs)
    ax.legend()
    ax.set_title(f'{reward_function.__name__}')
    if save:
        fig.savefig(f'figures/{reward_function.__name__}.png', dpi=100)
    if show:
        plt.show()
    plt.close()


def snooker_rewards(**kwargs):
    """
    Prizes in GBP
    Winner: 500,000
    Runner-up: 200,000
    Semi-final: 100,000
    Quarter-final: 50,000
    Last 16: 30,000
    Last 32: 20,000
    Last 48: 15,000
    Last 80: 10,000

    out of 128
    """

    size = kwargs['size']

    scores = [1] * size
    finalist = size // 128
    next_two = size * 2 // 128
    next_four = size * 4 // 128
    next_eight = size * 8 // 128
    next_sixteen = size * 16 // 128
    next_thirty_two = size * 32 // 128

    brackets = [0, finalist, finalist, next_two, next_four, next_eight, next_sixteen, next_sixteen, next_thirty_two]
    single_scores = [500, 200, 100, 50, 30, 20, 15, 10]

    end = 0

    for i in range(len(brackets) - 1):
        start = end
        end += brackets[i + 1]
        scores[start: end] = [single_scores[i]] * (end - start)

    return scores


def linear_rewards(**kwargs):
    size = kwargs['size']
    slope = kwargs['slope']
    intercept = kwargs['intercept']
    values = [i for i in reversed(range(size))]
    return [value * slope + intercept for value in values]


def simple_proportional_rewards(**kwargs):
    data = kwargs['data']
    slope = kwargs['slope']
    intercept = kwargs['intercept']
    return data[OUTCOME_COL] * slope + intercept


def exp_proportional_rewards(**kwargs):
    data = kwargs['data']
    return np.exp(data[OUTCOME_COL])


def rank_rewards(reward_function, **kwargs):
    reward_applicator(reward_function, is_rank=True, **kwargs)


def proportional_rewards(reward_function, **kwargs):
    reward_applicator(reward_function, is_rank=False, **kwargs)


def reward_applicator(reward_function, is_rank, **kwargs):
    data = kwargs['data']
    if is_rank:
        data.sort_values(by=[OUTCOME_COL], inplace=True, ascending=False)
    rewards = reward_function(**kwargs)
    data[TOTAL_COL] = data[TOTAL_COL].add(rewards)


def plot_score_hist(data, col_name):
    totals = data[col_name]
    density = gaussian_kde(totals)
    xs = np.linspace(min(0, np.min(totals)), np.max(totals), 300)
    density.covariance_factor = lambda: .25
    density._compute_covariance()
    return xs, density(xs)


if __name__ == "__main__":
    seeds = initialize_seeds(1024, 0, 1)
    # simulate(seeds, 1001, 1, rank_rewards, snooker_rewards, size=1024)
    # simulate(seeds, 1001, 1, rank_rewards, linear_rewards, size=1024, slope=2, intercept=3)
    # simulate(seeds, 1001, 1, proportional_rewards, simple_proportional_rewards, size=1024, slope=2, intercept=3)
    simulate(seeds, 1001, 2, proportional_rewards, exp_proportional_rewards, size=1024)
