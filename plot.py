import numpy as np
from typing import Tuple, Generator
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from itertools import product


def plot_mean_variance(avg_returns, cov_matrix):
    """ Plot mean-variance spots for all possible weight combinations. """
    num_assets = len(avg_returns)
    grid = np.arange(0, 1.1, 0.1)
    
    results = []
    for weights in product(grid, repeat=num_assets):
        if sum(weights) == 1:  # Ensure weights sum to 1
            portfolio_return = np.dot(weights, avg_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            results.append((portfolio_variance, portfolio_return))
    
    results = np.array(results)
    plt.scatter(results[:, 0], results[:, 1], c='blue')
    plt.xlabel('Variance')
    plt.ylabel('Expected Return')
    plt.title('Mean-Variance Plot of All Possible Portfolios')
    plt.show()



