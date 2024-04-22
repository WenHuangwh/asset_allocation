import numpy as np
from typing import Tuple, Generator
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from itertools import product
import utils as utils

def plot_mean_variance(returns, cov_matrix):
    """ Plot mean-variance spots for all possible weight combinations. """
    num_assets = len(returns)
    grid = np.arange(0, 1.1, 0.05)
    
    results = []
    for weights in product(grid, repeat=num_assets):
        if sum(weights) == 1:  # Ensure weights sum to 1
            portfolio_return = np.dot(weights, returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            results.append((portfolio_variance, portfolio_return))
    
    results = np.array(results)
    plt.scatter(results[:, 0], results[:, 1], c='blue')
    plt.xlabel('Variance')
    plt.ylabel('Expected Return')
    plt.title('Mean-Variance Plot of All Possible Portfolios')
    plt.show()

def plot_mean_variance_with_fitness(returns, cov_matrix, capm_weights, min_weights, max_weights):
    num_assets = len(returns)
    valid_weights = generate_valid_weights(num_assets, step=0.05, min_weights=min_weights, max_weights=max_weights)
    
    results = []
    for weights in valid_weights:
        # Convert tuple weights to numpy array for processing
        weights_array = np.array(weights)
        fitness_value = fitness(weights_array, returns, cov_matrix, capm_weights)
        portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
        results.append((portfolio_variance, fitness_value[0]))
    
    results = np.array(results)
    plt.scatter(results[:, 0], results[:, 1], c='blue')
    plt.xlabel('Variance')
    plt.ylabel('Fitness Value')
    plt.title('Fitness-Variance of All Possible Portfolios')
    plt.show()

def generate_valid_weights(num_assets, step=0.04, min_weights=None, max_weights=None):
    """ Generate all valid weights within given constraints. """
    grid = np.arange(0, 1 + step, step)
    for weights in product(grid, repeat=num_assets):
        if np.isclose(sum(weights), 1, atol=0.05):  # Allow some tolerance
            if all(min_weights[i] <= weights[i] <= max_weights[i] for i in range(num_assets)):
                yield weights

def fitness(weights, expected_returns, cov_matrix, capm_weights):
    """ Calculate the fitness of a portfolio configuration. """
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_risk
    capm_deviation = np.sum((weights - capm_weights) ** 2)
    
    # Enhance penalty non-linearly
    penalty = np.sum(capm_deviation ** 2)
    
    return sharpe_ratio - penalty,

import numpy as np
import matplotlib.pyplot as plt

def plot_all_returns_and_std(returns_MV, std_MV, returns_GA, std_GA, returns_CAPM, std_CAPM, returns_HG, std_HG):
    plt.figure(figsize=(14, 7))

    # Plot Mean-Variance results
    plt.scatter(std_MV, returns_MV, c='blue', marker='o', label='AI Mean-Variance')

    # Plot Genetic Algorithm results
    plt.scatter(std_GA, returns_GA, c='green', marker='x', label='AI Genetic Algorithm')

    # Plot CAPM results
    plt.scatter(std_CAPM, returns_CAPM, c='red', marker='s', label='CAPM')

    # Plot Historical Genetic results
    plt.scatter(std_HG, returns_HG, c='purple', marker='^', label='Historical Genetic')

    plt.title('Portfolio Optimization Comparison')
    plt.xlabel('Portfolio Standard Deviation (Risk)')
    plt.ylabel('Portfolio Return')
    plt.legend(loc='best')
    plt.show()

def plot_all_time_series(returns_MV, std_MV, returns_GA, std_GA, returns_CAPM, std_CAPM, returns_HG, std_HG):
    t = np.arange(len(returns_MV))  # Assuming all series are of the same length

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plotting returns for all strategies
    ax1.set_xlabel('Time (months)')
    ax1.set_ylabel('Portfolio Returns')
    ax1.plot(t, returns_MV, label='AI Mean-Variance', color='blue')
    ax1.plot(t, returns_GA, label='AI Genetic Algorithm', color='green')
    ax1.plot(t, returns_CAPM, label='CAPM', color='red')
    ax1.plot(t, returns_HG, label='Historical Genetic', color='purple')
    ax1.legend(loc='upper left')

    # Creating a second y-axis for standard deviations
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Portfolio Standard Deviation (Risk)')
    ax2.plot(t, std_MV, label='AI Mean-Variance Std Dev', linestyle='--', color='blue')
    ax2.plot(t, std_GA, label='AI Genetic Algorithm Std Dev', linestyle='--', color='green')
    ax2.plot(t, std_CAPM, label='CAPM Std Dev', linestyle='--', color='red')
    ax2.plot(t, std_HG, label='Historical Genetic Std Dev', linestyle='--', color='purple')
    ax2.legend(loc='upper right')

    plt.title('Portfolio Optimization Comparison Over Time')
    plt.show()

def plot_all_geometric_cumulative_returns(returns_MV, returns_GA, returns_CAPM, returns_HG):
    # Convert returns to growth factors
    growth_factors_MV = 1 + np.array(returns_MV)
    growth_factors_GA = 1 + np.array(returns_GA)
    growth_factors_CAPM = 1 + np.array(returns_CAPM)
    growth_factors_HG = 1 + np.array(returns_HG)

    # Calculate cumulative returns
    cumulative_returns_MV = np.cumprod(growth_factors_MV) - 1
    cumulative_returns_GA = np.cumprod(growth_factors_GA) - 1
    cumulative_returns_CAPM = np.cumprod(growth_factors_CAPM) - 1
    cumulative_returns_HG = np.cumprod(growth_factors_HG) - 1

    t = np.arange(len(returns_MV))

    plt.figure(figsize=(14, 7))
    plt.plot(t, cumulative_returns_MV, label='AI Mean-Variance', color='blue')
    plt.plot(t, cumulative_returns_GA, label='AI Genetic Algorithm', color='green')
    plt.plot(t, cumulative_returns_CAPM, label='CAPM', color='red')
    plt.plot(t, cumulative_returns_HG, label='Historical Genetic', color='purple')
    
    plt.title('Geometric Cumulative Portfolio Returns Over Time')
    plt.xlabel('Time (months)')
    plt.ylabel('Geometric Cumulative Returns')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


