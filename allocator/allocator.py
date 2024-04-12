import numpy as np
from typing import Tuple, Generator
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def geneticAllocator(expected_returns: np.ndarray, std_deviations: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """
    Determine optimal asset weights using a genetic algorithm.

    :param expected_returns: NumPy array of predicted expected returns.
    :param std_deviations: NumPy array of predicted standard deviations.
    :param cov_matrix: NumPy array of predicted covariance matrix.
    :return: NumPy array of optimal weights.
    """
    pass

def meanVarianceAllocator(expected_returns: np.ndarray, 
                          std_deviations: np.ndarray, 
                          cov_matrix: np.ndarray) -> np.ndarray:
    """
    Determine optimal asset weights using the algebraic mean-variance method.

    :param expected_returns: NumPy array of predicted expected returns.
    :param std_deviations: NumPy array of predicted standard deviations.
    :param cov_matrix: NumPy array of predicted covariance matrix.
    :return: NumPy array of optimal weights.
    """

    # Number of assets
    num_assets = len(expected_returns)

    # Portfolio standard deviation function
    def portfolio_std(weights: np.ndarray) -> float:
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Constraints for the optimization:
    # 1. The sum of weights is 1 (fully invested portfolio).
    # 2. No short selling (weights are between 0 and 1).
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                   {'type': 'ineq', 'fun': lambda x: x})

    # Initial guess for the weights
    initial_weights = np.ones(num_assets) / num_assets

    # Bounds for each weight to enforce no short selling
    bounds = tuple((0, 1) for asset in range(num_assets))

    # Perform the optimization
    result = minimize(portfolio_std, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    # Check if the optimization was successful
    if result.success:
        return result.x
    else:
        raise ValueError('Mean-Variance optimization did not converge')

def plotMeanVariance(expected_returns: np.ndarray, cov_matrix: np.ndarray, margin: float = 0.1):
    """
    Plots all possible portfolios in the mean-variance space.

    :param expected_returns: NumPy array of predicted expected returns.
    :param cov_matrix: NumPy array of predicted covariance matrix.
    :param margin: Step size for the brute force weight generation.
    """
    num_assets = expected_returns.shape[0]
    results = np.zeros((3, int(1/margin)**num_assets))

    counter = 0
    for weights in generate_weights(num_assets, margin):
        if np.sum(weights) == 1:
            returns = np.dot(weights, expected_returns)
            variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            results[0, counter] = variance
            results[1, counter] = returns
            results[2, counter] = returns / np.sqrt(variance)  # Sharpe ratio for plotting
            counter += 1

    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='YlGnBu', marker='o')
    plt.title('Mean-Variance Plot of Possible Portfolios')
    plt.xlabel('Variance')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

def generate_weights(num_assets: int, margin: float) -> Generator[np.ndarray, None, None]:
    """
    Generates all possible combinations of portfolio weights.

    :param num_assets: Number of assets in the portfolio.
    :param margin: Step size for the brute force weight generation.
    :return: Generator object that yields arrays of portfolio weights.
    """
    # This function generates weights for all possible combinations
    # where the sum of the weights is less than or equal to 1.
    from itertools import product
    levels = np.arange(0, 1.01, margin)
    for weights in product(levels, repeat=num_assets):
        if np.sum(weights) <= 1:
            yield np.array(weights)