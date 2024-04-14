import numpy as np
from typing import Tuple, Generator
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

def aiAllocator(expected_returns: np.ndarray, std_deviations: np.ndarray, cov_matrix: np.ndarray,
                min_weights: np.ndarray, max_weights: np.ndarray, capm_weights: np.ndarray) -> np.ndarray:
    """
    Determine optimal asset weights using a genetic algorithm, maximizing return/volatility ratio 
    while applying a nonlinear penalty for deviations from CAPM weights.

    :param expected_returns: Predicted returns for each asset.
    :param std_deviations: Standard deviations for each asset.
    :param cov_matrix: Covariance matrix for all assets.
    :param min_weights: Minimum allowed weights for each asset.
    :param max_weights: Maximum allowed weights for each asset.
    :param capm_weights: Market value weights for each asset, used for the penalty function.
    :return: Optimal portfolio weights.
    """
    num_assets = len(expected_returns)

    # Normalize CAPM weights to ensure they sum to 1
    capm_weights /= np.sum(capm_weights)

    # Fitness function definition
    def fitness(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk
        capm_deviation = np.sum((weights - capm_weights) ** 2)

        # Enhance penalty non-linearly
        penalty = np.sum(capm_deviation ** 2)

        return sharpe_ratio - penalty,

    # Genetic Algorithm Setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attr_float", np.random.uniform, 0.01, 1)
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, num_assets)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operator registering
    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=min_weights, up=max_weights, eta=2.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=min_weights, up=max_weights, eta=1.0, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Enforcing sum of weights to 1 using a decorator
    def check_sum(weights):
        return np.sum(weights) == 1

    toolbox.decorate("mate", tools.staticLimit(key=check_sum, max_value=1))
    toolbox.decorate("mutate", tools.staticLimit(key=check_sum, max_value=1))

    # Algorithm parameters
    pop_size = 300
    num_gen = 50
    cxpb, mutpb = 0.5, 0.2

    # Run the genetic algorithm
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb, mutpb, num_gen, stats=stats, halloffame=hof)

    return np.array(hof[0])  # Return the best individual as the optimal portfolio


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


def plot_all_possible_portfolios(expected_returns: np.ndarray, cov_matrix: np.ndarray):
    """
    Plot all possible portfolio combinations on a mean-variance plot to show the linear solutions.

    :param expected_returns: np.ndarray - Predicted expected returns for assets.
    :param cov_matrix: np.ndarray - Covariance matrix for assets.
    """
    num_assets = len(expected_returns)
    step = 0.05  # Define the step size for weight generation

    results = []  # Store results for plotting

    # Generate and evaluate all possible weight combinations
    for weights in generate_weights(num_assets, step):
        if sum(weights) == 1:  # Ensure the weights sum to 1
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            results.append((portfolio_variance, portfolio_return))
    
    results = np.array(results)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(results[:, 0], results[:, 1], c='blue', marker='o', label='All Portfolios')
    plt.title('Mean-Variance Plot of All Possible Portfolios')
    plt.xlabel('Portfolio Variance')
    plt.ylabel('Expected Portfolio Return')
    plt.grid(True)
    plt.legend()
    plt.show()

def generate_weights(num_assets: int, step: float):
    """
    Generates weights that sum to 1 with a given step size.

    :param num_assets: Number of assets.
    :param step: Step size for generating weights.
    """
    from itertools import product
    grid = np.arange(0, 1 + step, step)
    for weights in product(grid, repeat=num_assets):
        if np.sum(weights) <= 1:
            # Normalize weights to sum to 1
            normalized_weights = np.array(weights) / np.sum(weights)
            yield normalized_weights
