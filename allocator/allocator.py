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