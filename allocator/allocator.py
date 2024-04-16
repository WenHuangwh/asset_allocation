import numpy as np
from typing import Tuple, Generator
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Custom similarity function for HallOfFame
def similar(ind1, ind2):
    return np.allclose(ind1, ind2, atol=1e-5)

def aiAllocator(expected_returns: np.ndarray, std_deviations: np.ndarray, cov_matrix: np.ndarray,
                min_weights: np.ndarray, max_weights: np.ndarray, capm_weights: np.ndarray, risk_free_rate: float) -> np.ndarray:
    """
    Determine optimal asset weights using a genetic algorithm, maximizing return/volatility ratio
    while applying a nonlinear penalty for deviations from CAPM weights.
    """

    num_assets = len(expected_returns)

    normalized_capm_weights = capm_weights / np.sum(capm_weights)

    def evaluate(individual):
        weights = np.clip(individual, min_weights, max_weights)
        weights /= np.sum(weights)
        individual[:] = weights
        
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk
        capm_deviation_penalty = np.sum((weights - normalized_capm_weights)**2)
        
        return sharpe_ratio - capm_deviation_penalty,

    # Set up genetic algorithm
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, min_weights, max_weights, num_assets)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.1)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # Enforce constraints directly in the mate and mutate functions
    def mate(ind1, ind2):
        tools.cxBlend(ind1, ind2, alpha=0.1)
        ind1[:] = np.clip(ind1, min_weights, max_weights)
        ind2[:] = np.clip(ind2, min_weights, max_weights)
        ind1[:] /= np.sum(ind1)
        ind2[:] /= np.sum(ind2)
        return ind1, ind2

    def mutate(ind):
        tools.mutGaussian(ind, mu=0, sigma=0.1, indpb=0.1)
        ind[:] = np.clip(ind, min_weights, max_weights)
        ind[:] /= np.sum(ind)
        return ind,

    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1, similar=similar)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 50, stats=stats, halloffame=hof)

    return hof[0]


def meanVarianceAllocator(expected_returns: np.ndarray, cov_matrix: np.ndarray, risk_free_rate: float) -> np.ndarray:
    """
    Determine optimal asset weights using the mean-variance method to maximize the Sharpe ratio.
    
    :param expected_returns: NumPy array of predicted expected returns.
    :param cov_matrix: NumPy array of predicted covariance matrix.
    :return: NumPy array of optimal weights.
    """

    num_assets = len(expected_returns)

    # Objective function: minimize negative Sharpe Ratio
    def objective(weights: np.ndarray) -> float:
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio  # Minimize negative Sharpe to maximize Sharpe

    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights must sum to 1
                   {'type': 'ineq', 'fun': lambda x: x})           # Weights must be non-negative

    # Initial guess and bounds
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Optimization
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        raise ValueError("Mean-Variance optimization did not converge")