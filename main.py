from data_parser import data_parser
from allocator import allocator
from predictor import predictor
from models import Asset, Factor
import utils as utils
import numpy as np
import plot as plot
import yfinance as yf


OBSERVATION_PERIOD = 5 * 12

MAX_WEIGHT = []
MIN_WEIGHT = []

def _getCAPMWeights(market_caps):
    total_market_cap = np.sum(market_caps, axis=0)
    capm_weights = market_caps / total_market_cap
    return capm_weights

def historyPredictWithMeanVariance(allAssets, allFactors):
    length = len(next(iter(allAssets.values())).returns)
    start = int(length * 3 / 4)
    portfolio_returns = []
    portfolio_std_devs = []

    # Iterate over each time point from start to the end of the dataset
    for i in range(start, length):
        expected_returns = []
        standard_deviations = []
        current_returns = []

        # Collect data for all assets up to the current time point
        for asset_name, asset in allAssets.items():
            # Assume predictor.historyPredictor returns the expected return and std for the next period
            cur_return, cur_std = predictor.historyPredictor(asset.returns[i - OBSERVATION_PERIOD:i], asset.stds[i - OBSERVATION_PERIOD:i])
            expected_returns.append(cur_return)
            standard_deviations.append(cur_std)
            current_returns.append(asset.returns[i])

        # Get covariance matrix from the returns up to the current time
        cov_matrix = utils.calculate_cov(allAssets, OBSERVATION_PERIOD)

        weights = allocator.meanVarianceAllocator(np.array(expected_returns), cov_matrix, riskFreeAssets.returns[i - 1])
        
        # Calculate portfolio return for the current month using dot product of weights and actual returns
        month_return = np.dot(weights, current_returns)
        portfolio_returns.append(month_return)
        month_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        portfolio_std_devs.append(month_std_dev)

    return np.array(portfolio_returns), np.array(portfolio_std_devs)

def MarketWeightAllocation(allAssets, allFactors):
    length = len(next(iter(allAssets.values())).returns)
    start = int(length * 3 / 4)
    portfolio_returns = []
    portfolio_std_devs = []

    # Iterate over each time point from start to the end of the dataset
    for i in range(start, length):
        current_returns = []

        # Collect data for all assets up to the current time point
        for asset_name, asset in allAssets.items():
            current_returns.append(asset.returns[i])

        # Get covariance matrix from the returns up to the current time
        cov_matrix = utils.calculate_cov(allAssets, OBSERVATION_PERIOD)

        weights = utils.calculate_market_weights(allAssets, i)
        
        # Calculate portfolio return for the current month using dot product of weights and actual returns
        month_return = np.dot(weights, current_returns)
        portfolio_returns.append(month_return)
        month_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        portfolio_std_devs.append(month_std_dev)

    return np.array(portfolio_returns), np.array(portfolio_std_devs)

def historyPredictWithGeneticAllocation(allAssets, allFactors):
    MAX_WEIGHT = np.array([1 for _ in range(len(allAssets))])
    MIN_WEIGHT = np.array([0 for _ in range(len(allAssets))])
    length = len(next(iter(allAssets.values())).returns)
    start = int(length * 3 / 4)
    portfolio_returns = []
    portfolio_std_devs = []

    # Iterate over each time point from start to the end of the dataset
    for i in range(start, length):
        expected_returns = []
        standard_deviations = []
        current_returns = []

        # Collect data for all assets up to the current time point
        for asset_name, asset in allAssets.items():
            # Assume predictor.historyPredictor returns the expected return and std for the next period
            cur_return, cur_std = predictor.historyPredictor(asset.returns[i - OBSERVATION_PERIOD:i], asset.stds[i - OBSERVATION_PERIOD:i])
            expected_returns.append(cur_return)
            standard_deviations.append(cur_std)
            current_returns.append(asset.returns[i])

        # Get covariance matrix from the returns up to the current time
        cov_matrix = utils.calculate_cov(allAssets, OBSERVATION_PERIOD)
        capm_weights = utils.calculate_market_weights(allAssets, i)
        weights = allocator.aiAllocator(np.array(expected_returns), np.array(standard_deviations), cov_matrix, MIN_WEIGHT, MAX_WEIGHT, capm_weights, riskFreeAssets.returns[i - 1])
        
        # Calculate portfolio return for the current month using dot product of weights and actual returns
        month_return = np.dot(weights, current_returns)
        portfolio_returns.append(month_return)
        month_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        portfolio_std_devs.append(month_std_dev)

    return np.array(portfolio_returns), np.array(portfolio_std_devs)


def bayesianPredictWithGeneticAllocation(allAssets, allFactors):
    length = len(next(iter(allAssets.values())).returns)
    start = int(length * 3 / 4)
    portfolio_returns = []
    portfolio_std_devs = []


    for i in range(start, length):
        expected_returns = []
        standard_deviations = []
        current_returns = []
        market_caps = []

        for asset in allAssets.values():
            cur_return, cur_std = predictor.aiPredictor(asset.returns[i - OBSERVATION_PERIOD:i], asset.stds[i - OBSERVATION_PERIOD:i], [factor.data[i - OBSERVATION_PERIOD:i] for factor in allFactors.values()])
            expected_returns.append(cur_return)
            standard_deviations.append(cur_std)
            current_returns.append(asset.returns[i])
            market_caps.append(asset.market_cap[i - 1])

        cov_matrix = utils.calculate_cov(allAssets, OBSERVATION_PERIOD)
        capm_weights = utils.calculate_market_weights(allAssets, i)
        weights = allocator.aiAllocator(np.array(expected_returns), np.array(standard_deviations), cov_matrix, MIN_WEIGHT, MAX_WEIGHT, capm_weights)
        
        month_return = np.dot(weights, current_returns)
        portfolio_returns.append(month_return)
        month_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        portfolio_std_devs.append(month_std_dev)

    return np.array(portfolio_returns), np.array(portfolio_std_devs)

# Initialize allAssets and allFactors with the data from data_parser modules

# data_parser.fetch_and_save_all_data()
allAssets = data_parser.readAssetDailyData()
riskFreeAssets = data_parser.readRiskFreeData()
returns_H_G, std_H_G = historyPredictWithGeneticAllocation(allAssets, None)
returns_H_M, std_H_M = historyPredictWithMeanVariance(allAssets, None)


plot.plot_returns_and_std(returns_H_M, std_H_M, returns_H_G, std_H_G)
plot.plot_time_series(returns_H_M, std_H_M, returns_H_G, std_H_G)
plot.plot_geometric_cumulative_returns(returns_H_M, returns_H_G)

# Plot

# cov_matrix = utils.calculate_cov(allAssets, OBSERVATION_PERIOD)
# returns = [asset.returns[-1] for asset in allAssets.values()]
# market_weights = utils.calculate_market_weights(allAssets)
# plot.plot_mean_variance(returns, cov_matrix)
# plot.plot_mean_variance_with_fitness(returns, cov_matrix, market_weights, MIN_WEIGHT, MAX_WEIGHT)

# allFactors = data_parser.readFactorData()
# portfolio_returns = historyPredictWithMeanVariance(allAssets, allFactors)
# portfolio_returns_ai = bayesianPredictWithGeneticAllocation(allAssets, allFactors)
