from data_parser import data_parser
from allocator import allocator
from predictor import predictor
from models import Asset, Factor
import utils as utils
import numpy as np
import plot as plot


OBSERVATION_PERIOD = 5 * 12

MAX_WEIGHT = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
MIN_WEIGHT = [0, 0, 0, 0, 0, 0, 0]

def _getCovariance(returns):
    # This is a simplistic approach to compute covariance from returns.
    # numpy's cov function computes covariance matrix.
    return np.cov(returns, rowvar=False)

def _getCAPMWeights(market_caps):
    total_market_cap = np.sum(market_caps, axis=0)
    capm_weights = market_caps / total_market_cap
    return capm_weights

def historyPredictWithMeanVariance(allAssets, allFactors):
    length = len(next(iter(allAssets.values())).returns)
    start = int(length * 3 / 4)
    returns = []

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
        cov_matrix = _getCovariance([asset.returns[i - OBSERVATION_PERIOD:i] for asset in allAssets.values()])
        weights = allocator.meanVarianceAllocator(np.array(expected_returns), np.array(standard_deviations), cov_matrix)
        
        # Calculate portfolio return for the current month using dot product of weights and actual returns
        month_return = np.dot(weights, current_returns)
        returns.append(month_return)

    return np.array(returns)


def bayesianPredictWithGeneticAllocation(allAssets, allFactors):
    length = len(next(iter(allAssets.values())).returns)
    start = int(length * 3 / 4)
    returns = []

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

        cov_matrix = _getCovariance([asset.returns[i - OBSERVATION_PERIOD:i] for asset in allAssets.values()])
        market_caps = np.array(market_caps)
        capm_weights = _getCAPMWeights(market_caps)
        weights = allocator.aiAllocator(np.array(expected_returns), np.array(standard_deviations), cov_matrix, MIN_WEIGHT, MAX_WEIGHT, capm_weights)
        
        month_return = np.dot(weights, current_returns)
        returns.append(month_return)

    return np.array(returns)

# Initialize allAssets and allFactors with the data from data_parser modules

data_parser.fetch_and_save_all_data()
allAssets = data_parser.readAssetDailyData()

# Plot

# avg_returns, avg_stds, cov_matrix = utils.calculate_averages_and_cov(allAssets)
# market_weights = utils.calculate_market_weights(allAssets)
# plot.plot_mean_variance(avg_returns, cov_matrix)


# allFactors = data_parser.readFactorData()
# portfolio_returns = historyPredictWithMeanVariance(allAssets, allFactors)
# portfolio_returns_ai = bayesianPredictWithGeneticAllocation(allAssets, allFactors)

# test aiPredictor
cur_return, cur_std = predictor.aiPredictor(allAssets['SP500'].returns, allAssets['SP500'].stds, None)
print(cur_return, cur_std)