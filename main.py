from data_parser import data_parser
from allocator import allocator
from predictor import predictor
from models import Asset, Factor
import utils as utils
import numpy as np
import plot as plot
import yfinance as yf


OBSERVATION_PERIOD = 5 * 12
PREDICTION_PERIOD = 5 * 12

MAX_WEIGHT = []
MIN_WEIGHT = []

def _getCAPMWeights(market_caps):
    total_market_cap = np.sum(market_caps, axis=0)
    capm_weights = market_caps / total_market_cap
    return capm_weights



def marketWeightAllocation(allAssets, allFactors):
    portfolio_returns = []
    portfolio_std_devs = []
    # Iterate over each time point from start to the end of the dataset
    for i in range(start, start + PREDICTION_PERIOD):
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
    portfolio_returns = []
    portfolio_std_devs = []

    # Iterate over each time point from start to the end of the dataset
    for i in range(start, start + PREDICTION_PERIOD):
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
        weights = allocator.aiAllocator(np.array(expected_returns), np.array(standard_deviations), cov_matrix, MIN_WEIGHT, MAX_WEIGHT, capm_weights, riskFreeAssets.returns[i - 1], 0.2)
        
        # Calculate portfolio return for the current month using dot product of weights and actual returns
        month_return = np.dot(weights, current_returns)
        portfolio_returns.append(month_return)
        month_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        portfolio_std_devs.append(month_std_dev)

    return np.array(portfolio_returns), np.array(portfolio_std_devs)

# def historyPredictWithMeanVariance(allAssets, allFactors):
#     length = len(next(iter(allAssets.values())).returns)
#     start = int(length * 3 / 4)
#     portfolio_returns = []
#     portfolio_std_devs = []

#     # Iterate over each time point from start to the end of the dataset
#     for i in range(start, start + PREDICTION_PERIOD):
#         expected_returns = []
#         standard_deviations = []
#         current_returns = []

#         # Collect data for all assets up to the current time point
#         for asset_name, asset in allAssets.items():
#             # Assume predictor.historyPredictor returns the expected return and std for the next period
#             cur_return, cur_std = predictor.aiPredictor(asset.returns[i - OBSERVATION_PERIOD:i], asset.stds[i - OBSERVATION_PERIOD:i], [factor.data[i - OBSERVATION_PERIOD:i] for factor in allFactors.values()], i - OBSERVATION_PERIOD - 1)
#             expected_returns.append(cur_return)
#             standard_deviations.append(cur_std)
#             current_returns.append(asset.returns[i])

#         # Get covariance matrix from the returns up to the current time
#         cov_matrix = utils.calculate_cov(allAssets, OBSERVATION_PERIOD)

#         weights = allocator.meanVarianceAllocator(np.array(expected_returns), cov_matrix, riskFreeAssets.returns[i - 1])
        
#         # Calculate portfolio return for the current month using dot product of weights and actual returns
#         month_return = np.dot(weights, current_returns)
#         portfolio_returns.append(month_return)
#         month_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
#         portfolio_std_devs.append(month_std_dev)

#     return np.array(portfolio_returns), np.array(portfolio_std_devs)

# def AIPredictWithGeneticAllocation(allAssets, allFactors):
#     MAX_WEIGHT = np.array([1 for _ in range(len(allAssets))])
#     MIN_WEIGHT = np.array([0 for _ in range(len(allAssets))])
#     length = len(next(iter(allAssets.values())).returns)
#     start = int(length * 3 / 4)
#     portfolio_returns = []
#     portfolio_std_devs = []

#     for i in range(start, start + PREDICTION_PERIOD):
#         expected_returns = []
#         standard_deviations = []
#         current_returns = []
#         market_caps = []

#         for asset in allAssets.values():
#             cur_return, cur_std = predictor.aiPredictor(asset.returns[i - OBSERVATION_PERIOD:i], asset.stds[i - OBSERVATION_PERIOD:i], [factor.data[i - OBSERVATION_PERIOD:i] for factor in allFactors.values()], i - OBSERVATION_PERIOD - 1)
#             expected_returns.append(cur_return)
#             standard_deviations.append(cur_std)
#             current_returns.append(asset.returns[i])
#             market_caps.append(asset.market_cap[i - 1])

#         # Get covariance matrix from the returns up to the current time
#         cov_matrix = utils.calculate_cov(allAssets, OBSERVATION_PERIOD)
#         capm_weights = utils.calculate_market_weights(allAssets, i)
#         weights = allocator.aiAllocator(np.array(expected_returns), np.array(standard_deviations), cov_matrix, MIN_WEIGHT, MAX_WEIGHT, capm_weights, riskFreeAssets.returns[i - 1])
        
#         # Calculate portfolio return for the current month using dot product of weights and actual returns
#         month_return = np.dot(weights, current_returns)
#         portfolio_returns.append(month_return)
#         month_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
#         portfolio_std_devs.append(month_std_dev)

#     return np.array(portfolio_returns), np.array(portfolio_std_devs)


def predictAndAllocateAll(allAssets, allFactors):
    portfolio_returns_MV = []
    portfolio_std_devs_MV = []
    portfolio_returns_GA = []
    portfolio_std_devs_GA = []
    
    MAX_WEIGHT = np.array([1 for _ in range(len(allAssets))])
    MIN_WEIGHT = np.array([0 for _ in range(len(allAssets))])

    for i in range(start, start + PREDICTION_PERIOD):
        expected_returns = []
        standard_deviations = []
        current_returns = []
        market_caps = []

        total_loss = 0

        for asset_name, asset in allAssets.items():
            try:
                cur_return, cur_std, cur_loss = predictor.aiPredictor(
                    asset.returns[i - OBSERVATION_PERIOD:i], 
                    asset.stds[i - OBSERVATION_PERIOD:i], 
                    [factor.data[i - OBSERVATION_PERIOD:i] for factor in allFactors.values()], 
                    i - OBSERVATION_PERIOD - 1
                )
            except Exception as e:
                try:
                    cur_return, cur_std, cur_loss = predictor.aiPredictor(
                        asset.returns[i - OBSERVATION_PERIOD:i], 
                        asset.stds[i - OBSERVATION_PERIOD:i], 
                        [factor.data[i - OBSERVATION_PERIOD:i] for factor in allFactors.values()], 
                        i - OBSERVATION_PERIOD - 1
                    )
                except Exception as e:
                    cur_return, cur_std, cur_loss = predictor.aiPredictor(
                        asset.returns[i - OBSERVATION_PERIOD:i], 
                        [factor.data[i - OBSERVATION_PERIOD:i] for factor in allFactors.values()], 
                        i - OBSERVATION_PERIOD - 1
                    )

            total_loss += cur_loss
            expected_returns.append(cur_return)
            standard_deviations.append(cur_std)
            current_returns.append(asset.returns[i])
            market_caps.append(asset.market_cap[i - 1])

        cov_matrix = utils.calculate_cov(allAssets, OBSERVATION_PERIOD)

        # For Mean-Variance Allocation
        weights_MV = allocator.meanVarianceAllocator(np.array(expected_returns), cov_matrix, riskFreeAssets.returns[i - 1])
        month_return_MV = np.dot(weights_MV, current_returns)
        portfolio_returns_MV.append(month_return_MV)
        month_std_dev_MV = np.sqrt(np.dot(weights_MV.T, np.dot(cov_matrix, weights_MV)))
        portfolio_std_devs_MV.append(month_std_dev_MV)

        # For Genetic Algorithm Allocation
        capm_weights = utils.calculate_market_weights(allAssets, i)
        weights_GA = allocator.aiAllocator(
            np.array(expected_returns), np.array(standard_deviations), cov_matrix, 
            MIN_WEIGHT, MAX_WEIGHT, capm_weights, riskFreeAssets.returns[i - 1], total_loss
        )
        month_return_GA = np.dot(weights_GA, current_returns)
        portfolio_returns_GA.append(month_return_GA)
        month_std_dev_GA = np.sqrt(np.dot(weights_GA.T, np.dot(cov_matrix, weights_GA)))
        portfolio_std_devs_GA.append(month_std_dev_GA)

    return (np.array(portfolio_returns_MV), np.array(portfolio_std_devs_MV), np.array(portfolio_returns_GA), np.array(portfolio_std_devs_GA))



# Initialize allAssets and allFactors with the data from data_parser modules

# data_parser.fetch_and_save_all_data()
allFactors = data_parser.readFactorData()
allAssets = data_parser.readAssetDailyData()
riskFreeAssets = data_parser.readRiskFreeData()
length = len(next(iter(allAssets.values())).returns)
start = int(length * 3 / 5)
returns_CAPM, std_CAPM = marketWeightAllocation(allAssets, allFactors)
returns_History_Genetic, std_History_Genetic = historyPredictWithGeneticAllocation(allAssets, allFactors)
returns_Ai_MV, std_Ai_MV, returns_Ai_Genetic, std_Ai_Genetic= predictAndAllocateAll(allAssets, allFactors)

# Call the plotting functions
plot.plot_all_returns_and_std(returns_Ai_MV, std_Ai_MV, returns_Ai_Genetic, std_Ai_Genetic, returns_CAPM, std_CAPM, returns_History_Genetic, std_History_Genetic)
plot.plot_all_time_series(returns_Ai_MV, std_Ai_MV, returns_Ai_Genetic, std_Ai_Genetic, returns_CAPM, std_CAPM, returns_History_Genetic, std_History_Genetic)
plot.plot_all_geometric_cumulative_returns(returns_Ai_MV, returns_Ai_Genetic, returns_CAPM, returns_History_Genetic)

risk_free_rate = 0.001  # Example risk-free rate

# Assuming returns_Ai_MV, returns_Ai_Genetic, returns_CAPM, returns_History_Genetic are defined
returns = {
    "AI Mean-Variance": returns_Ai_MV,
    "AI Genetic": returns_Ai_Genetic,
    "CAPM": returns_CAPM,
    "Historical Genetic": returns_History_Genetic
}

stds = {
    "AI Mean-Variance": std_Ai_MV,
    "AI Genetic": std_Ai_Genetic,
    "CAPM": std_CAPM,
    "Historical Genetic": std_History_Genetic
}

sharpe_ratios = {
    "AI Mean-Variance": utils.calculate_sharpe_ratio(returns_Ai_MV, risk_free_rate),
    "AI Genetic": utils.calculate_sharpe_ratio(returns_Ai_Genetic, risk_free_rate),
    "CAPM": utils.calculate_sharpe_ratio(returns_CAPM, risk_free_rate),
    "Historical Genetic": utils.calculate_sharpe_ratio(returns_History_Genetic, risk_free_rate)
}

sortino_ratios = {
    "AI Mean-Variance": utils.calculate_sortino_ratio(returns_Ai_MV, risk_free_rate),
    "AI Genetic": utils.calculate_sortino_ratio(returns_Ai_Genetic, risk_free_rate),
    "CAPM": utils.calculate_sortino_ratio(returns_CAPM, risk_free_rate),
    "Historical Genetic": utils.calculate_sortino_ratio(returns_History_Genetic, risk_free_rate)
}

vars_95 = {
    "AI Mean-Variance": utils.calculate_var(returns_Ai_MV),
    "AI Genetic": utils.calculate_var(returns_Ai_Genetic),
    "CAPM": utils.calculate_var(returns_CAPM),
    "Historical Genetic": utils.calculate_var(returns_History_Genetic)
}

print("returns:", returns)
print("stds:", stds)
print("Sharpe Ratios:", sharpe_ratios)
print("Sortino Ratios:", sortino_ratios)
print("Value at Risk (95% Confidence):", vars_95)


# Plot

# cov_matrix = utils.calculate_cov(allAssets, OBSERVATION_PERIOD)
# returns = [asset.returns[-1] for asset in allAssets.values()]
# market_weights = utils.calculate_market_weights(allAssets)
# plot.plot_mean_variance(returns, cov_matrix)
# plot.plot_mean_variance_with_fitness(returns, cov_matrix, market_weights, MIN_WEIGHT, MAX_WEIGHT)

# allFactors = data_parser.readFactorData()
# portfolio_returns = historyPredictWithMeanVariance(allAssets, allFactors)
# portfolio_returns_ai = AIPredictWithGeneticAllocation(allAssets, allFactors)
