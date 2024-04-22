import numpy as np
import pandas as pd
from datetime import datetime
import scipy.stats as stats

factor_tickers = {
    "CPI": "CPIAUCSL", # Consumer Price Index, YoY
    "UnempRate": "UNRATE", # Unemployment Rate, MoM
    "FedFunds": "FEDFUNDS", # Federal Funds Rate, MoM
    "Yield10Yr": "DGS10", # 10-Year Treasury Constant Maturity Rate, MoM
    "HousingStarts": "HOUST", # Housing Starts, YoY
    "ConsSentiment": "UMCSENT", # Consumer Sentiment, YoY
    "ManufIndex": "INDPRO", # Industrial Production Index, YoY
    "PersConsExp": "PCE", # Personal Consumption Expenditures, YoY
    "Gold": "GLD",
    "Oil": "USO",
}


def calculate_market_weights(all_assets, i):
    """ Calculate market weights based on the market cap. """
    total_market_cap = sum(asset.market_cap[i] for asset in all_assets.values())
    return np.array([asset.market_cap[i] / total_market_cap for asset in all_assets.values()])

def calculate_cov(all_assets, observation_period):
    """ Calculate average returns, standard deviations, and covariance matrix for the past 5 years. """
    values = []
    
    for name, asset in all_assets.items():
        # Filter data to the last five years
        recent_data = asset.returns[-observation_period:]
        values.append(recent_data)

    # Calculate covariance matrix from recent data values
    cov_matrix = np.cov(values)
    
    return cov_matrix

def calculate_sharpe_ratio(returns, risk, risk_free_rate=0.0):
    """ Calculate the Sharpe ratio for given returns and risk. """
    return (returns - risk_free_rate) / risk if risk != 0 else np.inf

def calculate_penalty(weights, target_weights):
    """ Calculate a penalty for deviation from target weights. """
    deviation = np.sum((weights - target_weights) ** 2)
    return np.sum(deviation ** 2)  # Non-linear penalty

def fitness(weights, expected_returns, cov_matrix, capm_weights):
    """ Calculate the fitness of a portfolio configuration. """
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    sharpe_ratio = calculate_sharpe_ratio(portfolio_return, portfolio_risk)
    penalty = calculate_penalty(weights, capm_weights)

    return sharpe_ratio - penalty,


def calculate_sharpe_ratio(returns, risk_free_rate):
    mean_return = np.mean(returns)
    return_std = np.std(returns)
    return (mean_return - risk_free_rate) / return_std

def calculate_sortino_ratio(returns, risk_free_rate):
    mean_return = np.mean(returns)
    negative_returns = [min(0, x - risk_free_rate) for x in returns]
    downside_std = np.std(negative_returns)
    return (mean_return - risk_free_rate) / downside_std

def calculate_var(returns, confidence_level=0.95):
    if len(returns) == 0:
        return None
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    return sorted_returns[index]