import numpy as np
import pandas as pd
from datetime import datetime

factor_tickers = {
    "CPI": "CPIAUCSL", # Consumer Price Index, YoY
    "UnempRate": "UNRATE", # Unemployment Rate, MoM
    "FedFunds": "FEDFUNDS", # Federal Funds Rate, MoM
    "Yield10Yr": "DGS10", # 10-Year Treasury Constant Maturity Rate, MoM
    "HousingStarts": "HOUST", # Housing Starts, YoY
    "ConsSentiment": "UMCSENT", # Consumer Sentiment, YoY
    "ManufIndex": "INDPRO", # Industrial Production Index, YoY
    "PersConsExp": "PCE" # Personal Consumption Expenditures, YoY
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
