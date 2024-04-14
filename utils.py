import numpy as np
import pandas as pd
from datetime import datetime


def calculate_market_weights(all_assets):
    """ Calculate market weights based on the market cap. """
    total_market_cap = sum(asset.market_cap[-1] for asset in all_assets.values())
    return np.array([asset.market_cap[-1] / total_market_cap for asset in all_assets.values()])


def calculate_averages_and_cov(all_assets):
    """ Calculate average returns, standard deviations, and covariance matrix for the past 5 years. """
    now = datetime.now()
    past_date = now.replace(year=now.year - 5)
    
    avg_returns = []
    avg_stds = []
    values = []
    
    for name, asset in all_assets.items():
        # Filter data to the last five years
        recent_data = asset.returns[(asset.returns.index > past_date)]
        avg_return = np.mean(recent_data)
        avg_std = np.mean(asset.stds[(asset.stds.index > past_date)])
        
        avg_returns.append(avg_return)
        avg_stds.append(avg_std)
        values.append(recent_data)

    # Calculate covariance matrix from recent data values
    cov_matrix = np.cov(values)
    
    return np.array(avg_returns), np.array(avg_stds), cov_matrix

def fitness(weights, expected_returns, cov_matrix, capm_weights):
    """ Calculate the fitness of a portfolio configuration. """
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_risk
    capm_deviation = np.sum((weights - capm_weights) ** 2)
    
    # Enhance penalty non-linearly
    penalty = np.sum(capm_deviation ** 2)
    
    return sharpe_ratio - penalty,
