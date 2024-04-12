"""
Contains all data-related classes and functions.
"""
import numpy as np

class Asset:
    def __init__(self, returns, stds, market_cap, volume):
        self.returns = np.array(returns)  # Monthly returns
        self.stds = np.array(stds)        # Monthly standard deviations
        self.market_cap = np.array(market_cap)
        self.volume = np.array(volume)

class Factor:
    def __init__(self, data):
        self.data = np.array(data)        # Monthly factor data