import numpy as np
import pandas as pd
from typing import Tuple, Dict
from models import Asset, Factor

def aiPredictor(asset_returns: np.ndarray, asset_std: np.ndarray, factor_data: np.ndarray) -> Tuple[int, int]:
    """
    Predict asset monthly expected return, standard deviation, and covariance using GARCH and Bayesian Network.

    :param asset_returns: NumPy array of asset monthly returns.
    :param asset_std: NumPy array of asset monthly standard deviations.
    :param factor_data: NumPy array of factor data.
    :return: predict return, predict std
    """
    pass

def historyPredictor(asset_returns: np.ndarray, asset_std: np.ndarray) -> Tuple[int, int]:
    """
    Predict asset monthly expected return, standard deviation.

    :param asset_returns: NumPy array of asset monthly returns.
    :param asset_std: NumPy array of asset monthly standard deviations.
    :param factor_data: NumPy array of factor data.
    :return: predict return, predict std
    """

    window_size = 36
    rolling_returns = np.convolve(asset_returns, np.ones(window_size)/window_size, 'valid')
    rolling_std_devs = np.convolve(asset_std, np.ones(window_size)/window_size, 'valid')

    return (rolling_returns[-1], rolling_std_devs[-1])