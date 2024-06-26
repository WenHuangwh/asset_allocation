from datetime import timedelta
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from models import Asset, Factor
from . import NN, garch


def aiPredictor(asset_returns: np.ndarray, asset_std: np.ndarray, factor_data: np.ndarray, month_index = 0) -> Tuple[int, int]:
    """
    Predict asset monthly expected return, standard deviation, and covariance using GARCH and Neural Network.

    :param asset_returns: NumPy array of asset monthly returns.
    :param asset_std: NumPy array of asset monthly standard deviations.
    :param factor_data: NumPy array of factor data.
    :param month_index: Index of the last month.
    :return: predict return, predict std
    """
    # Plot asset returns and standard deviation
    # plt.figure(figsize=(10, 6))
    # plt.plot(asset_std, label='Asset Standard Deviation')
    # plt.plot(asset_returns, label='Asset Returns')
    # plt.legend(['Asset Standard Deviation', 'Asset Returns'])
    # plt.title('Asset Returns and Standard Deviation')

    # Get standard deviation and return prediction using GARCH
    pred_return, pred_std = garch.garch_forecast(asset_returns)

    # Get refined return prediction using Neural Network
    pred_return, loss = NN.NN_predicate(pred_std, asset_returns, factor_data, month_index)

    scale_factor = 4

    return (pred_return, pred_std / scale_factor, loss)


def historyPredictor(asset_returns: np.ndarray, asset_std: np.ndarray) -> Tuple[int, int]:
    """
    Predict asset monthly expected return, standard deviation.

    :param asset_returns: NumPy array of asset monthly returns.
    :param asset_std: NumPy array of asset monthly standard deviations.
    :param factor_data: NumPy array of factor data.
    :return: predict return, predict std
    """

    
    rolling_returns = np.convolve(asset_returns, np.ones(garch.window_size)/garch.window_size, 'valid')
    rolling_std_devs = np.convolve(asset_std, np.ones(garch.window_size)/garch.window_size, 'valid')

    return (rolling_returns[-1], rolling_std_devs[-1])