from datetime import timedelta
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from models import Asset, Factor
from pomegranate import bayesian_network as BayesianNetwork
from . import garch


def aiPredictor(asset_returns: np.ndarray, asset_std: np.ndarray, factor_data: np.ndarray) -> Tuple[int, int]:
    """
    Predict asset monthly expected return, standard deviation, and covariance using GARCH and Bayesian Network.

    :param asset_returns: NumPy array of asset monthly returns.
    :param asset_std: NumPy array of asset monthly standard deviations.
    :param factor_data: NumPy array of factor data.
    :return: predict return, predict std
    """
    # Plot asset returns and standard deviation
    plt.figure(figsize=(10, 6))
    plt.plot(asset_std, label='Asset Standard Deviation')
    plt.plot(asset_returns, label='Asset Returns')
    plt.legend(['Asset Standard Deviation', 'Asset Returns'])
    plt.title('Asset Returns and Standard Deviation')

    # Get standard deviation and return prediction using GARCH
    pred_std, pred_return = garch.GARCH_predicate(asset_returns)

    # TODO: use bayesian network to predict return
    structure = ((), (), (), (0, 1, 2))
    model = BayesianNetwork._from_structure(factor_data, structure)
    pred_return = model.predict_proba({})[2]


    return (pred_return, pred_std)


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