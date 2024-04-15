from datetime import timedelta
import numpy as np
import pandas as pd
import itertools
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from typing import Tuple, Dict
from models import Asset, Factor
from pomegranate import bayesian_network as BayesianNetwork
import warnings

warnings.filterwarnings('ignore')

window_size = 36

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

    # Plot PACF
    plot_pacf(asset_returns**2)

    # Fit GARCH model
    # best_aic = np.inf
    best_p, best_q = 1, 1
    # for p, q in itertools.product(range(1, 4), repeat=2):
    #     try:
    #         temp_model = arch_model(asset_returns, p=p, q=q)
    #         temp_model_fit = temp_model.fit(disp='off')
    #         if temp_model_fit.aic < best_aic:
    #             best_aic = temp_model_fit.aic
    #             best_p, best_q = p, q
    #     except ValueError:
    #         continue
    
    model = arch_model(asset_returns, p=best_p, q=best_q)
    model_fit = model.fit()
    print(model_fit.summary())

    # Rolling forecast
    rolling_predictions = []
    if len(asset_returns) >= window_size:
        for i in range(len(asset_returns) - window_size + 1):
            train = asset_returns[i:i + window_size]
            model = arch_model(train, p=best_p, q=best_q)
            model_fit = model.fit(disp='off')
            pred = model_fit.forecast(horizon=1)
            rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))
    
    # Plot rolling predictions
    plt.figure(figsize=(10, 6))
    plt.plot(asset_returns, label='True Returns')
    plt.plot(rolling_predictions, label='Predicted Volatility')
    plt.plot(asset_std, label='True Volatility')
    plt.title('Rolling Volatility Prediction')
    plt.legend(['True Returns', 'Predicted Volatility', 'True Volatility'])

    pred = model_fit.forecast(horizon=7)
    pred = pd.Series(np.sqrt(pred.variance.values[-1, :]))
    plt.figure(figsize=(10, 6))
    plt.plot(pred)
    plt.title('7-Day Volatility Forecast')

    plt.show()

    pred = model_fit.forecast(horizon=1)
    pred_std = np.sqrt(pred.variance.values[-1, :][0])

    # predict return
    # error: normal distribution N(0, 1)
    curr_return = np.random.normal(0, 1) * pred_std

    # use bayesian network to predict return
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

    
    rolling_returns = np.convolve(asset_returns, np.ones(window_size)/window_size, 'valid')
    rolling_std_devs = np.convolve(asset_std, np.ones(window_size)/window_size, 'valid')

    return (rolling_returns[-1], rolling_std_devs[-1])