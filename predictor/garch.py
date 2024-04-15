from arch import arch_model
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

window_size = 36

def GARCH_predicate(asset_returns):
    # Plot PACF
    plot_pacf(asset_returns**2)

    # Find best p, q
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


    # Fit GARCH model
    model = arch_model(asset_returns, rescale=False, p=best_p, q=best_q)
    model_fit = model.fit()
    print(model_fit.summary())

    # Rolling forecast
    rolling_predictions = []
    if len(asset_returns) >= window_size:
        for i in range(len(asset_returns) - window_size + 1):
            train = asset_returns[i:i + window_size]
            model = arch_model(train, rescale=False, p=best_p, q=best_q)
            model_fit = model.fit(disp='off')
            pred = model_fit.forecast(horizon=1)
            rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0]))
    
    # Plot rolling predictions
    if hasattr(asset_returns, 'index'):
        rolling_predictions = pd.Series(rolling_predictions, index = asset_returns.index[window_size-1:])
    plt.figure(figsize=(10, 6))
    plt.plot(asset_returns, label='True Returns')
    plt.plot(rolling_predictions, label='Predicted Volatility')
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
    pred_return = np.random.normal(0, 1) * pred_std
    return pred_std, pred_return