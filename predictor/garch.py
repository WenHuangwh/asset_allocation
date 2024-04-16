from arch import arch_model
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

window_size = 36

def find_best_garch_model(returns, max_p=3, max_q=3):
    best_aic, best_bic, best_order = np.inf, np.inf, None
    results = []

    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                model = arch_model(returns, vol='Garch', p=p, q=q)
                model_fit = model.fit(disp='off')
                aic = model_fit.aic
                bic = model_fit.bic
                results.append((p, q, aic, bic))
                
                if aic < best_aic:
                    best_aic, best_bic, best_order = aic, bic, (p, q)
            except:
                continue

    result_table = pd.DataFrame(results, columns=['p', 'q', 'AIC', 'BIC'])
    print(result_table)
    print(f"Best model: GARCH({best_order}) with AIC: {best_aic} and BIC: {best_bic}")

    return best_order


def find_best_arima_model(data, max_p=5, max_q=5, seasonal=False):
    best_aic, best_bic, best_order = np.inf, np.inf, None
    results = []

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue  
            try:
                model = ARIMA(data, order=(p, 0, q))
                model_fit = model.fit()

                aic = model_fit.aic
                bic = model_fit.bic
                results.append((p, q, aic, bic))

                if aic < best_aic:
                    best_aic, best_bic, best_order = aic, bic, (p, q)
            except Exception as e:
                continue

    result_table = pd.DataFrame(results, columns=['p', 'q', 'AIC', 'BIC'])
    print("Results for each (p, q) configuration:")
    print(result_table)

    print(f"Best model: ARIMA{best_order} with AIC: {best_aic} and BIC: {best_bic}")
    return best_order

def GARCH_predicate(asset_returns):
    # Plot PACF
    plot_pacf(asset_returns**2)

    # Fit GARCH model
    best_p, best_q = find_best_garch_model(asset_returns)
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
    pred_var = pred.variance.values[-1, :][0]
    pred_std = np.sqrt(pred_var)
    print("Predicted volatility for next month:", pred_std)

    # predict return with ARMA
    best_p, best_q = find_best_arima_model(asset_returns)
    model = ARIMA(asset_returns, order=(best_p, 0, best_q))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=1)
    pred_return = forecast.predicted_mean
    conf_int = forecast.conf_int()

    print("Predicted return for next month:", pred_return[0])
    print("95% confidence interval:", conf_int)

    return pred_return, pred_std