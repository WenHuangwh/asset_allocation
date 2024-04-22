import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import factor_tickers as factor_tickers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf


def NN_predicate(pred_return, asset_returns:np.ndarray, factor_data: np.ndarray, month_index):
    """
    Predict asset monthly expected return using Neural Network.

    :param pred_return: predicted return from GARCH.
    :param asset_returns: NumPy array of asset monthly returns.
    :param factor_data: NumPy array of factor data.
    :return: predict return
    """
    factor_data = np.array(factor_data)
    asset_returns = np.array(asset_returns)

    # Prepare data for NN
    df = pd.DataFrame(factor_data.T, columns=factor_tickers.keys())
    df['pred_return'] = [pred_return] * len(df)
    # df['refined_return'] = asset_returns

    # Standardize the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

    # Split data into training and testing sets
    X = df_scaled.drop('pred_return', axis=1)  # Features
    y = df_scaled['pred_return']               # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the NN model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)), # Input layer
        tf.keras.layers.Dense(32, activation='relu'), # Hidden layer
        tf.keras.layers.Dense(1) # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, validation_split=0.1)

    # Evaluate the loss
    test_loss = model.evaluate(X_test, y_test)
    # print('Test Loss:', test_loss)

    # Predict refined return
    refined_return = model.predict(X_test)
    # print(f"Predicted return: {refined_return}")
    return (float(refined_return[-1]), test_loss)