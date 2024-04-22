from data_parser import data_parser
from allocator import allocator
from predictor import predictor
from models import Asset, Factor
import utils as utils
import numpy as np
import matplotlib.pyplot as plt

# Constants
OBSERVATION_PERIOD = 5 * 12  # 60 months
PREDICTION_PERIOD = 5 * 12   # 60 months

def predictAndPlotSingleAsset(asset, allFactors, start):
    predicted_returns = []
    actual_returns = []
    predicted_stds = []
    actual_stds = []
    
    for i in range(start, start + PREDICTION_PERIOD):
        try:
            cur_return, cur_std, _ = predictor.aiPredictor(
                asset.returns[i - OBSERVATION_PERIOD:i], 
                asset.stds[i - OBSERVATION_PERIOD:i], 
                [factor.data[i - OBSERVATION_PERIOD:i] for factor in allFactors.values()],
                i - OBSERVATION_PERIOD - 1
            )
        except Exception as e:
            print(f"Prediction failed: {e}")
            continue
        
        predicted_returns.append(cur_return)
        predicted_stds.append(cur_std)
        actual_returns.append(asset.returns[i])
        actual_stds.append(asset.stds[i])

    print("Average of actual std", np.average(actual_stds))
    print("Average of predict std", np.average(predicted_stds))
    
    time = np.arange(len(predicted_returns))
    
    # Plot for Returns
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.scatter(time, predicted_returns, color='g', label='Predicted Returns', marker='o')
    ax1.set_xlabel('Time (Months)')
    ax1.set_ylabel('Predicted Returns', color='g')
    ax1.tick_params(axis='y', labelcolor='g')

    ax2 = ax1.twinx()
    ax2.scatter(time, actual_returns, color='b', label='Actual Returns', marker='x')
    ax2.set_ylabel('Actual Returns', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    fig.suptitle('Comparison of Predicted and Actual Returns')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()

    # Plot for Standard Deviations
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(time, predicted_stds, color='r', label='Predicted Std Dev', marker='o')
    ax.scatter(time, actual_stds, color='y', label='Actual Std Dev', marker='x')
    ax.set_xlabel('Time (Months)')
    ax.set_ylabel('Standard Deviations')
    ax.legend(loc='upper left')
    fig.suptitle('Comparison of Predicted and Actual Standard Deviations')
    plt.show()

# Initialize data and usage
allFactors = data_parser.readFactorData()
allAssets = data_parser.readAssetDailyData()
length = len(next(iter(allAssets.values())).returns)
start = int(length * 3 / 5)

asset = allAssets["InvestmentGradeCorpBond"]
predictAndPlotSingleAsset(asset, allFactors, start)
