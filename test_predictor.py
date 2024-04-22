"""
main for the predictor tests
"""

from data_parser import data_parser
from predictor import predictor
from models import Asset, Factor
import utils as utils
import numpy as np
import plot as plot
import yfinance as yf

# data_parser.fetch_and_save_all_data()
allAssets = data_parser.readAssetDailyData()
allFactors = data_parser.readFactorData()

# test aiPredictor
# cur_return, cur_std = predictor.aiPredictor(allAssets['NASDAQ100'].returns, allAssets['NASDAQ100'].stds, allFactors, 100)
asset = allAssets['NASDAQ100']
OBSERVATION_PERIOD = 60
length = len(next(iter(allAssets.values())).returns)
start = int(length * 3 / 4)

for i in range(start, start + 36):
# for asset in allAssets.values():
    cur_return, cur_std = predictor.aiPredictor(asset.returns[i - OBSERVATION_PERIOD:i], asset.stds[i - OBSERVATION_PERIOD:i], [factor.data[i - OBSERVATION_PERIOD:i] for factor in allFactors.values()], i - OBSERVATION_PERIOD - 1)


# test on daily data
# start = "2015-01-01"
# end = "2020-01-01"

# dis = yf.Ticker('QQQ').history(start=start, end=end)

# returns = 100 * dis.Close.pct_change().dropna()
# std_dev = 100 * dis.Close.pct_change().rolling(21).std().dropna()
# cur_return, cur_std = predictor.aiPredictor(returns, std_dev, None)

print(cur_return, cur_std)