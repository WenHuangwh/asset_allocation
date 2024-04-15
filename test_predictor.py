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

# test aiPredictor
cur_return, cur_std = predictor.aiPredictor(allAssets['NASDAQ100'].returns, allAssets['NASDAQ100'].stds, None)

# test on daily data
# start = "2015-01-01"
# end = "2020-01-01"

# dis = yf.Ticker('QQQ').history(start=start, end=end)

# returns = 100 * dis.Close.pct_change().dropna()
# std_dev = 100 * dis.Close.pct_change().rolling(21).std().dropna()
# cur_return, cur_std = predictor.aiPredictor(returns, std_dev, None)

# print(cur_return, cur_std)