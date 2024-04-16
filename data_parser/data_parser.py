import os
import pandas as pd
import numpy as np
import yfinance as yf
from models import Asset, Factor
from typing import Dict
from fredapi import Fred
from utils import factor_tickers as factor_tickers

start_date = "2009-01-01"
fred_api_key = os.getenv('FRED_API_KEY')
fred = Fred(api_key=fred_api_key)

# tickers = {
#     "SP500": ("SPY", "SPY"),
#     "NASDAQ100": ("QQQ", "QQQ"),
#     "DowJones": ("DIA", "DIA"),
#     "Russell1000Growth": ("IWF", "IWF"),
#     "Russell1000Value": ("IWD", "IWD"),
#     "Russell2000": ("IWM", "IWM"),
#     "1MonthTreasury": ("BIL", "BIL"),
#     "2YearTreasury": ("SHY", "SHY"),
#     "10YearTreasury": ("IEF", "IEF"),
#     "InvestmentGradeCorpBond": ("LQD", "LQD"),
#     "HighYieldCorpBond": ("HYG", "HYG"),
#     "REITs": ("VNQ", "VNQ"),
#     "Gold": ("GLD", "GLD"),
#     "Oil": ("USO", "USO")
# }

tickers = {
    "SP500": ("SPY", "SPY"),
    "NASDAQ100": ("QQQ", "QQQ"),
    "DowJones": ("DIA", "DIA"),
    "Russell1000Growth": ("IWF", "IWF"),
    "Russell1000Value": ("IWD", "IWD"),
    "Russell2000": ("IWM", "IWM"),
    "10YearTreasury": ("IEF", "IEF"),
    "InvestmentGradeCorpBond": ("LQD", "LQD"),
    "HighYieldCorpBond": ("HYG", "HYG"),
    "REITs": ("VNQ", "VNQ"),
    "Gold": ("GLD", "GLD"),
    "Oil": ("USO", "USO")
}

risk_free_ticker = {
    "1MonthTreasury": ("BIL", "BIL"),
}

# tickers = {
#     "SP500": ("SPY", "SPY"),
#     "NASDAQ100": ("QQQ", "QQQ"),
#     "DowJones": ("DIA", "DIA"),
#     "Russell2000": ("IWM", "IWM"),
# }


def fetch_and_save_all_data(start_date = start_date):
    # Fetch and save data for all index tickers
    for name, (index_ticker, etf_ticker) in tickers.items():
        print(f"Fetching and saving data for {name} using index {index_ticker} and ETF {etf_ticker}...")
        fetch_data(index_ticker, etf_ticker, start_date)

    # Fetch and save data for all factor tickers
    for name, ticker in factor_tickers.items():
        print(f"Fetching and saving factor data for {name} using ticker {ticker}...")
        fetch_factor_data(ticker, start_date)

def fetch_factor_data(factor_ticker, start_date):
    filename = f"data/{factor_ticker}.csv"

    # Fetching factor data from FRED
    factor = fred.get_series(factor_ticker, observation_start=start_date)

    # Resampling to monthly data
    factor_data_monthly = factor.resample('ME').last()
    factor_data_monthly.dropna(inplace=True)

    # Setting column and saving the combined data to CSV
    factor_data_monthly = pd.DataFrame(factor_data_monthly, columns=[factor_ticker])
    factor_data_monthly.to_csv(filename, header=True, index_label='Date')
    print(f"Data for {factor_ticker} saved to {filename}")

def fetch_data(index_ticker, etf_ticker, start_date):
    filename = f"data/{index_ticker}.csv"
    index = yf.Ticker(index_ticker)
    daily_data = index.history(start=start_date, interval='1d')
    
    # Ensure no missing data in daily data
    daily_data.dropna(inplace=True)

    # Calculate daily returns
    daily_data['Daily Return'] = daily_data['Close'].pct_change()
    daily_data['MarketCap'] = daily_data['Close'] * (index.info.get('totalAssets', 0) / daily_data['Close'].iloc[0])
    
    # Resample to monthly and calculate needed metrics
    index_data = daily_data.resample('ME').agg({
        'Close': 'last',
        'Volume': 'sum',
        'Daily Return': 'std',  # Monthly standard deviation of daily returns
        'MarketCap': 'last'
    }).rename(columns={'Daily Return': 'Monthly Std Dev'})

    # Calculate monthly return from the resampled monthly close prices
    index_data['Monthly Return'] = index_data['Close'].pct_change()

    # Saving the combined data to CSV
    index_data[['Close', 'Volume', 'MarketCap', 'Monthly Return', 'Monthly Std Dev']].to_csv(filename)
    print(f"Data for {index_ticker} saved to {filename}")

def readAssetDailyData():
    all_data = {}
    for name, (index_ticker, etf_ticker) in tickers.items():
        filename = f"data/{index_ticker}.csv"
        if os.path.exists(filename):
            data = pd.read_csv(filename, index_col='Date', parse_dates=True)
            data.fillna(0, inplace=True)
            asset_data = Asset(
                data.get('Monthly Return', 0), 
                data.get('Monthly Std Dev', 0), 
                data.get('MarketCap', 0), 
                data.get('Volume', 0)
            )
            all_data[name] = asset_data
            print(f"Data for {name} loaded successfully.")
        else:
            print(f"No data file found for {name}. Please run fetch_data first.")
    return all_data

def readRiskFreeData():
    all_data = {}
    for name, (index_ticker, etf_ticker) in risk_free_ticker.items():
        filename = f"data/{index_ticker}.csv"
        if os.path.exists(filename):
            data = pd.read_csv(filename, index_col='Date', parse_dates=True)
            data.fillna(0, inplace=True)
            asset_data = Asset(
                data.get('Monthly Return', 0), 
                data.get('Monthly Std Dev', 0), 
                data.get('MarketCap', 0), 
                data.get('Volume', 0)
            )
            all_data[name] = asset_data
            print(f"Data for {name} loaded successfully.")
            return asset_data
        else:
            print(f"No data file found for {name}. Please run fetch_data first.")
    
    return all_data

def readFactorData() -> Dict[str, Factor]:
    all_factors ={}

    for name, ticker in factor_tickers.items():
        filename = f"data/{ticker}.csv"
        if os.path.exists(filename):
            data = pd.read_csv(filename, index_col='Date', parse_dates=True)
            data.fillna(0, inplace=True)

            # Month over month/Year over year percentage change
            if name == "UnempRate" or "FedFunds" or "Yield10Yr":
                factor_data = Factor(data.get(ticker, 0).pct_change() * 100) # MoM
            else:
                factor_data = Factor(data.get(ticker, 0).pct_change(periods=12) * 100) # YoY
            all_factors[name] = factor_data

            print(f"Data for {name} loaded successfully.")
        else:
            print(f"No data file found for {name}. Please run fetch_data first.")

    return all_factors
