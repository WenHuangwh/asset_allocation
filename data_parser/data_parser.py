import os
import pandas as pd
import numpy as np
import yfinance as yf
from models import Asset, Factor
from typing import Dict

tickers = {
    "SP500": ("SPY", "SPY"),
    "NASDAQ100": ("QQQ", "QQQ"),
    "DowJones": ("DIA", "DIA"),
    "Disney": ("DIS", "DIS"),
    "Russell1000Growth": ("IWF", "IWF"),
    "Russell1000Value": ("IWD", "IWD"),
    "Russell2000": ("IWM", "IWM"),
    "1MonthTreasury": ("BIL", "BIL"),
    "2YearTreasury": ("SHY", "SHY"),
    "10YearTreasury": ("IEF", "IEF"),
    "InvestmentGradeCorpBond": ("LQD", "LQD"),
    "HighYieldCorpBond": ("HYG", "HYG"),
    "REITs": ("VNQ", "VNQ"),
    "Gold": ("GLD", "GLD"),
    "Oil": ("USO", "USO")
}

def fetch_and_save_all_data(start_date="2009-01-01"):
    for name, (index_ticker, etf_ticker) in tickers.items():
        print(f"Fetching and saving data for {name} using index {index_ticker} and ETF {etf_ticker}...")
        fetch_data(index_ticker, etf_ticker, start_date)

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

def readFactorData(path: str) -> Dict[str, Factor]:
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True).to_dict()
    return {}
