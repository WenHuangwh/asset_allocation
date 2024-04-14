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

def fetch_and_save_all_data(start_date="1990-01-01"):
    """
    Fetch historical data for all defined indexes and their corresponding ETFs from Yahoo Finance, 
    then save the data to CSV files.

    :param start_date: The starting date from which to fetch historical data.
    """
    for name, (index_ticker, etf_ticker) in tickers.items():
        print(f"Fetching and saving data for {name} using index {index_ticker} and ETF {etf_ticker}...")
        fetch_data(index_ticker, etf_ticker, start_date)

def save_data(df, filename):
    """ Save DataFrame to a CSV file. """
    df.to_csv(filename)

def load_data(filename):
    """ Load DataFrame from a CSV file. """
    if os.path.exists(filename):
        return pd.read_csv(filename, index_col=0, parse_dates=True)
    return None

def fetch_data(index_ticker, etf_ticker, start_date="1990-01-01"):
    filename = f"data/data_{index_ticker}.csv"
    index = yf.Ticker(index_ticker)
    index_data = index.history(start=start_date)
    
    index_data['Volatility'] = index_data['Close'].rolling(window=21).std() * np.sqrt(21)
    index_data['Monthly Return'] = index_data['Close'].resample('M').last().pct_change()
    index_data['Monthly Std Dev'] = index_data['Close'].pct_change().resample('M').std()
    index_data = index_data.resample('M').last()

    etf = yf.Ticker(etf_ticker)
    etf_data = etf.history(start=start_date).resample('M').last()

    # Get static totalAssets as fallback
    total_assets = etf.info.get('totalAssets', 0)
    # Calculate dynamic market cap estimate by scaling totalAssets by price change ratio
    initial_price = etf_data['Close'].iloc[0]
    etf_data['MarketCap'] = etf_data['Close'].apply(lambda x: total_assets * (x / initial_price))
    
    combined_data = index_data[['Close', 'Volume', 'Volatility', 'Monthly Return', 'Monthly Std Dev']]
    combined_data['MarketCap'] = etf_data['MarketCap']
    
    combined_data.to_csv(filename)
    print(f"Data for {index_ticker} saved to {filename}")



def readAssetDailyData():
    """ Create Asset instances from data stored in CSV files. """
    all_data = {}
    for name, (index_ticker, etf_ticker) in tickers.items():
        filename = f"data/data_{index_ticker}.csv"
        print(f"Loading data for {name} from {filename}...")
        if os.path.exists(filename):
            data = pd.read_csv(filename, index_col=0, parse_dates=True)
            data.fillna(0, inplace=True)
            asset_data = Asset(
                data['Monthly Return'], 
                data['Monthly Std Dev'], 
                data['MarketCap'], 
                data['Volume']
            )
            all_data[name] = asset_data
            print(f"Data for {name} loaded successfully.")
        else:
            print(f"No data file found for {name}. Please run fetch_data first.")
    return all_data

def readFactorData(path: str) -> Dict[str, Factor]:
    """
    Read monthly factor data from a CSV file.

    :param path: Path to the CSV file containing monthly factor data.
    :return: A map of index name to Factor
    """
    pass