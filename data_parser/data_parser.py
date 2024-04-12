import numpy as np
import pandas as pd
import yfinance as yf
import warnings, os
from typing import Tuple, Dict
from models import Asset, Factor

warnings.simplefilter(action='ignore', category=FutureWarning)

tickers = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DowJones": "^DJI",
    "Russell1000Growth": "IWF",
    "Russell1000Value": "IWD",
    "Russell2000": "IWM",
    "1MonthTreasury": "BIL",  # Approximation for 1-Month Treasury Bill
    "2YearTreasury": "SHY",   # Approximation for 2-Year Treasury Note
    "10YearTreasury": "IEF",  # Approximation for 10-Year Treasury Note
    "InvestmentGradeCorpBond": "LQD",
    "HighYieldCorpBond": "HYG",
    "REITs": "VNQ",
    "Gold": "GLD",
    "Oil": "USO"
}

def fetch_data(ticker: str, start_date: str = "1990-01-01") -> pd.DataFrame:
    """
    Fetch historical data for a given ticker from Yahoo Finance.
    
    :param ticker: Ticker symbol of the asset.
    :param start_date: Start date for historical data fetch.
    :return: DataFrame with historical data including Close Price and Volume.
    """
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date)

    data['Volatility'] = data['Close'].rolling(window=252).std() * (252 ** 0.5)  # Annualized volatility
    data['MarketCap'] = stock.info.get('marketCap', 0)

    # Resample to get end of month closing prices and calculate monthly returns
    monthly_prices = data['Close'].resample('M').last()
    data['Monthly Return'] = monthly_prices.pct_change()

    # Resample to calculate the monthly standard deviation of daily returns
    daily_returns = data['Close'].pct_change()
    data['Monthly Std Dev'] = daily_returns.resample('M').std()

    # Keep only the last entry of each month
    data = data.resample('M').last()

    return data[['Close', 'Volume', 'Volatility', 'MarketCap', 'Monthly Return', 'Monthly Std Dev']]

def readAssetDailyData() -> Dict[str, Asset]:
    """
    Read daily asset data through Yahoo Finance api

    :return: A map of index name to Asset object.
    """
    all_data = {}
    for name, ticker in tickers.items():
        print(f"Fetching data for {name} ({ticker})...")
        data = fetch_data(ticker)
        Asset_data = Asset(data['Monthly Return'], data['Monthly Std Dev'], data['MarketCap'], data['Volume'])
        all_data[name] = Asset_data
    return all_data

def get_column_names(path: str) -> list:
    """
    Read the first row of the Excel file to get the column names.

    :param path: Path to the Excel file.
    :return: A list of column names.
    """
    df = pd.read_excel(path, nrows=0)
    print(df.columns.tolist())
    return df.columns.tolist()

def readAssetDailyData_Excel(path: str) -> Dict[str, Asset]:
    """
    Read and parse daily asset data from excel and calculate monthly return and standard deviation.
    The format of the Excel file should be as follows:
    - The first column should be 'Name' containing the date in 'YYYY-MM-DD' format.
    - The subsequent columns should contain the daily asset data.

    :param path: Path to the CSV file containing daily asset data.
    :return: A map of index name to Asset.
    """
    # read excel file
    df = pd.read_excel(path)

    # Get the column names
    column_names = get_column_names(path)

    # Check if the 'Name' column is present and convert it to datetime if not already
    if 'Name' not in column_names:
        raise ValueError("No 'Date' column found in the Excel file")
    
    df['Name'] = pd.to_datetime(df['Name'])

    df.columns = column_names

    df.set_index(column_names[0], inplace=True)

    result = {}

    # Forward fill the missing values for all Indices
    for index in column_names[1:]:
        df[index].fillna(method='ffill', inplace=True)

        # Calculate daily returns
        index_name = index + ' Daily Return'
        df[index_name] = df[index].pct_change()

        # Remove the first row with NaN return
        df.dropna(subset=[index_name], inplace=True)

        # Group by year and month to calculate the monthly return
        monthly_return = df[index_name].resample('ME').apply(lambda x: (1 + x).prod() - 1)

        # Group by year and month to calculate the monthly standard deviation
        monthly_std = df[index_name].resample('ME').std()

        # Store the monthly return and standard deviation in the result dictionary
        
        result[index] = Asset(monthly_return, monthly_std, None, None)

    # Convert the monthly return and standard deviation to NumPy arrays and return them
    return result

def readFactorData(path: str) -> Dict[str, Factor]:
    """
    Read monthly factor data from a CSV file.

    :param path: Path to the CSV file containing monthly factor data.
    :return: A map of index name to Factor
    """
    pass