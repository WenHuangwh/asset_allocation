import numpy as np
import pandas as pd
import warnings, os
from typing import Tuple, Dict
from main import Asset, Factor

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_column_names(path: str) -> list:
    """
    Read the first row of the Excel file to get the column names.

    :param path: Path to the Excel file.
    :return: A list of column names.
    """
    df = pd.read_excel(path, nrows=0)
    print(df.columns.tolist())
    return df.columns.tolist()
    

def readAssetDailyData(path: str) -> Dict[str, Asset]:
    """
    Read daily asset data and calculate monthly return and standard deviation.

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
        result[index] = (monthly_return.values, monthly_std.values)

    # Convert the monthly return and standard deviation to NumPy arrays and return them
    return result

def readFactorData(path: str) -> Dict[str: Factor]:
    """
    Read monthly factor data from a CSV file.

    :param path: Path to the CSV file containing monthly factor data.
    :return: A map of index name to Factor
    """
    pass

# TESTS
# dict = readAssetDailyData("../data/USA_S&P 500 Index.xlsx")
# dict = readAssetDailyData("../data/Equity.xlsx")
# for key, value in dict.items():
#     with np.printoptions(threshold=5):
#         print(key)
#         print(value[0])
#         print(value[1])
#         print(value[0].shape)
#         print(value[1].shape)
# monthly_returns, monthly_stds = readAssetDailyData("../data/Equity.xlsx")
# with np.printoptions(threshold=np.inf):
#     print(monthly_returns)
#     print(monthly_stds)
# print(monthly_returns.shape)
# print(monthly_stds.shape)
# column_names = get_column_names("../data/USA_S&P 500 Index.xlsx")
# print(column_names)