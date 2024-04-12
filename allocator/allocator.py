import numpy as np
import pandas as pd
from typing import Tuple

def geneticAllocator(expected_returns: np.ndarray, std_deviations: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """
    Determine optimal asset weights using a genetic algorithm.

    :param expected_returns: NumPy array of predicted expected returns.
    :param std_deviations: NumPy array of predicted standard deviations.
    :param cov_matrix: NumPy array of predicted covariance matrix.
    :return: NumPy array of optimal weights.
    """
    pass

def meanVarianceAllocator(expected_returns: np.ndarray, std_deviations: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """
    Determine optimal asset weights using the algebraic mean-variance method.

    :param expected_returns: NumPy array of predicted expected returns.
    :param std_deviations: NumPy array of predicted standard deviations.
    :param cov_matrix: NumPy array of predicted covariance matrix.
    :return: NumPy array of optimal weights.
    """
    pass