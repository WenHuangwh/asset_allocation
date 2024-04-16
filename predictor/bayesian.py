import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pomegranate.bayesian_network import BayesianNetwork
from pomegranate.distributions import *



def bayesian_predicate(pred_return, factor_data: np.ndarray, month_index):
    """
    Predict asset monthly expected return using Bayesian Network.

    :param pred_return: predicted return from GARCH.
    :param factor_data: NumPy array of factor data.
    :return: predict return
    """
    # Get factor data for the current month
    curr_month_factor = []
    for factor in factor_data:
        curr_month_factor.append(factor[month_index])
    
    # TODO: use bayesian network to predict return
    network = BayesianNetwork("Asset Return Prediction")
    
    
    return pred_return