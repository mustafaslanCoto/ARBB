import pandas as pd
import numpy as np
##Stationarity Check
from statsmodels.tsa.stattools import adfuller, kpss
def unit_root_test(series, method = "ADF"):
    if method == "ADF":
        adf = adfuller(series, autolag = 'AIC')[1]
        if adf < 0.05:
            return adf, print('ADF p-value: %f' % adf + " and data is stationary at 5% significance level")
        else:
            return adf, print('ADF p-value: %f' % adf + " and data is non-stationary at 5% significance level")
    elif method == "KPSS":
        kps = kpss(series)[1]
        if kps < 0.05:
            return kps, print('KPSS p-value: %f' % kps + " and data is non-stationary at 5% significance level")
        else:
            return kps, print('KPSS p-value: %f' % kps + " and data is stationary at 5% significance level")
    else:
        return print('Enter a valid unit root test method')

## Serial Corelation Check
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def plot_PACF_ACF(series, lag_num, figsize = (15, 8)):
    fig, ax = pyplot.subplots(2,1, figsize=figsize)
    plot_pacf(series, lags= lag_num, ax = ax[0])
    plot_acf(series, lags= lag_num, ax = ax[1])
    ax[0].grid(which='both')
    ax[1].grid(which='both')
    pyplot.show()

def fourier_terms(start, stop, period, num_terms, df_index):
    '''
    Returns fourier terms for the given seasonal period and dataframe.

            Parameters:
                    start (int): An integer that should be 0 for the training dataset, whereas for the test dataset, 
                    it should correspond to the length of the training data.
                    stop (int): An integer representing the length of the training dataset is required for the training dataset,
                    while for the testing dataset, it should be the sum of the lengths of both the training and test datasets.
                    period (int): the seosanal period.
                    num_terms (int): It specifies how many pairs of sin and cos terms to include.
                    df_index: a dataframe (training or test dataset). It specify whether to use the indexes of the training or 
                    test dataset for the returned dataframe
    '''
    t = np.arange(start, stop)
    df = pd.DataFrame(index=df_index.index)
    for i in range(1, num_terms + 1):
        df["sin_"+str(i-1)+"_"+str(period)] = np.sin(2 * np.pi * i * t / period)
        df["cos_"+str(i-1)+"_"+str(period)] = np.cos(2 * np.pi * i * t / period)
    return df

def rmse(y_true, y_pred):
    """
    Calculate Root Mean Square Error (RMSE).

    Parameters:
    - y_true: Array of true values.
    - y_pred: Array of predicted values.

    Returns:
    - rmse: Root Mean Square Error.
    """
    # Ensure both arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("Input arrays must have the same length.")

    # Calculate squared differences
    squared_diff = (y_true - y_pred) ** 2

    # Calculate mean squared error
    mean_squared_error = np.mean(squared_diff)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error)

    return rmse

def smape(y_true, y_pred):
    return 1/len(y_true) * np.sum(2 * np.abs(y_pred-y_true) / (np.abs(y_true) + np.abs(y_pred))*100)