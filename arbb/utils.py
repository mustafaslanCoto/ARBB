import pandas as pd
import numpy as np
from scipy import stats
from numba import jit
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.linear_model import LinearRegression
from numba import jit
##Stationarity Check
from statsmodels.tsa.stattools import adfuller, kpss
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval

#------------------------------------------------------------------------------
# Unit Root Test and Serial Correlation Check
#------------------------------------------------------------------------------

def unit_root_test(series, method = "ADF", n_lag = None):
    """
    Performs a unit root test on the given time series data to check for stationarity.
    Args:
        series (pd.Series): The time series data to be tested.
        method (str): The method for the unit root test, either "ADF" for Augmented Dickey-Fuller or "KPSS" for Kwiatkowski-Phillips-Schmidt-Shin.
        n_lag (int, optional): The number of lags to include in the test. If None, the default lag will be used.
    Returns:
        float: The p-value from the unit root test.
    """
    if method == "ADF":
        if n_lag ==None:
            adf = adfuller(series)[1]
        else:
            adf = adfuller(series, maxlag = n_lag)[1]        
        if adf < 0.05:
            return adf, print('ADF p-value: %f' % adf + " and data is stationary at 5% significance level")
        else:
            return adf, print('ADF p-value: %f' % adf + " and data is non-stationary at 5% significance level")
    elif method == "KPSS":
        if n_lag == None:
            kps = kpss(series)[1]
        else:
            kps = kpss(series, nlags = n_lag)[1]
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
    """
    Plots the Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF) of a time series.
    Args:
        series (pd.Series): The time series data.
        lag_num (int): The number of lags to consider.
        figsize (tuple): Size of the figure for the plots.
    Returns:
        None: Displays the plots of PACF and ACF.
    """
    fig, ax = pyplot.subplots(2,1, figsize=figsize)
    plot_pacf(series, lags= lag_num, ax = ax[0])
    plot_acf(series, lags= lag_num, ax = ax[1])
    ax[0].grid(which='both')
    ax[1].grid(which='both')
    pyplot.show()

#------------------------------------------------------------------------------
# Transformation Utility Functions
#------------------------------------------------------------------------------

def fourier_terms(start, stop, period, num_terms, df_index):
    '''
    Returns fourier terms for the given seasonal period and dataframe.

            Parameters:
                    start (int): An integer that should be 0 for the training dataset,
                    whereas for the test dataset, it should correspond to the length of the training data.
                    stop (int): An integer representing the length of the training dataset is required
                    for the training dataset, while for the testing dataset, it should be
                    the sum of the lengths of both the training and test datasets.
                    period (int): the seosanal period.
                    num_terms (int): It specifies how many pairs of sin and cos terms to include.
                    df_index: a dataframe (training or test dataset).
                    It specify whether to use the indexes of the training or test dataset for the returned dataframe
    '''
    t = np.arange(start, stop)
    df = pd.DataFrame(index=df_index.index)
    for i in range(1, num_terms + 1):
        df["sin_"+str(i-1)+"_"+str(period)] = np.sin(2 * np.pi * i * t / period)
        df["cos_"+str(i-1)+"_"+str(period)] = np.cos(2 * np.pi * i * t / period)
    return df

# @jit(nopython=True)
# Your transformation classes
class rolling_mean:
    """
    A class to compute rolling mean with a specified window size and minimum samples.
    Args:
        shift (int): The number of periods to shift time series data.
        window_size (int): The size of the rolling window.
        min_samples (int): Minimum number of samples required to compute the mean.
    """

    def __init__(self, window_size, shift=1, min_samples=1):
        self.shift = shift
        self.window_size = window_size
        self.min_samples = min_samples

    def __call__(self, data, is_forecast=False):
        """
        Compute the rolling mean of the data with the specified parameters.
        Args:
            data (pd.Series or np.ndarray): The input data to compute the rolling mean.
            is_forecast (bool): If True, adjust the shift for forecasting purposes.
        Returns:
            pd.Series: The rolling mean of the input data.
        """

        if is_forecast:
            # For example, if it's a forecast, for the forecasting next value, we might want to shift by one less than usual to align with the forecasted period
            return pd.Series(data).shift(self.shift-1).rolling(self.window_size, min_periods=self.min_samples).mean()
        # If not a forecast, apply the usual shift
        else:
            # If not a forecast, apply the usual shift
            return pd.Series(data).shift(self.shift).rolling(self.window_size, min_periods=self.min_samples).mean()

class rolling_quantile:
    """
    A class to compute rolling quantile with a specified window size and minimum samples.
    Args:
        shift (int): The number of periods to shift time series data.
        window_size (int): The size of the rolling window.
        min_samples (int): Minimum number of samples required to compute the quantile.
    """

    def __init__(self, window_size, quantile, shift=1, min_samples=3):
        self.shift = shift
        self.window_size = window_size
        self.quantile = quantile
        self.min_samples = min_samples

    def __call__(self, data, is_forecast=False):
        # Return the rolling quantile with the specified window size and minimum samples
        if is_forecast:
            return pd.Series(data).shift(self.shift-1).rolling(self.window_size, min_periods=self.min_samples).quantile(self.quantile)
        # If not a forecast, apply the usual shift
        else:
            # If not a forecast, apply the usual shift
            return pd.Series(data).shift(self.shift).rolling(self.window_size, min_periods=self.min_samples).quantile(self.quantile)
    
class rolling_std:
    """
    A class to compute rolling std with a specified window size and minimum samples.
    Args:
        shift (int): The number of periods to shift time series data.
        window_size (int): The size of the rolling window.
        min_samples (int): Minimum number of samples required to compute the std.
    """

    def __init__(self, window_size, shift=1, min_samples=3):
        self.shift = shift
        self.window_size = window_size
        self.min_samples = min_samples

    def __call__(self, data, is_forecast=False):
        # Return the rolling std with the specified window size and minimum samples
        if is_forecast:
            return pd.Series(data).shift(self.shift-1).rolling(self.window_size, min_periods=self.min_samples).std()
        else:
            return pd.Series(data).shift(self.shift).rolling(self.window_size, min_periods=self.min_samples).std()

class expanding_std:
    """
    A class to compute expanding standard deviation.
    Args:
        shift (int): The number of periods to shift time series data.
    """
    def __init__(self, shift=1):
        self.shift = shift
    def __call__(self, data, is_forecast=False):
        if is_forecast:
            return pd.Series(data).shift(self.shift-1).expanding().std()
        else:
            return pd.Series(data).shift(self.shift).expanding().std()

class expanding_mean:
    """
    A class to compute expanding mean.
    Args:
        shift (int): The number of periods to shift time series data.
    """
    def __init__(self, shift=1):
        self.shift = shift
    def __call__(self, data, is_forecast=False):
        if is_forecast:
            return pd.Series(data).shift(self.shift-1).expanding().mean()
        else:
            return pd.Series(data).shift(self.shift).expanding().mean()

class expanding_quantile:
    """
    A class to compute expanding quantile.
    Args:
        quantile (float): The quantile to compute.
    """
    def __init__(self, shift=1, quantile=0.5):
        self.shift = shift
        self.quantile = quantile

    def __call__(self, data, is_forecast=False):
        """
        Compute the expanding quantile.
        """
        if is_forecast:
            return pd.Series(data).shift(self.shift-1).expanding().quantile(self.quantile)
        else:
            return pd.Series(data).shift(self.shift).expanding().quantile(self.quantile)

#------------------------------------------------------------------------------
# Lag Selection Algorithms
#------------------------------------------------------------------------------

def forward_lag_selection(df, max_lags, n_folds, H, model, model_params, metrics, verbose = False):
    """
    Performs forward lag selection for Regression models.
    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        max_lags (int): Maximum number of lags to consider.
        n_folds (int): Number of folds for cross-validation.
        H (int): Forecast horizon.
        model: The forecasting model to be used.
        model_params (dict): Parameters for the model.
        metrics (list): List of metrics to evaluate the model.
        verbose (bool, optional): Whether to print progress. Defaults to False.
    Returns:
        list: List of best lags selected.
    """
    max_lag = max_lags
    orj_lags = list(range(1, max_lags+1))
    lags = list(range(1, max_lags+1))
    best_lags = []
    
    best_score = list(np.repeat(float('inf'), len(metrics)))
    
    while max_lag>0:
        for i in range(max_lag):
            best_lag = None
            for lg in lags:
                current_lag = best_lags + [lg]
                current_lag.sort()
                lag_model = model(**model_params, n_lag = current_lag)
                my_cv = lag_model.cv(df, n_folds,H, metrics)
                scores = my_cv["score"].tolist()
                if scores<best_score:
                    best_score = scores
                    best_lag = lg 
            if best_lag is not None:
                best_lags.append(best_lag)
                lags.remove(best_lag)
                best_lags.sort()
                if verbose == True:
                    print(len(best_lags))
            else:
                break
        if best_lag is None:
            break
        lags = [item for item in orj_lags if item not in best_lags]
        lags.sort()
        if lags == orj_lags:
            break
        else:
            orj_lags = [item for item in orj_lags if item not in best_lags]
            orj_lags.sort()
            max_lag = len(orj_lags)
    return best_lags

def backward_lag_selection(df, max_lags,min_lags, n_folds, H, model, model_params, metrics, forward_back=False, verbose = False):
    """
    Performs backward lag selection for Regression models.
    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        max_lags (int): Maximum number of lags to consider.
        min_lags (int): Minimum number of lags to retain.
        n_folds (int): Number of folds for cross-validation.
        H (int): Forecast horizon.
        model: The forecasting model to be used.
        model_params (dict): Parameters for the model.
        metrics (list): List of metrics to evaluate the model.
        forward_back (bool, optional): Whether to perform forward-backward selection. Defaults to False.
        verbose (bool, optional): Whether to print progress. Defaults to False.
    Returns:
        dict: Dictionary of best lags for each variable.
    """
    max_lag = max_lags
    min_lag =min_lags
    lags = list(range(1, max_lag+1))
    
    best_score = list(np.repeat(float('inf'), len(metrics)))
    
    worst_lags = []
    best_lags = None
    while len(lags) >= min_lag:
        worst_lag=None
        for lg in lags:
            lags_to_test = [x for x in lags if x != lg]
            lags_to_test.sort()
            lag_model = model(**model_params, n_lag = lags_to_test)
            my_cv = lag_model.cv(df, n_folds,H, metrics)
            scores = my_cv["score"].tolist()
            if scores<best_score:
                best_lags = lags_to_test
                worst_lag = lg
        if worst_lag is not None:
            # lags.append(best_lag)
            lags.remove(worst_lag)
            worst_lags.append(worst_lag)
            # print(lags)
            lags.sort()
            best_lags.sort()
            if verbose == True:
                print(len(best_lags))
        else:
            break
    
    if forward_back ==True:
    
        orj_lags = worst_lags.copy()
        orj_lags.sort()
        len_worst = len(orj_lags)
        
        while len(worst_lags)>0:
            for i in range(len_worst):
                best_lag = None
                for lg in worst_lags:
                    current_lag = best_lags + [lg]
                    current_lag.sort()
                    lag_model = model(**model_params, n_lag = current_lag)
                    my_cv = lag_model.cv(df, n_folds,H, metrics)
                    scores = my_cv["score"].tolist()
                    if scores<best_score:
                        best_score = scores
                        best_lag = lg 
                        
                if best_lag is not None:
                    best_lags.append(best_lag)
                    worst_lags.remove(best_lag)
                    if verbose == True:
                        print(len(best_lags))
                    best_lags.sort()
                    # if verbose = True:
                    #     print(best_lags)
                else:
                    break
            if best_lag is None:
                break
            worst_lags = [item for item in orj_lags if item not in best_lags]
            lags.sort()
            if worst_lags == orj_lags:
                break
            else:
                orj_lags = [item for item in orj_lags if item not in best_lags]
                orj_lags.sort()
    return best_lags


def var_forward_lag_selection(df, model, max_lags, target_col, n_folds, H, model_params, metrics, verbose = False):
    """
    Performs forward lag selection for Vektor Autoregressive models.
    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        model: The forecasting model to be used.
        max_lags (dict): Dictionary of maximum lags for each variable.
        n_folds (int): Number of folds for cross-validation.
        target_col (str): The target column for forecasting.
        H (int): Forecast horizon.
        model_params (dict): Parameters for the model.
        metrics (list): List of metrics to evaluate the model.
        verbose (bool, optional): Whether to print progress. Defaults to False.
    Returns:
        dict: Dictionary of best lags for each variable.
    """

    max_lag = sum(x for x in max_lags.values())
        
    orj_lags = {i:list(range(1, max_lags[i]+1)) for i in max_lags}
    # lags = list(range(1, max_lags+1))
    lags = {i:list(range(1, max_lags[i]+1)) for i in max_lags}
    best_lags = {i:[] for i in max_lags}
    
    best_score = list(np.repeat(float('inf'), len(metrics)))
    
    while max_lag>0:
        for i in range(max_lag):
            best_lag = None
            best_target = None
            for k, lg in lags.items():
                for x in lg:
                    current_lag = {a:b for a, b in best_lags.items()}
                    current_lag[k] = best_lags[k] + [x]
                    current_lag[k].sort()
                    lag_model = model(**model_params, lag_dict = current_lag)
                    my_cv = lag_model.cv_var(df, target_col, n_folds, H, metrics)
                    scores = my_cv["score"].tolist()
                    if scores<best_score:
                        best_score = scores
                        best_lag = x
                        best_target = k
            if best_lag is not None:
                best_lags[best_target].append(best_lag)
                lags[best_target].remove(best_lag)
                best_lags[best_target].sort()
                max_lag = sum(len(x) for x in lags.values())
                if verbose == True:
                    print("Best lag: ", best_target, best_lag)
            else:
                break
        lags = {i:[item for item in orj_lags[i] if item not in best_lags[i]] for i in max_lags.keys()}
        # lags.sort()
        if lags == orj_lags:
            break
        else:
            orj_lags = {i:[item for item in orj_lags[i] if item not in best_lags[i]] for i in max_lags.keys()}
            # orj_lags.sort()
            max_lag = sum(len(x) for x in orj_lags.values())

    return best_lags

def var_backward_lag_selection(df, model, max_lags, min_lags, n_folds,target_col, H, model_params, metrics, forward_back=False, verbose = False):
    """
    Performs backward lag selection for Vektor Autoregressive models.
    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        model: The forecasting model to be used.
        max_lags (dict): Dictionary of maximum lags for each variable.
        min_lags (int): Minimum number of lags to retain.
        n_folds (int): Number of folds for cross-validation.
        target_col (str): The target column for forecasting.
        H (int): Forecast horizon.
        model_params (dict): Parameters for the model.
        metrics (list): List of metrics to evaluate the model.
        forward_back (bool, optional): Whether to perform forward-backward selection. Defaults to False.
        verbose (bool, optional): Whether to print progress. Defaults to False.
    Returns:
        dict: Dictionary of best lags for each variable.
    """

    max_lag = sum(x for x in max_lags.values())
    min_lag =min_lags   
    
    orj_lags = {i:list(range(1, max_lags[i]+1)) for i in max_lags}
    # lags = list(range(1, max_lags+1))
    lags = {i:list(range(1, max_lags[i]+1)) for i in max_lags}
    best_lags = {i:[] for i in max_lags}
    
    best_score = list(np.repeat(float('inf'), len(metrics)))
    
    worst_lags = {i:[] for i in max_lags}
    while max_lag >= min_lag:
        worst_lag=None
        worst_k = None
        for k, lg in lags.items():
            for r in lg:
                lags_to_test = {a:b for a, b in lags.items()}
                lags_to_test[k] = [x for x in lg if x != r]
                lags_to_test[k].sort()
                lag_model = model(**model_params, lag_dict = lags_to_test)
                my_cv = lag_model.cv_var(df, target_col, n_folds, H, metrics)
                scores = my_cv["score"].tolist()
                if scores<best_score:
                    best_score = scores
                    worst_lag = r
                    worst_k = k
                    best_lags = lags_to_test
        if worst_lag is not None:
            # lags.append(best_lag)
            lags[worst_k].remove(worst_lag)
            lags[worst_k].sort()
            # best_lags[worst_k].remove(worst_lag)
            worst_lags[worst_k].append(worst_lag)
            worst_lags[worst_k].sort()
            if verbose == True:
                print("worst_lag: ", worst_k, worst_lag)
            max_lag = sum(len(x) for x in lags.values())
                
        else:
            break
    
    if forward_back ==True:
    
        orj_lags = worst_lags.copy()
        # orj_lags.sort()
        len_worst = sum(len(x) for x in worst_lags.values())
        
        while len_worst>0:
            for i in range(len_worst):
                best_lag = None
                best_k = None
                for k, lg in worst_lags.items():
                    for x in lg:
                        current_lag = {a:b for a, b in best_lags.items()}
                        current_lag[k] = best_lags[k] + [x]
                        current_lag[k].sort()
                        lag_model = model(**model_params, lag_dict = current_lag)
                        my_cv = lag_model.cv_var(df, target_col, n_folds, H, metrics)
                        scores = my_cv["score"].tolist()
                        if scores<best_score:
                            best_score = scores
                            best_lag = x
                            best_k = k
       
                if best_lag is not None:
                    best_lags[best_k].append(best_lag)
                    worst_lags[best_k].remove(best_lag)

                    best_lags[best_k].sort()
                    len_worst = sum(len(x) for x in worst_lags.values()) #  update len worst
                else:
                    break
                    
            worst_lags = {i:[item for item in orj_lags[i] if item not in best_lags[i]] for i in max_lags}
            len_worst = sum(len(x) for x in worst_lags.values())
            
            if worst_lags == orj_lags: #check if no lags is added
                break
            else:
                orj_lags = {i:[item for item in orj_lags[i] if item not in best_lags[i]] for i in max_lags}
    return best_lags


#------------------------------------------------------------------------------
# K-Fold Target Encoding
#------------------------------------------------------------------------------

def Kfold_target(train, test, cat_var, target_col, encoded_colname, split_num):
    """ 
    Perform K-Fold encoding for categorical variables.

    Args:
        train (pd.DataFrame): Training dataset.
        test (pd.DataFrame): Test dataset.
        cat_var (str): Categorical variable to encode.
        target_col (str): Target variable for encoding.
        encoded_colname (str): Name of the new encoded column.
        split_num (int): Number of splits for K-Fold encoding.
    Returns:
        pd.DataFrame: Updated training and test datasets with the new encoded column.
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = split_num, shuffle = True)
    train[encoded_colname] = np.nan
    for tr_ind, val_ind in kf.split(train):
        X_tr, X_val = train.iloc[tr_ind], train.iloc[val_ind]
        cal_df = X_tr.groupby(cat_var)[target_col].mean().reset_index()
        train.loc[train.index[val_ind], encoded_colname] = X_val[cat_var].merge(cal_df, on = cat_var, how = "left")[target_col].values
        
    train.loc[train[encoded_colname].isnull(), encoded_colname] = train[train[encoded_colname].isnull()][encoded_colname].mean()
    map_df = train.groupby(cat_var)[encoded_colname].mean().reset_index()
    test[encoded_colname] = test[cat_var].merge(map_df, on = cat_var, how= "left")[encoded_colname].values
    test.loc[test[encoded_colname].isnull(), encoded_colname] = map_df[encoded_colname].mean()
    return train, test

#------------------------------------------------------------------------------
# Evaluation Metrics
#------------------------------------------------------------------------------

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

def MeanAbsoluteScaledError(y_true, y_pred, y_train):
    """
    Calculate Mean Absolute Scaled Error (MASE)
    
    Parameters:
    y_true (array-like): Actual values
    y_pred (array-like): Predicted values
    y_train (array-like): Training data used to scale the error
    
    Returns:
    float: MASE value
    """
    # Calculate the mean absolute error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate the scaled error
    scaled_error = np.mean(np.abs(np.diff(y_train)))
    
    # Calculate MASE
    mase = mae / scaled_error
    
    return mase

def MedianAbsoluteScaledError(y_true, y_pred, y_train):
    """
    Calculate Median Absolute Scaled Error (MASE)
    
    Parameters:
    y_true (array-like): Actual values
    y_pred (array-like): Predicted values
    y_train (array-like): Training data used to scale the error
    
    Returns:
    float: MASE value
    """
    # Calculate the mean absolute error
    mae = np.median(np.abs(y_true - y_pred))
    
    # Calculate the scaled error
    scaled_error = np.median(np.abs(np.diff(y_train)))
    
    # Calculate MASE
    mase = mae / scaled_error
    
    return mase


def cfe(y_true, y_pred):
    return np.cumsum([a - f for a, f in zip(y_true, y_pred)])[-1]
def cfe_abs(y_true, y_pred):
    cfe_t = np.cumsum([a - f for a, f in zip(y_true, y_pred)])
    return np.abs(cfe_t[-1])

def wmape(y_true, y_pred):
    """
    Calculate Weighted Mean Absolute Percentage Error (WMAPE).
    
    Parameters:
    y_true (array-like): Actual values.
    y_pred (array-like): Forecasted values.
    
    Returns:
    float: WMAPE value.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    wmape_value = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
    
    return wmape_value

#------------------------------------------------------------------------------
# Box-Cox Transformation Utility Functions
#------------------------------------------------------------------------------

def box_cox_transform(x, shift=False, box_cox_lmda = None):
    """
    Applies a Box-Cox transformation to a series x.

    Args:
        x (array-like): The input data.
        shift (bool): add 1 to each element of data if True, default is False.
        box_cox_lmda (float): The lambda parameter for the Box-Cox transformation.

    Returns:
        tuple: (transformed data, updated lambda)
    """
    if (box_cox_lmda == None):
        if shift ==True:
            transformed_data, lmbda = boxcox((np.array(x)+1))
        else:
            transformed_data, lmbda = boxcox(np.array(x))
            
    if (box_cox_lmda != None):
        if shift ==True:
            lmbda = box_cox_lmda
            transformed_data = boxcox((np.array(x)+1), lmbda)
        else:
            lmbda = box_cox_lmda
            transformed_data = boxcox(np.array(x), lmbda)
    return transformed_data, lmbda


def back_box_cox_transform(y_pred, lmda, shift=False, box_cox_biasadj=False):
    """
    Inverse Box-Cox transform.

    Args:
        y_pred (array-like): Transformed forecast.
        lmda (float): Box-Cox lambda.
        shift (bool): Whether the original data was shifted.
        box_cox_biasadj (bool): Whether bias adjustment is applied.
    
    Returns:
        array-like: Back-transformed forecast.
    """
    if (box_cox_biasadj==False):
        if shift == True:
            forecast = inv_boxcox(y_pred, lmda)-1
        else:
            forecast = inv_boxcox(y_pred, lmda)
            
    if (box_cox_biasadj== True):
        pred_var = np.var(y_pred)
        if shift == True:
            if lmda ==0:
                forecast = np.exp(y_pred)*(1+pred_var/2)-1
            else:
                forecast = ((lmda*y_pred+1)**(1/lmda))*(1+((1-lmda)*pred_var)/(2*((lmda*y_pred+1)**2)))-1
        else:
            if lmda ==0:
                forecast = np.exp(y_pred)*(1+pred_var/2)
            else:
                forecast = ((lmda*y_pred+1)**(1/lmda))*(1+((1-lmda)*pred_var)/(2*((lmda*y_pred+1)**2)))
    return forecast

#------------------------------------------------------------------------------
# Differencing Utility Functions
#------------------------------------------------------------------------------

def undiff_ts(original_data, differenced_data, n):
    """
    Inverts (ordinary) differencing.

    Args:
        original_data: The original time series.
        differenced_data: The n-differenced series.
        n (int): The degree of differencing.

    Returns:
        array-like: Undifferenced series.
    """
    undiff_data = np.array(differenced_data)
    if n > 1:
        for i in range(n-1, 0, -1):
            undiff_data = np.diff(original_data, i)[-1]+np.cumsum(undiff_data)
    
    return original_data[-1]+np.cumsum(undiff_data)

def seasonal_diff(data, seasonal_period):
    """
    Computes seasonal differences on array x.

    Args:
        x (array-like): The time series.
        seasonal_period (int): The seasonal period.
        
    Returns:
        array-like: Seasonally differenced time series.
    """
    orig_data = list(np.repeat(np.nan, seasonal_period))+[data[i] - data[i - seasonal_period] for i in range(seasonal_period, len(data))]
    return np.array(orig_data)

# invert difference
def invert_seasonal_diff(orig_data, diff_data, seasonal_period):
    """
    Inverts seasonal differencing.

    Args:
        orig (list): The original series values.
        diff (array-like): The differenced series.
        seasonal_period (int): The seasonal period.
        
    Returns:
        array-like: Reconstructed series.
    """
    conc_data = list(orig_data[-seasonal_period:]) + list(diff_data)
    for i in range(len(conc_data)-seasonal_period):
        conc_data[i+seasonal_period] = conc_data[i]+conc_data[i+seasonal_period]

    return np.array(conc_data[-len(diff_data):])

#------------------------------------------------------------------------------
# Croston's Method metrics
#------------------------------------------------------------------------------

# @jit(nopython=True)
def nzInterval(data, lag=0):
    """
    Computes the intervals between non-zero values in a time series.

    Args:
        data (array-like): Input time series data.
        lag (int): Number of lags to consider. Default is 0.

    Returns:
        numpy.ndarray: An array of intervals between non-zero values, with NaNs for the first `lag` values.
    """
    intervals = []
    last_nonzero_times = []
    if lag !=0:
        arr = data[:-lag]
    else:
        arr = data

    for i, j in enumerate(arr):
        if j > 0.5:
            last_nonzero_times.append(i)
        
        if len(last_nonzero_times)==1:
            intervals.append(1)
        else:
            inter = last_nonzero_times[-1]-last_nonzero_times[-2]
            intervals.append(inter)
    
    if lag !=0:
        nas = list(np.repeat(np.nan, lag))
        intervals = nas+intervals
    return np.array(intervals)

# @jit(nopython=True)
def zeroCumulative(data, lag=0):
    """
    Computes the cumulative count of consecutive zeros in a time series.
    Args:
        data (array-like): Input time series data.
        lag (int): Number of lags to consider. Default is 0.
    Returns:
        numpy.ndarray: An array of cumulative counts of consecutive zeros, with NaNs for the first `lag` values.
    """
    if lag !=0:
        arr = data[:-lag]
    else:
        arr = data

    count = 0
    result = []
    
    for value in arr:
        if value < 0.5:
            count += 1
        else:
            count = 0
        result.append(count)

    if lag !=0:
        nas = list(np.repeat(np.nan, lag))
        result = nas+result
    return np.array(result)


#------------------------------------------------------------------------------
# Target Encoding Utility Functions
#------------------------------------------------------------------------------

def kfold_target_encoder(df, feature_col, target_col, n_splits=5):
    """
    Perform K-fold target encoding for a given feature.

    Args:
        df (pd.DataFrame): Input dataframe.
        feature_col (str): Column name of the categorical feature.
        target_col (str): Target column name.
        n_splits (int): Number of folds for target encoding.

    Returns:
        np.array: Encoded target values.
    """
    df_encoded = df.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoded_col_name = f"{feature_col}_target_encoded"
    df_encoded[encoded_col_name] = np.nan
    feat_idx = df_encoded.columns.get_loc(feature_col)
    encode_idx = df_encoded.columns.get_loc(encoded_col_name)
    # Perform KFold encoding
    for train_idx, val_idx in kf.split(df):
        # Calculate target mean per category using only the training fold
        target_means = df.iloc[train_idx].groupby(feature_col)[target_col].mean()
        # Map means to the validation fold
        df_encoded.iloc[val_idx, encode_idx] = df_encoded.iloc[val_idx, feat_idx].map(target_means)
    # Fill missing values with overall target mean
    overall_mean = df[target_col].mean()
    df_encoded[encoded_col_name].fillna(overall_mean, inplace=True)
    return df_encoded[encoded_col_name].values

def target_encoder_for_test(train_df, test_df, feature_col):
    """
    Apply target encoding to the test set based on the training set.

    Args:
        train_df (pd.DataFrame): Training dataframe with encoded feature.
        test_df (pd.DataFrame): Test dataframe.
        feature_col (str): Column name to encode.

    Returns:
        np.array: Encoded values for the test set.
    """
    encoded_col_name = f"{feature_col}_target_encoded"
    target_means = train_df.groupby(feature_col)[encoded_col_name].mean()
    overall_mean = train_df[encoded_col_name].mean()
    test_encoded = test_df.copy()
    test_encoded[encoded_col_name] = test_df[feature_col].map(target_means)
    test_encoded[encoded_col_name].fillna(overall_mean, inplace=True)
    return test_encoded[encoded_col_name].values

#------------------------------------------------------------------------------
# Holt-Winters Exponential Smoothing Model Tuning
#------------------------------------------------------------------------------

def tune_ets(data, param_space, cv_splits, horizon, eval_metric, eval_num, append_horizons = False, verbose = False):
    """
    Tune ETS model hyperparameters using Hyperopt.

    Args:
        data (array-like): Time series data.
        param_space (dict): Hyperparameter search space.
        cv_splits (int): Number of cross-validation splits.
        horizon (int): Forecast horizon.
        eval_metric (function): Evaluation metric function.
        eval_num (int): Number of evaluations for hyperparameter tuning.
        append_horizons (bool): Whether to append horizons to the forecast.
        verbose (bool): Whether to print progress.
    Returns:
        tuple: Best model parameters and fit parameters.
    """

    from sklearn.model_selection import TimeSeriesSplit
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
    from hyperopt.pyll import scope
    tscv = TimeSeriesSplit(n_splits=cv_splits, test_size=horizon)
    # Define the objective function for hyperparameter tuning
    def objective(params):
        if (params["trend"] != None) & (params["seasonal"] != None): # Both trend and seasonal are specified
            alpha = params['smoothing_level'] # Smoothing level for the level component
            beta = params['smoothing_trend'] # Smoothing level for the trend component
            gamma = params['smoothing_seasonal'] # Smoothing level for the seasonal component
            trend_type = params['trend'] # Trend type
            season_type = params['seasonal'] # Seasonal type
            S = params['seasonal_periods'] # Seasonal periods
            if params["damped_trend"] == True: # Damped trend
                damped_bool = params["damped_trend"]
                damp_trend = params['damping_trend']
            else:
                damped_bool = params["damped_trend"]
                damp_trend = None

        elif (params["trend"] != None) & (params["seasonal"] == None): # Trend is specified, seasonal is not
            alpha = params['smoothing_level']
            beta = params['smoothing_trend']
            gamma = None
            trend_type = params['trend']
            season_type = params['seasonal']
            S=None
            if params["damped_trend"] == True:
                damped_bool = params["damped_trend"]
                damp_trend = params['damping_trend']
            else:
                damped_bool = params["damped_trend"]
                damp_trend = None
                
        elif (params["trend"] == None) & (params["seasonal"] != None): # Seasonal is specified, trend is not
            alpha = params['smoothing_level']
            beta = None
            gamma = params['smoothing_seasonal']
            trend_type = params['trend']
            season_type = params['seasonal']
            S=params['seasonal_periods']
            if params["damped_trend"] == True:
                damped_bool = False
                damp_trend = None
            else:
                damped_bool = params["damped_trend"]
                damp_trend = None
                
        else: # Neither trend nor seasonal is specified
            alpha = params['smoothing_level']
            beta = None
            gamma = None
            trend_type = params['trend']
            season_type = params['seasonal']
            S=None
            if params["damped_trend"] == True:
                damped_bool = False
                damp_trend = None
            else:
                damped_bool = params["damped_trend"]
                damp_trend = None
            
    
    
        if append_horizons is True:
            actuals = []
            forecasts = []
        metric = [] # List to store evaluation metrics
        # Perform time series cross-validation
        for train_index, test_index in tscv.split(data):
            train, test = data[train_index], data[test_index]
            # Fit the Holt-Winters model with the specified parameters
            hw_fit = ExponentialSmoothing(train ,seasonal_periods=S , seasonal=season_type, trend=trend_type, damped_trend = damped_bool).fit(smoothing_level = alpha, 
                                                                                                                      smoothing_trend = beta,
                                                                                                                      smoothing_seasonal = gamma,
                                                                                                damping_trend=damp_trend)
            
            hw_forecast = hw_fit.forecast(len(test))
            forecast_filled = np.nan_to_num(hw_forecast, nan=0)
            if append_horizons is True:
                forecasts += list(forecast_filled)
                actuals += list(test)
            else:
                if eval_metric.__name__== 'mase':
                    accuracy = eval_metric(test, forecast_filled, np.array(train))
                else:
                    accuracy = eval_metric(test, forecast_filled)
                metric.append(accuracy)
        if append_horizons is True:
            forecasts = np.array(forecasts)
            actuals = np.array(actuals)
            if eval_metric.__name__== 'mase':
                accuracy = eval_metric(actuals, forecasts, np.array(train))
            else:
                accuracy = eval_metric(actuals, forecasts)
            metric.append(accuracy)
        score = np.mean(metric)
        if verbose ==True:
            print ("SCORE:", score)
        return {'loss':score, 'status':STATUS_OK}
    
    # Perform hyperparameter optimization using Hyperopt
    trials = Trials()
    
    best_hyperparams = fmin(fn = objective,
                    space = param_space,
                    algo = tpe.suggest,
                    max_evals = eval_num,
                    trials = trials)
    best_params = space_eval(param_space, best_hyperparams)
    model_params = {"trend": best_params["trend"], "seasonal_periods": best_params["seasonal_periods"], "seasonal": best_params["seasonal"], 
                    "damped_trend": best_params["damped_trend"]}

    fit_params = {"smoothing_level": best_params["smoothing_level"], "smoothing_trend": best_params["smoothing_trend"], "smoothing_seasonal": best_params["smoothing_seasonal"], 
                  "damping_trend": best_params["damping_trend"]}
    if model_params["trend"]==None:
        model_params.pop('trend')
        model_params.pop('damped_trend')
        fit_params.pop('damping_trend')
        fit_params.pop('smoothing_trend')

    if model_params["seasonal"]==None:
        model_params.pop('seasonal')
        model_params.pop('seasonal_periods')
        fit_params.pop('smoothing_seasonal')
    return model_params, fit_params

#------------------------------------------------------------------------------
# SARIMA Model Tuning
#------------------------------------------------------------------------------

def tune_sarima(y, d, D, season,p_range, q_range, P_range, Q_range, X=None):
    """
    Finds the best SARIMA parameters using AIC as the evaluation metric.
    Args:
        y (array-like): The time series data.
        d (int): The non-seasonal differencing order.
        D (int): The seasonal differencing order.
        season (int): The seasonal period.
        p_range (list): Range of non-seasonal AR orders to test.
        q_range (list): Range of non-seasonal MA orders to test.
        P_range (list): Range of seasonal AR orders to test.
        Q_range (list): Range of seasonal MA orders to test.
        X (array-like, optional): Exogenous variables. Defaults to None.
    Returns:
        pd.DataFrame: A DataFrame containing the combinations of parameters and their corresponding AIC values.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from tqdm import tqdm_notebook
    from itertools import product
    if X is not None:
        X = np.array(X, dtype = np.float64)
    p = p_range
    q = q_range # MA(q)
    P = P_range # Seasonal autoregressive order.
    Q = Q_range #Seasonal moving average order.
    parameters = product(p, q, P, Q) # combinations of parameters(cartesian product)
    parameters_list = list(parameters)
    result = []
    for param in tqdm_notebook(parameters_list):
        try:
            model = SARIMAX(endog=y, exog = X, order = (param[0], d, param[1]), seasonal_order= (param[2], D, param[3], season)).fit(disp = -1)
        except:
            continue
                            
        aic = model.aic
        result.append([param, aic])
    result_df = pd.DataFrame(result)
    result_df.columns = ["(p, q)x(P, Q)", "AIC"] 
    result_df = result_df.sort_values("AIC", ascending = True) #Sort in ascending order, lower AIC is better
    return result_df

#------------------------------------------------------------------------------
# ML tuning utility function
#------------------------------------------------------------------------------

def cross_validate(self, model, df, cv_split, test_size, metrics):
    """
    Run cross-validation using time series splits.

    Args:
        model (class): Machine learning model class (e.g., CatBoostRegressor, LGBMRegressor).
        df (pd.DataFrame): Input data.
        cv_split (int): Number of splits in TimeSeriesSplit.
        test_size (int): Size of test window.
        metrics (list): List of metric functions.
    
    Returns:
        pd.DataFrame: Performance metrics for CV.
    """
    tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
    self.metrics_dict = {m.__name__: [] for m in metrics}
    for train_index, test_index in tscv.split(df):
        train, test = df.iloc[train_index], df.iloc[test_index]
        x_test = test.drop(columns=[model.target_col])
        y_test = np.array(test[model.target_col])
        model.fit(train)
        bb_forecast = model.forecast(test_size, x_test=x_test)
        # Evaluate each metric
        for m in metrics:
            if m.__name__ == 'mean_squared_error':
                eval_val = m(y_test, bb_forecast, squared=False)
            elif m.__name__ in ['MeanAbsoluteScaledError', 'MedianAbsoluteScaledError']:
                eval_val = m(y_test, bb_forecast, np.array(train[self.target_col]))
            else:
                eval_val = m(y_test, bb_forecast)
            self.metrics_dict[m.__name__].append(eval_val)
    overall_performance = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]
    return pd.DataFrame(overall_performance).rename(columns={0: "eval_metric", 1: "score"})


def bidirectional_cross_validate(self, model, df, cv_split, test_size, metrics):
    """
    Cross-validate the bidirectional CatBoost model with time series split.
    Args:
        df (pd.DataFrame): Input dataframe.
        cv_split (int): Number of folds.
        test_size (int): Forecast window for each split.
        metrics (list): List of evaluation metric functions.
    Returns:
        pd.DataFrame: CV performance metrics for each target variable.
    """
    tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
    self.metrics_dict = {m.__name__: [] for m in metrics}
    self.cv_fi = pd.DataFrame()
    self.cv_forecasts_df = pd.DataFrame()
    for i, (train_index, test_index) in enumerate(tscv.split(df)):
        train, test = df.iloc[train_index], df.iloc[test_index]
        x_test = test.drop(columns=model.target_cols)
        y_test1 = np.array(test[model.target_cols[0]])
        y_test2 = np.array(test[model.target_cols[1]])
        
        model.fit(train)

        forecast_vals1, forecast_vals2 = model.forecast(test_size, x_test=x_test)
        forecat_df = test[self.target_cols]
        forecat_df["forecasts1"] = forecast_vals1
        forecat_df["forecasts2"] = forecast_vals2
        self.cv_forecasts_df = pd.concat([self.cv_forecasts_df, forecat_df], axis=0)
        for m in metrics:
            if m.__name__ == 'mean_squared_error':
                val1 = m(y_test1, forecast_vals1, squared=False)
                val2 = m(y_test2, forecast_vals2, squared=False)
            elif m.__name__ in ['MeanAbsoluteScaledError', 'MedianAbsoluteScaledError']:
                val1 = m(y_test1, forecast_vals1, np.array(train[self.target_cols[0]]))
                val2 = m(y_test2, forecast_vals2, np.array(train[self.target_cols[1]]))
            else:
                val1 = m(y_test1, forecast_vals1)
                val2 = m(y_test2, forecast_vals2)
            self.metrics_dict[m.__name__].append([val1, val2])
        cv_tr_df1 = pd.DataFrame({"feat_name": self.model1_fit.feature_names_in_,
                                "importance": self.model1_fit.feature_importances_}).sort_values(by="importance", ascending=False)
        cv_tr_df1["target"] = self.target_cols[0]
        cv_tr_df1["fold"] = i
        cv_tr_df2 = pd.DataFrame({"feat_name": self.model2_fit.feature_names_in_,
                                "importance": self.model2_fit.feature_importances_}).sort_values(by="importance", ascending=False)
        cv_tr_df2["target"] = self.target_cols[1]
        cv_tr_df2["fold"] = i
        self.cv_fi = pd.concat([self.cv_fi, cv_tr_df1, cv_tr_df2], axis=0)
    overall = [[m.__name__, np.mean([v[0] for v in self.metrics_dict[m.__name__]])] for m in metrics]
    # pd.DataFrame(overall).rename(columns={0: "eval_metric", 1: "score1", 2: "score2"})
    return pd.DataFrame(overall).rename(columns={0: "eval_metric", 1: f"score_{self.target_cols[0]}", 2: f"score_{self.target_cols[1]}"})

def cv_tune(
    model,
    df,
    cv_split,
    test_size,
    param_space,
    eval_metric,
    step_size=None,
    opt_horizon=None,
    eval_num=100,
    verbose=False,
):
    """
    Tune forecasting model hyperparameters using cross-validation and Bayesian optimization.

    Args:
        model: Forecasting model object with .fit and .forecast methods and relevant attributes.
        df (pd.DataFrame): Time series dataframe.
        cv_split (int): Number of time series splits.
        test_size (int): Size of test window for each split.
        step_size (int): Step size for moving the window. Defaults to None (equal to test_size).
        param_space (dict): Hyperopt parameter search space.
        eval_metric (callable): Evaluation metric function.
        opt_horizon (int, optional): Evaluate only on last N points of each split. Defaults to None (all points).
        eval_num (int, optional): Number of hyperopt evaluations. Defaults to 100.
        verbose (bool, optional): Print progress. Defaults to False.

    Returns:
        dict: Best hyperparameter values found.
    """
    tscv = ParametricTimeSeriesSplit(n_splits=cv_split, test_size=test_size, step_size=step_size)

    def _set_model_params(params):
        # Handle special model parameters that are not passed to model constructor
        # and must be set directly on the forecasting model object
        if "n_lag" in params:
            model.n_lag = list(params["n_lag"]) if isinstance(params["n_lag"], tuple) else range(1, params["n_lag"] + 1)
        if "difference" in params:
            model.difference = params["difference"]
        if "box_cox" in params:
            model.box_cox = params["box_cox"]
        if "box_cox_lmda" in params:
            model.lmda = params["box_cox_lmda"]
        if "box_cox_biasadj" in params:
            model.biasadj = params["box_cox_biasadj"]

        # Handle ETS trend settings
        if getattr(model, "trend_type", None) in ("ses", "feature_ses"):
            model.ets_model = {
                k: params[k] for k in ["trend", "damped_trend", "seasonal", "seasonal_periods"] if k in params
            }
            model.ets_fit = {}
            for k in ["smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend"]:
                if k in params:
                    # Only set "damping_trend" if "damped_trend" is True
                    if k == "damping_trend" and "damped_trend" in params and not params["damped_trend"]:
                        continue
                    model.ets_fit[k] = params[k]

    def _get_model_params_for_fit(params):
        # Exclude special parameters that should not be passed to the model constructor
        skip_keys = {
            "box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj",
            "trend", "damped_trend", "seasonal", "seasonal_periods",
            "smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend",
            "differencing_number"
        }
        return {k: v for k, v in params.items() if k not in skip_keys}

    def objective(params):
        _set_model_params(params)
        model_params = _get_model_params_for_fit(params) if getattr(model, "model", None) and model.model.__name__ != "LinearRegression" else None

        metrics = []
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test = test.drop(columns=[model.target_col])
            y_test = np.array(test[model.target_col])

            if model_params is not None:
                model.model.set_params(**model_params)
                model.fit(train)
            else:
                model.fit(train)
            y_pred = model.forecast(n_ahead=len(y_test), x_test=x_test)

            # Evaluate using the specified metric
            if eval_metric.__name__ == "mean_squared_error":
                score = eval_metric(
                    y_test[-opt_horizon:] if opt_horizon else y_test,
                    y_pred[-opt_horizon:] if opt_horizon else y_pred,
                    squared=False,
                )
            elif eval_metric.__name__ in ("MeanAbsoluteScaledError", "MedianAbsoluteScaledError"):
                score = eval_metric(
                    y_test[-opt_horizon:] if opt_horizon else y_test,
                    y_pred[-opt_horizon:] if opt_horizon else y_pred,
                    np.array(train[model.target_col])
                )
            else:
                score = eval_metric(
                    y_test[-opt_horizon:] if opt_horizon else y_test,
                    y_pred[-opt_horizon:] if opt_horizon else y_pred,
                )
            metrics.append(score)

        mean_score = np.mean(metrics)
        if verbose:
            print("Score:", mean_score)
        return {"loss": mean_score, "status": STATUS_OK}

    trials = Trials()
    best_hyperparams = fmin(
        fn=objective,
        space=param_space,
        algo=tpe.suggest,
        max_evals=eval_num,
        trials=trials,
    )

    model.tuned_params = [
        space_eval(param_space, {k: v[0] for k, v in t["misc"]["vals"].items()})
        for t in trials.trials
    ]

    return space_eval(param_space, best_hyperparams)

#------------------------------------------------------------------------------
# Regression Detrending and Forecasting
#------------------------------------------------------------------------------

def regression_detrend(series):
    """
    Detrends a time series using linear regression.
    Args:
        series (array-like): The time series data to be detrended.
    Returns:
        np.ndarray: The detrended time series.
    """
    model = LinearRegression().fit(np.array(range(len(series))).reshape(-1, 1),series)
# Make predictions
    y_pred = model.predict(np.array(range(len(series))).reshape(-1, 1))
    return np.array(series)-y_pred

def forecast_trend(train_series, H):
    """ Forecasts the trend of a time series using linear regression.
    Args:
        train_series (array-like): The training time series data.
        H (int): The forecast horizon.
    Returns:
        np.ndarray: The forecasted trend values.
    """
    model = LinearRegression().fit(np.array(range(len(train_series))).reshape(-1, 1),train_series)
    return model.predict(np.array(range(len(train_series), len(train_series)+H)).reshape(-1, 1))


#------------------------------------------------------------------------------
# Parametric Time Series Split
#------------------------------------------------------------------------------
class ParametricTimeSeriesSplit:
    """
    Rolling window time series cross-validator with fixed step size

    Parameters:
        test_size (int): Number of test samples in each split.
        step_size (int): Number of steps to move backward for each split.
        n_splits (int, optional): Number of splits to generate.

    Yields:
        train_index, test_index: Indices for training and test sets.
    """
    def __init__(self, n_splits, test_size, step_size=None):
        self.test_size = test_size
        self.step_size = test_size if step_size is None else step_size
        self.n_splits = n_splits

    def split(self, X):
        n_samples = len(X)
        split_starts = []
        # Start the last test set at the last possible position
        last_test_start = n_samples - self.test_size
        # Build test starts, moving backward with fixed step_size
        current = last_test_start
        while current >= 0:
            split_starts.append(current)
            current -= self.step_size
        split_starts = split_starts[::-1]  # Reverse to start from earliest

        # Use only the last n_splits
        split_starts = split_starts[-self.n_splits:]

        for test_start in split_starts:
            test_end = test_start + self.test_size
            train_index = np.arange(0, test_start)
            test_index = np.arange(test_start, test_end)
            yield train_index, test_index


def tune_model_bi1(self, df, forecast_col, cv_split, test_size, param_space, eval_metric, eval_num = 100):
    if self.cat_variables is not None:
        self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
        self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
    tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
    
    def objective(params):
        model1=self.model(**params)
        model2 =self.model(**params)

        metric = []
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[forecast_col])
            model_train = self.data_prep(train)
            
            self.X, self.y1, self.y2 = model_train.drop(columns =self.target_col), model_train[self.target_col[0]], model_train[self.target_col[1]]
            self.model_ada1 = model1.fit(self.X, self.y1)
            self.model_ada2 = model2.fit(self.X, self.y2)
            yhat = self.forecast(n_ahead =len(y_test), x_test=x_test)
            
            if eval_metric.__name__== 'mean_squared_error':
                if forecast_col == self.target_col[0]:
                    accuracy = eval_metric(y_test, yhat[0], squared=False)
                else:
                    accuracy = eval_metric(y_test, yhat[1], squared=False)
                
            else:
                if forecast_col == self.target_col[0]:
                    accuracy = eval_metric(y_test, yhat[0])
                    
                else:
                    accuracy = eval_metric(y_test, yhat[1])
#                 print(str(accuracy)+" and len is "+str(len(test)))
            metric.append(accuracy)
        score = np.mean(metric)

        print ("SCORE:", score)
        return {'loss':score, 'status':STATUS_OK}
        
    trials = Trials()

    best_hyperparams = fmin(fn = objective,
                    space = param_space,
                    algo = tpe.suggest,
                    max_evals = eval_num,
                    trials = trials)
    # best_params = {i: int(best_hyperparams[i]) if i in ["num_iterations", "num_leaves", "max_depth","min_data_in_leaf", "top_k"] 
    #                    else best_hyperparams[i] for i in best_hyperparams}
    return space_eval(param_space, best_hyperparams)

def tune_model_bi2(df, forecast_col, cv_split, test_size, param_space, eval_metric, eval_num = 100):
    tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
    
    def objective(params):
        model1=self.model(**params, verbose=-1)
        model2 =self.model(**params, verbose=-1)

        metric = []
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[forecast_col])
            model_train = self.data_prep(train)
            
            self.X, self.y1, self.y2 = model_train.drop(columns =self.target_col), model_train[self.target_col[0]], model_train[self.target_col[1]]
            self.model_lgb1 = model1.fit(self.X, self.y1, categorical_feature=self.cat_var)
            self.model_lgb2 = model2.fit(self.X, self.y2, categorical_feature=self.cat_var)
            yhat = self.forecast(n_ahead =len(y_test), x_test=x_test)
            
            if eval_metric.__name__== 'mean_squared_error':
                if forecast_col == self.target_col[0]:
                    accuracy = eval_metric(y_test, yhat[0], squared=False)
                else:
                    accuracy = eval_metric(y_test, yhat[1], squared=False)
                
            else:
                if forecast_col == self.target_col[0]:
                    accuracy = eval_metric(y_test, yhat[0])
                    
                else:
                    accuracy = eval_metric(y_test, yhat[1])
#                 print(str(accuracy)+" and len is "+str(len(test)))
            metric.append(accuracy)
        score = np.mean(metric)

        print ("SCORE:", score)
        return {'loss':score, 'status':STATUS_OK}
        
    trials = Trials()

    best_hyperparams = fmin(fn = objective,
                    space = param_space,
                    algo = tpe.suggest,
                    max_evals = eval_num,
                    trials = trials)
    # best_params = {i: int(best_hyperparams[i]) if i in ["num_iterations", "num_leaves", "max_depth","min_data_in_leaf", "top_k"] 
    #                    else best_hyperparams[i] for i in best_hyperparams}
    return space_eval(param_space, best_hyperparams)


def cv_tune_bidirectional(
    model,
    df,
    forecast_col,
    cv_split,
    test_size,
    param_space,
    eval_metric,
    step_size=None,
    opt_horizon=None,
    eval_num=100,
    verbose=False,
):
    """
    Tune forecasting model hyperparameters using cross-validation and Bayesian optimization.

    Args:
        model: Forecasting model object with .fit and .forecast methods and relevant attributes.
        df (pd.DataFrame): Time series dataframe.
        forecast_col (str): Name of the target variable to test.
        cv_split (int): Number of time series splits.
        test_size (int): Size of test window for each split.
        step_size (int): Step size for moving the window. Defaults to None (equal to test_size).
        param_space (dict): Hyperopt parameter search space. params for lags, differencing, etc. can be {'n_lag': (hp.choice('lag_y1', [1,2,3]), hp.choice('lag_y2', [1,2]))}
        eval_metric (callable): Evaluation metric function.
        opt_horizon (int, optional): Evaluate only on last N points of each split. Defaults to None (all points).
        eval_num (int, optional): Number of hyperopt evaluations. Defaults to 100.
        verbose (bool, optional): Print progress. Defaults to False.

    Returns:
        dict: Best hyperparameter values found.
    """

    target_cols = model.target_cols
    tscv = ParametricTimeSeriesSplit(n_splits=cv_split, test_size=test_size, step_size=step_size)

    def _set_model_params(params):
        # Handle special model parameters that are not passed to model constructor
        # and must be set directly on the forecasting model object
        if "n_lag" in params:
            if isinstance(params["n_lag"], (tuple, list)):
                for idx, target_col in enumerate(target_cols):
                    model.n_lag[target_col] = params["n_lag"][idx]
        if "difference" in params:
            if isinstance(params["difference"], (tuple, list)):
                # If difference is a tuple or list, set it for each target column
                for idx, target_col in enumerate(target_cols):
                    model.difference[target_col] = params["difference"][idx]

        if "box_cox" in params:
            if isinstance(params["box_cox"], (tuple, list)):
                for idx, target_col in enumerate(target_cols):
                    model.box_cox[target_col] = params["box_cox"][idx]
        if "box_cox_lmda" in params:
            if isinstance(params["box_cox_lmda"], (tuple, list)):
                for idx, target_col in enumerate(target_cols):
                    model.lmda[target_col] = params["box_cox_lmda"][idx]
        if "box_cox_biasadj" in params:
            if isinstance(params["box_cox_biasadj"], (tuple, list)):
                for idx, target_col in enumerate(target_cols):
                    model.biasadj[target_col] = params["box_cox_biasadj"][idx]
        # if "trend" in params:
        #     for target_col in target_cols:
        #         model.trend[target_col] = params["trend"]

        # Handle ETS trend settings
        # ets_params (dict, optional): Dictionary of ETS model parameters (values are tuples of dictionaries of params) and fit settings for each target variable. Example: {'Target1': ({'trend': 'add', 'seasonal': 'add'}, {'damped_trend': True}), 'Target2': ({'trend': 'mul', 'seasonal': 'mul'}, {'damped_trend': False})}.
        if model.trend is not None:
            for idx, target_col in enumerate(target_cols):
                if model.trend[target_col] and model.trend_type[target_col] in ("ses", "feature_ses"):
                    model.ets_params[target_col] = {target_col: [{k: params[k][idx] for k in ["trend", "damped_trend", "seasonal", "seasonal_periods"] if k in params}] for target_col in target_cols} # Set trend and seasonal parameters for each target column
                    model.ets_fit = {}
                    for k in ["smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend"]:
                        if k in params:
                            # Only set "damping_trend" if "damped_trend" is True
                            if (k == "damping_trend") and ("damped_trend" in params and not params["damped_trend"]):
                                continue
                        model.ets_fit[k] = params[k][idx]
                    # append model.ets_fit to model.ets_params[target_col]
                    model.ets_params[target_col].append(model.ets_fit)

    def _get_model_params_for_fit(params):
        # Exclude special parameters that should not be passed to the model constructor
        skip_keys = {
            "box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj",
            "trend", "damped_trend", "seasonal", "seasonal_periods",
            "smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend",
            "difference"
        }
        return {k: v for k, v in params.items() if k not in skip_keys}

    def objective(params):
        _set_model_params(params)
        model_params = _get_model_params_for_fit(params) if getattr(model, "model", None) and model.model.__name__ != "LinearRegression" else None

        metrics = []
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test = test.drop(columns=[target_cols])
            y_test = np.array(test[forecast_col])

            if model_params is not None:
                model.model.set_params(**model_params)
                model.fit(train, model_params)
            else:
                model.fit(train)
            y_pred = model.forecast(n_ahead=len(y_test), x_test=x_test)[forecast_col]

            # Evaluate using the specified metric
            if eval_metric.__name__ == "mean_squared_error":
                score = eval_metric(
                    y_test[-opt_horizon:] if opt_horizon else y_test,
                    y_pred[-opt_horizon:] if opt_horizon else y_pred,
                    squared=False,
                )
            elif eval_metric.__name__ in ("MeanAbsoluteScaledError", "MedianAbsoluteScaledError"):
                score = eval_metric(
                    y_test[-opt_horizon:] if opt_horizon else y_test,
                    y_pred[-opt_horizon:] if opt_horizon else y_pred,
                    np.array(train[model.target_col])
                )
            else:
                score = eval_metric(
                    y_test[-opt_horizon:] if opt_horizon else y_test,
                    y_pred[-opt_horizon:] if opt_horizon else y_pred,
                )
            metrics.append(score)

        mean_score = np.mean(metrics)
        if verbose:
            print("Score:", mean_score)
        return {"loss": mean_score, "status": STATUS_OK}

    trials = Trials()
    best_hyperparams = fmin(
        fn=objective,
        space=param_space,
        algo=tpe.suggest,
        max_evals=eval_num,
        trials=trials,
    )

    model.tuned_params = [
        space_eval(param_space, {k: v[0] for k, v in t["misc"]["vals"].items()})
        for t in trials.trials
    ]

    return space_eval(param_space, best_hyperparams)



def prob_param_forecasts(model, H, train_df, test_df=None):
    prob_forecasts = []
    for params in model.tuned_params:
        if ('n_lag' in params) |('differencing_number' in params)|('box_cox' in params)|('box_cox_lmda' in params)|('box_cox_biasadj' in params):
            if ('n_lag' in params):
                if type(params["n_lag"]) is tuple:
                    model.n_lag = list(params["n_lag"])
                else:
                    model.n_lag = range(1, params["n_lag"]+1)

            if ('differencing_number' in params):
                model.difference = params["differencing_number"]
            if ('box_cox' in params):
                model.box_cox = params["box_cox"]
            if ('box_cox_lmda' in params):
                model.lmda = params["box_cox_lmda"]

            if ('box_cox_biasadj' in params):
                model.biasadj = params["box_cox_biasadj"]

        
        if model.model.__name__ != 'LinearRegression':
            model_params = {k: v for k, v in params.items() if (k not in ["box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj"])}
            model.fit(train_df, model_params)
        else:
            model.fit(train_df)
        if test_df is not None:
            forecasts = model.forecast(H, test_df)
        else:
            forecasts = model.forecast(H)

        prob_forecasts.append(forecasts)
    prob_forecasts = np.row_stack(prob_forecasts)
    prob_forecasts=pd.DataFrame(prob_forecasts)
    prob_forecasts.columns = ["horizon_"+str(i+1) for i in prob_forecasts.columns]
    return prob_forecasts