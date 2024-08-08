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
def unit_root_test(series, method = "ADF", n_lag = None):
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

@jit(nopython=True)
def rolling_median(arr, window):
    """
    Calculate the rolling median of an array.
    
    Parameters:
    arr (array-like): Input array
    window (int): Size of the rolling window
    
    Returns:
    numpy.ndarray: Array of rolling medians, same length as input array
    """
    # Convert input to numpy array if it's not already
    arr = np.asarray(arr)
    
    # Create output array filled with NaN
    result = np.full(arr.shape, np.nan)
    
    # Calculate rolling median
    for i in range(window - 1, len(arr)):
        result[i] = np.median(arr[i - window + 1 : i + 1])
    
    return result
    
# def rolling_quantile(arr, window, q):
#     """
#     Calculate the rolling quantile of an array.
    
#     Parameters:
#     arr (array-like): Input array
#     q (float): Quantile, which must be between 
#     0 and 1 inclusive
#     window (int): Size of the rolling window
    
#     Returns:
#     numpy.ndarray: Array of rolling quantiles, same length as input array
#     """
#     # Convert input to numpy array if it's not already
#     arr = np.asarray(arr)
    
#     # Create output array filled with NaN
#     result = np.full(arr.shape, np.nan)
    
#     # Calculate rolling quantile
#     for i in range(window - 1, len(arr)):
#         result[i] = np.quantile(arr[i - window + 1 : i + 1], q, method="lower")
#         # result[i] = stats.mstats.mquantiles(arr[i - window + 1 : i + 1], prob=q, alphap=0.5, betap=0.5)[0]
    
#     return result

@jit(nopython=True)
def rolling_quantile(arr, window, q):
    """
    Calculate the rolling quantile of an array.
    
    Parameters:
    arr (array-like): Input array
    q (float): Quantile, which must be between 0 and 1 inclusive
    window (int): Size of the rolling window
    
    Returns:
    numpy.ndarray: Array of rolling quantiles, same length as input array
    """
    
    result = np.full(arr.shape, np.nan)
    
    for i in range(window - 1, len(arr)):
        x = arr[i - window + 1 : i + 1]
        n = len(x)
        y = np.sort(x)
        result[i] = (y[int(q * (n - 1))] + y[int(np.ceil(q * (n - 1)))]) * 0.5
    
    return result

def target_power(series, p):
    return np.array(series)**p



def Kfold_target(train, test, cat_var, target_col, encoded_colname, split_num):
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


def box_cox_transform(x, shift=False, box_cox_lmda = None):
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

def undiff_ts(original_data, differenced_data, difference_number):
    
    undiff_data = np.array(differenced_data)
    if difference_number>1:
        for i in range(difference_number-1, 0, -1):
            undiff_data = np.diff(original_data, i)[-1]+np.cumsum(undiff_data)
    
    return original_data[-1]+np.cumsum(undiff_data)

def seasonal_diff(data, seasonal_length):
    orig_data = list(np.repeat(np.nan, seasonal_length))+[data[i] - data[i - seasonal_length] for i in range(seasonal_length, len(data))]
    return np.array(orig_data)

# invert difference
def invert_seasonal_diff(orig_data, diff_data, seasonal_length):
    conc_data = list(orig_data[-seasonal_length:]) + list(diff_data)
    for i in range(len(conc_data)-seasonal_length):
        conc_data[i+seasonal_length] = conc_data[i]+conc_data[i+seasonal_length]
        
    return np.array(conc_data[-len(diff_data):])


def tune_ets(data, param_space, cv_splits, horizon, eval_metric, eval_num, append_horizons = False, verbose = False):
    from sklearn.model_selection import TimeSeriesSplit
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
    from hyperopt.pyll import scope
    tscv = TimeSeriesSplit(n_splits=cv_splits, test_size=horizon)
    
    def objective(params):
        if (params["trend"] != None) & (params["seasonal"] != None):
            alpha = params['smoothing_level']
            beta = params['smoothing_trend']
            gamma = params['smoothing_seasonal']
            trend_type = params['trend']
            season_type = params['seasonal']
            S = params['seasonal_periods']
            if params["damped_trend"] == True:
                damped_bool = params["damped_trend"]
                damp_trend = params['damping_trend']
            else:
                damped_bool = params["damped_trend"]
                damp_trend = None
    
        elif (params["trend"] != None) & (params["seasonal"] == None):
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
                
        elif (params["trend"] == None) & (params["seasonal"] != None):
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
                
        else:
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
        metric = []
        for train_index, test_index in tscv.split(data):
            train, test = data[train_index], data[test_index]
    
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

def tune_sarima(y, d, D, season,p_range, q_range, P_range, Q_range, X=None):
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

def regression_detrend(series):
    model = LinearRegression().fit(np.array(range(len(series))).reshape(-1, 1),series)
# Make predictions
    y_pred = model.predict(np.array(range(len(series))).reshape(-1, 1))
    return np.array(series)-y_pred

def forecast_trend(train_series, H):
    model = LinearRegression().fit(np.array(range(len(train_series))).reshape(-1, 1),train_series)
    return model.predict(np.array(range(len(train_series), len(train_series)+H)).reshape(-1, 1))