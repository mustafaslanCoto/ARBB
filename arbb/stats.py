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
                    start (int): An integer that should be 0 for the training dataset, whereas for the test dataset, it should correspond to the length of the training data.
                    stop (int): An integer representing the length of the training dataset is required for the training dataset, while for the testing dataset, it should be the sum of the lengths of both the training and test datasets.
                    period (int): the seosanal period.
                    num_terms (int): It specifies how many pairs of sin and cos terms to include.
                    df_index: a dataframe (training or test dataset). It specify whether to use the indexes of the training or test dataset for the returned dataframe
    '''
    t = np.arange(start, stop)
    df = pd.DataFrame(index=df_index.index)
    for i in range(1, num_terms + 1):
        df["sin_"+str(i-1)+"_"+str(period)] = np.sin(2 * np.pi * i * t / period)
        df["cos_"+str(i-1)+"_"+str(period)] = np.cos(2 * np.pi * i * t / period)
    return df


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


def tune_ets(data, param_space, cv_splits, horizon, eval_metric, eval_num):
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
            
    
    
        metric = []
        for train_index, test_index in tscv.split(data):
            train, test = data[train_index], data[test_index]
    
            hw_fit = ExponentialSmoothing(train ,seasonal_periods=S , seasonal=season_type, trend=trend_type, damped_trend = damped_bool).fit(smoothing_level = alpha, 
                                                                                                                      smoothing_trend = beta,
                                                                                                                      smoothing_seasonal = gamma,
                                                                                                damping_trend=damp_trend)
            
            hw_forecast = hw_fit.forecast(len(test))
            forecast_filled = np.nan_to_num(hw_forecast, nan=0)
            accuracy = eval_metric(test, forecast_filled)
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

    return space_eval(param_space, best_hyperparams)