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

## Catboost
import catboost as cat
class cat_forecaster:
    def __init__(self, target_col, n_lag, cat_variables = None):
        self.target_col = target_col
        self.cat_var = cat_variables
        self.n_lag = n_lag
        
    def cat_data_prep(self, df):
        dfc = df.copy()
        if self.cat_var is not None:
            for c in self.cat_var:
                dfc[c] = dfc[c].astype('category')
        for i in range(1, self.n_lag+1):
            dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
        dfc = dfc.dropna()
        return dfc
    
    def fit_cat(self, df, param = None):
        if param is not None:
            model_cat = cat.CatBoostRegressor(**param)
        else:
            model_cat = cat.CatBoostRegressor()
        model_df = self.cat_data_prep(df)
        X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        model_cat.fit(X, self.y, cat_features=self.cat_var, verbose = True)
        self.model_cat = model_cat
    
    def cat_forecast(self, n_ahead, x_test = None):
        lags = self.y[-self.n_lag:].tolist()
        lags.reverse()
        predictions = []
        for i in range(n_ahead):
            if x_test is not None:
                inp = x_test.iloc[i, 0:].tolist()+lags
            else:
                inp = lags
            pred = self.model_cat.predict(inp)
            predictions.append(pred)
            lags.insert(0, pred)
            lags = lags[0:self.n_lag]
        return np.array(predictions)
