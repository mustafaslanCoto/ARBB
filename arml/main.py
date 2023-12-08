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
from sklearn.model_selection import TimeSeriesSplit
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
class cat_forecaster:
    def __init__(self, target_col, n_lag, cat_variables = None):
        self.target_col = target_col
        self.cat_var = cat_variables
        self.n_lag = n_lag
        
    def cat_data_prep(self, df):
        dfc = df.copy()
        if self.cat_var is not None:
            for c in self.cat_var:
                dfc[c] = dfc[c].astype('str')
        for i in self.n_lag:
            dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
        dfc = dfc.dropna()
        return dfc
    
    def fit(self, df, param = None):
        if param is not None:
            model_cat = cat.CatBoostRegressor(**param)
        else:
            model_cat = cat.CatBoostRegressor()
        model_df = self.cat_data_prep(df)
        X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        model_cat.fit(X, self.y, cat_features=self.cat_var, verbose = True)
        return model_cat
    
    def forecast(self, model, n_ahead, x_test = None):
        max_lag = self.n_lag[-1]
        lags = self.y[-max_lag:].tolist()
        predictions = []
        for i in range(n_ahead):
            inp_lag = [lags[-i] for i in self.n_lag]
            if x_test is not None:
                inp = x_test.iloc[i, 0:].tolist()+inp_lag
            else:
                inp = inp_lag
            pred = model.predict(inp)
            predictions.append(pred)
            lags.append(pred)
            lags = lags[-max_lag:]
        return np.array(predictions)
    

    def tune_model(self, df, cv_split, test_size, param_space, eval_num = 100):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)

        def objective(params):
            model =cat.CatBoostRegressor(**params)

            
            mape = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])
                model_train = self.cat_data_prep(train)
                X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                model.fit(X, self.y, cat_features=self.cat_var,
                            verbose = False)
                yhat = self.forecast(model, n_ahead =len(y_test), x_test=x_test)
                accuracy = mean_absolute_percentage_error(y_test, yhat)*100
#                 print(str(accuracy)+" and len is "+str(len(test)))
                mape.append(accuracy)
            score = np.mean(mape)

            print ("SCORE:", score)
            return {'loss':score, 'status':STATUS_OK}


        trials = Trials()

        best_hyperparams = fmin(fn = objective,
                        space = param_space,
                        algo = tpe.suggest,
                        max_evals = eval_num,
                        trials = trials)
        best_params = {i: int(best_hyperparams[i]) if i in ['depth', 'iterations'] else best_hyperparams[i] for i in best_hyperparams}
        return best_params
    
import lightgbm as lgb
class lightGBM_forecaster:
    def __init__(self, target_col, n_lag, cat_variables = None):
        self.target_col = target_col
        self.cat_var = cat_variables
        self.n_lag = n_lag
        
    def data_prep(self, df):
        dfc = df.copy()
        if self.cat_var is not None:
            for c in self.cat_var:
                dfc[c] = dfc[c].astype('category')
        for i in self.n_lag:
            dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
        dfc = dfc.dropna()
        return dfc
    
    def fit(self, df, param = None):
        if param is not None:
            model_lgb =lgb.LGBMRegressor(**param)
        else:
            model_lgb =lgb.LGBMRegressor()
        model_df = self.data_prep(df)
        self.X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        model_lgb.fit(self.X, self.y, categorical_feature=self.cat_var, verbose = True)
        return model_lgb
    
    def forecast(self, model, n_ahead, x_test = None):
        max_lag = self.n_lag[-1]
        lags = self.y[-max_lag:].tolist()
        predictions = []
        for i in range(n_ahead):
            inp_lag = [lags[-i] for i in self.n_lag]
            if x_test is not None:
                inp = x_test.iloc[i, 0:].tolist()+inp_lag
            else:
                inp = inp_lag
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns
            for i in df_inp.columns:
                if i in self.cat_var:
                    df_inp[i] = df_inp[i].astype('category')
                else:
                    df_inp[i] = df_inp[i].astype('float64')
            pred = model.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)
            lags = lags[-max_lag:]
        return np.array(predictions)
    
    def tune_model(self, df, cv_split, test_size, param_space, eval_num = 100):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            model =lgb.LGBMRegressor(**params)

            mape = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])
                model_train = self.data_prep(train)
                X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                model.fit(X, self.y, categorical_feature=self.cat_var,
                            verbose = False)
                yhat = self.forecast(model, n_ahead =len(y_test), x_test=x_test)
                accuracy = mean_absolute_percentage_error(y_test, yhat)*100
#                 print(str(accuracy)+" and len is "+str(len(test)))
                mape.append(accuracy)
            score = np.mean(mape)

            print ("SCORE:", score)
            return {'loss':score, 'status':STATUS_OK}
            
        trials = Trials()

        best_hyperparams = fmin(fn = objective,
                        space = param_space,
                        algo = tpe.suggest,
                        max_evals = eval_num,
                        trials = trials)
        best_params = {i: int(best_hyperparams[i]) if i in ["num_iterations", "num_leaves", "max_depth","min_data_in_leaf", "top_k"] 
                           else best_hyperparams[i] for i in best_hyperparams}
        return best_params
            
    

import xgboost as xgb
class xgboost_forecaster:
    def __init__(self, target_col, n_lag, cat_dict = None):
        self.target_col = target_col
        self.cat_var = cat_dict
        self.n_lag = n_lag
        
    
        
    def data_prep(self, df):
        dfc = df.copy()
        if self.cat_var is not None:
            for col, cat in self.cat_var.items():
                dfc[col] = dfc[col].astype('category')
                dfc[col] = dfc[col].cat.set_categories(cat)
            dfc = pd.get_dummies(dfc)
        if self.target_col in dfc.columns:
            for i in self.n_lag:
                dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
        dfc = dfc.dropna()
        return dfc
    
    def fit(self, df, param = None):
        if param is not None:
            model_xgb =xgb.XGBRegressor(**param)
        else:
            model_xgb =xgb.XGBRegressor()
        model_df = self.data_prep(df)
        self.X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        model_xgb.fit(self.X, self.y, verbose = True)
        return model_xgb
    
    def forecast(self, model, n_ahead, x_test = None):
        x_test = self.data_prep(x_test)
        max_lag = self.n_lag[-1]
        lags = self.y[-max_lag:].tolist()    
        predictions = []
        for i in range(n_ahead):
            if x_test is not None: 
                inp_lag = [lags[-i] for i in self.n_lag]
                inp = x_test.iloc[i, 0:].tolist()+inp_lag
            else:
                inp = [lags[-i] for i in self.n_lag]
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns

            pred = model.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)
            lags = lags[-max_lag:]
        return np.array(predictions)


    
    def tune_model(self, df, cv_split, test_size, param_space, eval_num= 100):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            model =xgb.XGBRegressor(**params)   
                
            mape = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])
                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                model.fit(self.X, self.y, verbose = True)
                yhat = self.forecast(model, n_ahead =len(y_test), x_test=x_test)
                accuracy = mean_absolute_percentage_error(y_test, yhat)*100
#                 print(str(accuracy)+" and len is "+str(len(test)))
                mape.append(accuracy)
            score = np.mean(mape)

            print ("SCORE:", score)
            return {'loss':score, 'status':STATUS_OK}
            
            
        trials = Trials()

        best_hyperparams = fmin(fn = objective,
                        space = param_space,
                        algo = tpe.suggest,
                        max_evals = eval_num,
                        trials = trials)
        best_params = {i: int(best_hyperparams[i]) if i in ["n_estimators", "max_depth"] 
                           else best_hyperparams[i] for i in best_hyperparams}
        return best_params