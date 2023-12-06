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
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
class cat_forecaster:
    def __init__(self, target_col, n_lag, cat_variables = None):
        self.target_col = target_col
        self.cat_var = cat_variables
        self.n_lag = n_lag
        
    def data_prep(self, df):
        dfc = df.copy()
        if self.cat_var is not None:
            for c in self.cat_var:
                dfc[c] = dfc[c].astype('str')
        for i in range(1, self.n_lag+1):
            dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
        dfc = dfc.dropna()
        return dfc
    
    def fit(self, df, param = None):
        if param is not None:
            model_cat = cat.CatBoostRegressor(**param)
        else:
            model_cat = cat.CatBoostRegressor()
        model_df = self.data_prep(df)
        X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        model_cat.fit(X, self.y, cat_features=self.cat_var, verbose = True)
        return model_cat
    
    def forecast(self, model, n_ahead, x_test = None):
        lags = self.y[-self.n_lag:].tolist()
        lags.reverse()
        predictions = []
        for i in range(n_ahead):
            if x_test is not None:
                inp = x_test.iloc[i, 0:].tolist()+lags
            else:
                inp = lags
            pred = model.predict(inp)
            predictions.append(pred)
            lags.insert(0, pred)
            lags = lags[0:self.n_lag]
        return np.array(predictions)
    

    def tune_model(self, df, cv_split, test_size, eval_num = 100, iterations = [80, 3000, 10], depth = [3, 12, 1], 
                learning_rate = [0.0001, 0.3, 0.00001],
                l2_leaf_reg =  [0, 12, 0.0001],
                bagging_temperature = [0, 30, 0.001]
                ):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)

        def objective(params):
            model =cat.CatBoostRegressor(iterations = int(params["iterations"]), 
                                                            learning_rate=params["learning_rate"],
                                                            depth=int(params["depth"]), 
                                                            l2_leaf_reg=params["l2_leaf_reg"],
                                bagging_temperature=params["bagging_temperature"])

            
            mape = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])
                model_train = self.data_prep(train)
                X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                model.fit(X, self.y, cat_features=self.cat_var,
                            verbose = False)
                yhat = self.forecast(model, n_ahead =len(y_test), x_test=x_test)
                accuracy = mean_absolute_percentage_error(y_test, yhat)*100
                mape.append(accuracy)
            score = np.mean(mape)

            print ("SCORE:", score)
            return {'loss':score, 'status':STATUS_OK}

        params={'depth': hp.quniform("depth", depth[0], depth[1], depth[2]),
                        'learning_rate': hp.quniform('learning_rate', learning_rate[0], learning_rate[1], learning_rate[2]),
                        'l2_leaf_reg' : hp.quniform('l2_leaf_reg', l2_leaf_reg[0], l2_leaf_reg[1], l2_leaf_reg[2]),
                        'bagging_temperature' : hp.quniform('bagging_temperature', bagging_temperature[0], bagging_temperature[1], bagging_temperature[2]),
                        'iterations': hp.quniform("iterations", iterations[0], iterations[1], iterations[2])
                    }


        trials = Trials()

        best_hyperparams = fmin(fn = objective,
                        space = params,
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
        for i in range(1, self.n_lag+1):
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
        lags = self.y[-self.n_lag:].tolist()
        lags.reverse()
        predictions = []
        for i in range(n_ahead):
            if x_test is not None:
                inp = x_test.iloc[i, 0:].tolist()+lags
            else:
                inp = lags
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns
            for i in df_inp.columns:
                if i in self.cat_var:
                    df_inp[i] = df_inp[i].astype('category')
                else:
                    df_inp[i] = df_inp[i].astype('float64')
            pred = model.predict(df_inp)[0]
            predictions.append(pred)
            lags.insert(0, pred)
            lags = lags[0:self.n_lag]
        return np.array(predictions)
    
    def tune_model(self, df, cv_split, test_size, eval_num = 100, num_iterations = [50, 2500, 10], learning_rate = [0.001, 0.4, 0.0001],
                  num_leaves=[5, 100, 1], max_depth = [5, 100, 1], bagging_fraction = [0.5, 1, 0.00001],
                  feature_fraction = [0.5, 1, 0.00001], min_data_in_leaf = [10, 50, 1], lambda_l2 = [0,10,0.00001],
                   lambda_l1 = [0, 10, 0.00001], min_gain_to_split = [0, 50, 0.00001], top_rate = [0.05, 0.4, 0.0001],
                  other_rate = [0.05, 0.3, 0.0001], top_k = [10, 40, 1]):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            model =lgb.LGBMRegressor(num_iterations =int(params['num_iterations']),
                                        num_leaves = int(params['num_leaves']),
                                        max_depth = int(params['max_depth']),
                                        min_data_in_leaf = int(params['min_data_in_leaf']),
                                        feature_fraction = params['feature_fraction'],
                                        bagging_fraction = params['bagging_fraction'], lambda_l2 = params['lambda_l2'],
                                     lambda_l1 = params['lambda_l1'],
                             min_gain_to_split = params['min_gain_to_split'], top_rate = params['top_rate'],
                                     other_rate=params['other_rate'], learning_rate = params['learning_rate'],
                                      top_k = int(params["top_k"]))

            mape = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])
                model_train = self.data_prep(train)
                X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                model.fit(X, self.y, categorical_feature=self.cat_var,
                            verbose = True)
                yhat = self.forecast(model, n_ahead =len(y_test), x_test=x_test)
                accuracy = mean_absolute_percentage_error(y_test, yhat)*100
                mape.append(accuracy)
            score = np.mean(mape)

            print ("SCORE:", score)
            return {'loss':score, 'status':STATUS_OK}
            
        params={'learning_rate': hp.quniform('learning_rate', learning_rate[0], learning_rate[1], learning_rate[2]),
                    'num_leaves': hp.quniform('num_leaves', num_leaves[0], num_leaves[1], num_leaves[2]),
                   'max_depth':hp.quniform('max_depth', max_depth[0], max_depth[1], max_depth[2]),
                    'bagging_fraction': hp.quniform('bagging_fraction', 0.5, 1, 0.00001),
                    'feature_fraction': hp.quniform('feature_fraction', 0.5, 1, 0.00001),
                   'min_data_in_leaf': hp.quniform ('min_data_in_leaf', 10, 50, 1), 
                    'lambda_l2' : hp.quniform('lambda_l2', 0,10,0.00001),
                   'lambda_l1' : hp.quniform('lambda_l1', 0, 10, 0.00001),
                    'min_gain_to_split':hp.quniform('min_gain_to_split', 0, 50, 0.00001),
                   'top_rate' : hp.quniform('top_rate', 0.05, 0.4, 0.0001),
                    'other_rate' : hp.quniform('other_rate', 0.05, 0.3, 0.0001),
                   'num_iterations': hp.quniform("num_iterations", 50, 2500, 10),
                   'top_k': hp.quniform('top_k', 10, 40, 1),
                   'seed': 0}
            
            
        trials = Trials()

        best_hyperparams = fmin(fn = objective,
                        space = params,
                        algo = tpe.suggest,
                        max_evals = eval_num,
                        trials = trials)
        best_params = {i: int(best_hyperparams[i]) if i in ["num_iterations", "num_leaves", "max_depth","min_data_in_leaf", "top_k"] 
                           else best_hyperparams[i] for i in best_hyperparams}
        return best_params
    

import xgboost as xgb
class xgboost_forecaster:
    def __init__(self, target_col, n_lag, cat_variables = None):
        self.target_col = target_col
        self.cat_var = cat_variables
        self.n_lag = n_lag
        
    
        
    def data_prep(self, df):
        dfc = df.copy()
        if self.cat_var is not None:
            for c in self.cat_var:
                dfc[c] = dfc[c].astype('category')
        for i in range(1, self.n_lag+1):
            dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
        dfc = dfc.dropna()
        return dfc
    
    def fit(self, df, param = None):
        if param is not None:
            if self.cat_var is not None:
                model_xgb =xgb.XGBRegressor(enable_categorical=True, **param)
            else:
                model_xgb =xgb.XGBRegressor(**param)
        else:
            if self.cat_var is not None:
                model_xgb =xgb.XGBRegressor(enable_categorical=True)
            else:
                model_xgb =xgb.XGBRegressor()
        model_df = self.data_prep(df)
        self.X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        model_xgb.fit(self.X, self.y, verbose = True)
        return model_xgb
    
    def forecast(self, model, n_ahead, x_test = None):
        lags = self.y[-self.n_lag:].tolist()
        lags.reverse()
        predictions = []
        for i in range(n_ahead):
            if x_test is not None:
                inp = x_test.iloc[i, 0:].tolist()+lags
            else:
                inp = lags
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns
            for i in df_inp.columns:
                if i in self.cat_var:
                    df_inp[i] = df_inp[i].astype('category')
                else:
                    df_inp[i] = df_inp[i].astype('float64')
            pred = model.predict(df_inp)[0]
            predictions.append(pred)
            lags.insert(0, pred)
            lags = lags[0:self.n_lag]
        return np.array(predictions)

    
    def tune_model(self, df, cv_split, test_size, eval_num= 100, n_estimators = [50, 2500, 10], max_depth = [3, 15, 1],
                  gamma = [0, 10, 0.0001], reg_lambda = [0, 10, 0.0001],
                  colsample_bytree = [0.5, 1, 0.001], min_child_weight = [0, 9, 0.001],
                  learning_rate = [0.001, 0.4, 0.0001], colsample_bynode = [0.5, 1, 0.0001], reg_alpha = [0,5,0.001]):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            if self.cat_var is not None:
                model =xgb.XGBRegressor(n_estimators =int(params['n_estimators']),
                                     max_depth = int(params['max_depth']), gamma = params['gamma'],
                                     reg_lambda = params['reg_lambda'], colsample_bytree = params['colsample_bytree'],
                                     min_child_weight=int(params['min_child_weight']), learning_rate = params['learning_rate'],
                                      colsample_bynode = params["colsample_bynode"],
                                      reg_alpha = params["reg_alpha"], enable_categorical = True)
            else:
                model =xgb.XGBRegressor(n_estimators =int(params['n_estimators']),
                                     max_depth = int(params['max_depth']), gamma = params['gamma'],
                                     reg_lambda = params['reg_lambda'], colsample_bytree = params['colsample_bytree'],
                                     min_child_weight=int(params['min_child_weight']), learning_rate = params['learning_rate'],
                                      colsample_bynode = params["colsample_bynode"],
                                      reg_alpha = params["reg_alpha"])   
                
            mape = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])
                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                model.fit(self.X, self.y, verbose = True)
                yhat = self.forecast(model, n_ahead =len(y_test), x_test=x_test)
                accuracy = mean_absolute_percentage_error(y_test, yhat)*100
                mape.append(accuracy)
            score = np.mean(mape)

            print ("SCORE:", score)
            return {'loss':score, 'status':STATUS_OK}
        
        params={'max_depth': hp.quniform("max_depth", max_depth[0], max_depth[1], max_depth[2]), 
                   'learning_rate': hp.quniform('learning_rate', learning_rate[0], learning_rate[1], learning_rate[2]),
                   'gamma': hp.quniform ('gamma', 0, 10, 0.0001), 'reg_alpha' : hp.quniform('reg_alpha', reg_alpha[0],reg_alpha[1],reg_alpha[2]),
                   'reg_lambda' : hp.quniform('reg_lambda', reg_lambda[0], reg_lambda[1], reg_lambda[2]),
                   'min_child_weight' : hp.quniform('min_child_weight', min_child_weight[0], min_child_weight[1], min_child_weight[2]),
                   'n_estimators': hp.quniform("n_estimators", n_estimators[0], n_estimators[1], n_estimators[2]),
                   'colsample_bytree': hp.quniform('colsample_bytree', colsample_bytree[0], colsample_bytree[1], colsample_bytree[2]),
                   'colsample_bynode': hp.quniform('colsample_bynode', colsample_bynode[0], colsample_bynode[1], colsample_bynode[2]),
                   'seed': 0}
            
            
        trials = Trials()

        best_hyperparams = fmin(fn = objective,
                        space = params,
                        algo = tpe.suggest,
                        max_evals = eval_num,
                        trials = trials)
        best_params = {i: int(best_hyperparams[i]) if i in ["n_estimators", "max_depth"] 
                           else best_hyperparams[i] for i in best_hyperparams}
        return best_params
            
