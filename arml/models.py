import pandas as pd
import numpy as np


## Catboost
from window_ops.rolling import rolling_mean, rolling_max, rolling_min, rolling_std
import catboost as cat
from sklearn.model_selection import TimeSeriesSplit
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

class cat_forecaster:
    def __init__(self, target_col, n_lag = None, lag_transform = None, cat_variables = None):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.target_col = target_col
        self.cat_var = cat_variables
        self.n_lag = n_lag
        self.lag_transform = lag_transform
        
    def data_prep(self, df):
        dfc = df.copy()
        if self.cat_var is not None:
            for c in self.cat_var:
                dfc[c] = dfc[c].astype('str')
        if self.n_lag is not None:
            for i in self.n_lag:
                dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
            
        if self.lag_transform is not None:
            df_array = np.array(dfc[self.target_col])
            for i, j in self.lag_transform.items():
                dfc[i.__name__+"_"+str(j)] = i(df_array, j) 
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
        lags = self.y.tolist()
        predictions = []
        for i in range(n_ahead):
            if x_test is not None:
                x_var = x_test.iloc[i, 0:].tolist()
            else:
                x_var = []
            if self.n_lag is not None:
                inp_lag = [lags[-i] for i in self.n_lag]
            else:
                inp_lag = []
                
            lag_array = np.array(lags) # array is needed for transformation fuctions
            if self.lag_transform is not None:
                transform_lag = []
                for method, lag in self.lag_transform.items():
                    tl = method(lag_array, lag)[-1]
                    transform_lag.append(tl)
            else:
                transform_lag = []
            inp = x_var + inp_lag+transform_lag
            pred = model.predict(inp)
            predictions.append(pred)
            lags.append(pred)
        return np.array(predictions)
    

    def tune_model(self, df, cv_split, test_size, param_space, eval_metric, eval_num = 100):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)

        def objective(params):
            model =cat.CatBoostRegressor(**params)

            
            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])
                model_train = self.cat_data_prep(train)
                X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                model.fit(X, self.y, cat_features=self.cat_var,
                            verbose = False)
                yhat = self.forecast(model, n_ahead =len(y_test), x_test=x_test)
                accuracy = eval_metric(y_test, yhat)
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
        # best_params = {i: int(best_hyperparams[i]) if i in ['depth', 'iterations'] else best_hyperparams[i] for i in best_hyperparams}
        return space_eval(param_space, best_hyperparams)
    
import lightgbm as lgb
class lightGBM_forecaster:
    def __init__(self, target_col, n_lag = None, lag_transform = None, cat_variables = None):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.target_col = target_col
        self.cat_var = cat_variables
        self.n_lag = n_lag
        self.lag_transform = lag_transform
        
    def data_prep(self, df):
        dfc = df.copy()
        if self.cat_var is not None:
            for c in self.cat_var:
                dfc[c] = dfc[c].astype('category')
                
        if self.n_lag is not None:
            for i in self.n_lag:
                dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
            
        if self.lag_transform is not None:
            df_array = np.array(dfc[self.target_col])
            for i, j in self.lag_transform.items():
                dfc[i.__name__+"_"+str(j)] = i(df_array, j)   
            
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
        lags = self.y.tolist()
        predictions = []
        for i in range(n_ahead):
            if x_test is not None:
                x_var = x_test.iloc[i, 0:].tolist()
            else:
                x_var = []
            if self.n_lag is not None:
                inp_lag = [lags[-i] for i in self.n_lag]
            else:
                inp_lag = []
                
            lag_array = np.array(lags) # array is needed for transformation fuctions
            if self.lag_transform is not None:
                transform_lag = []
                for method, lag in self.lag_transform.items():
                    tl = method(lag_array, lag)[-1]
                    transform_lag.append(tl)
            else:
                transform_lag = []
            inp = x_var + inp_lag+transform_lag
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns
            for i in df_inp.columns:
                if self.cat_var is not None:
                    if i in self.cat_var:
                        df_inp[i] = df_inp[i].astype('category')
                    else:
                        df_inp[i] = df_inp[i].astype('float64')
                else:
                    df_inp[i] = df_inp[i].astype('float64')
            pred = model.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)
        return np.array(predictions)
    
    def tune_model(self, df, cv_split, test_size, param_space,eval_metric, eval_num = 100):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            model =lgb.LGBMRegressor(**params)

            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])
                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                model.fit(self.X, self.y, categorical_feature=self.cat_var,
                            verbose = False)
                yhat = self.forecast(model, n_ahead =len(y_test), x_test=x_test)
                accuracy = eval_metric(y_test, yhat)
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
            
    

import xgboost as xgb
class xgboost_forecaster:
    def __init__(self, target_col, n_lag = None, lag_transform = None, cat_dict = None, drop_categ = None):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.target_col = target_col
        self.cat_var = cat_dict
        self.n_lag = n_lag
        self.lag_transform = lag_transform
        self.drop_categ = drop_categ
    
        
    def data_prep(self, df):
        dfc = df.copy()
        if self.cat_var is not None:
            for col, cat in self.cat_var.items():
                dfc[col] = dfc[col].astype('category')
                dfc[col] = dfc[col].cat.set_categories(cat)
            dfc = pd.get_dummies(dfc)
        if self.drop_categ is not None:
            for i in self.drop_categ:
                dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)
                
        if self.target_col in dfc.columns:
            if self.n_lag is not None:
                for i in self.n_lag:
                    dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
                
            if self.lag_transform is not None:
                df_array = np.array(dfc[self.target_col])
                for i, j in self.lag_transform.items():
                    dfc[i.__name__+"_"+str(j)] = i(df_array, j)    
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
        x_dummy = self.data_prep(x_test)
#         max_lag = self.n_lag[-1]
#         lags = self.y[-max_lag:].tolist()
        lags = self.y.tolist()
        predictions = []
        for i in range(n_ahead):
            if x_test is not None:
                x_var = x_dummy.iloc[i, 0:].tolist()
            else:
                x_var = []
                
            if self.n_lag is not None:
                inp_lag = [lags[-l] for l in self.n_lag] # to get defined lagged variables 
            else:
                inp_lag = []
            if self.lag_transform is not None:
                lag_array = np.array(lags) # array is needed for transformation fuctions
                transform_lag = []
                for method, lag in self.lag_transform.items():
                    tl = method(lag_array, lag)[-1]
                    transform_lag.append(tl)
            else:
                transform_lag = []
                    
                    
            inp = x_var+inp_lag+transform_lag
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns

            pred = model.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)
#             lags = lags[-max_lag:]
        return np.array(predictions)

    
    def tune_model(self, df, cv_split, test_size, param_space, eval_metric, eval_num= 100):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            model =xgb.XGBRegressor(**params)   
                
            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])
                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                model.fit(self.X, self.y, verbose = True)
                yhat = self.forecast(model, n_ahead =len(y_test), x_test=x_test)
                accuracy = eval_metric(y_test, yhat)
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
        # best_params = {i: int(best_hyperparams[i]) if i in ["n_estimators", "max_depth"] 
        #                    else best_hyperparams[i] for i in best_hyperparams}
        return space_eval(param_space, best_hyperparams)
    
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
class RandomForest_forecaster:
    def __init__(self, target_col, n_lag, lag_transform = None, cat_dict = None, drop_categ = None):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.target_col = target_col
        self.cat_var = cat_dict
        self.n_lag = n_lag
        self.lag_transform = lag_transform
        self.drop_categ = drop_categ
        
    
        
    def data_prep(self, df):
        dfc = df.copy()
        if self.cat_var is not None:
            for col, cat in self.cat_var.items():
                dfc[col] = dfc[col].astype('category')
                dfc[col] = dfc[col].cat.set_categories(cat)
            dfc = pd.get_dummies(dfc)
        if self.drop_categ is not None:
            for i in self.drop_categ:
                dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)
        if self.target_col in dfc.columns:
            if self.n_lag is not None:
                for i in self.n_lag:
                    dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
                
            if self.lag_transform is not None:
                df_array = np.array(dfc[self.target_col])
                for i, j in self.lag_transform.items():
                    dfc[i.__name__+"_"+str(j)] = i(df_array, j)    
        dfc = dfc.dropna()
        return dfc

    def fit(self, df, param = None):
        if param is not None:
            model_rf =RandomForestRegressor(**param)
        else:
            model_rf =RandomForestRegressor()
        model_df = self.data_prep(df)
        self.X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        model_rf.fit(self.X, self.y)
        return model_rf

    def forecast(self, model, n_ahead, x_test = None):
        x_dummy = self.data_prep(x_test)
#         max_lag = self.n_lag[-1]
#         lags = self.y[-max_lag:].tolist()
        lags = self.y.tolist()
        predictions = []
        for i in range(n_ahead):
            if x_test is not None:
                x_var = x_dummy.iloc[i, 0:].tolist()
            else:
                x_var = []
                
            if self.n_lag is not None:
                inp_lag = [lags[-l] for l in self.n_lag] # to get defined lagged variables 
            else:
                inp_lag = []
            if self.lag_transform is not None:
                lag_array = np.array(lags) # array is needed for transformation fuctions
                transform_lag = []
                for method, lag in self.lag_transform.items():
                    tl = method(lag_array, lag)[-1]
                    transform_lag.append(tl)
            else:
                transform_lag = []
                    
                    
            inp = x_var+inp_lag+transform_lag
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns

            pred = model.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)
#             lags = lags[-max_lag:]
        return np.array(predictions)

    
    def tune_model(self, df, cv_split, test_size, param_space, eval_metric, eval_num= 100):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            model =RandomForestRegressor(**params)   
                
            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])
                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                model.fit(self.X, self.y)
                yhat = self.forecast(model, n_ahead =len(y_test), x_test=x_test)
                accuracy = eval_metric(y_test, yhat)
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
        # best_params = {i: int(best_hyperparams[i]) if i in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"] 
        #                    else best_hyperparams[i] for i in best_hyperparams}
        return space_eval(param_space, best_hyperparams)
    
class AdaBoost_forecaster:
    def __init__(self, target_col, n_lag, lag_transform = None, cat_dict = None, drop_categ = None):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.target_col = target_col
        self.cat_var = cat_dict
        self.n_lag = n_lag
        self.lag_transform = lag_transform
        self.drop_categ = drop_categ
        
    
        
    def data_prep(self, df):
        dfc = df.copy()
        if self.cat_var is not None:
            for col, cat in self.cat_var.items():
                dfc[col] = dfc[col].astype('category')
                dfc[col] = dfc[col].cat.set_categories(cat)
            dfc = pd.get_dummies(dfc)
        if self.drop_categ is not None:
            for i in self.drop_categ:
                dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)
        if self.target_col in dfc.columns:
            if self.n_lag is not None:
                for i in self.n_lag:
                    dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
                
            if self.lag_transform is not None:
                df_array = np.array(dfc[self.target_col])
                for i, j in self.lag_transform.items():
                    dfc[i.__name__+"_"+str(j)] = i(df_array, j)    
        dfc = dfc.dropna()
        return dfc

    def fit(self, df, param = None):
        if param is not None:
            model_ada =AdaBoostRegressor(**param)
        else:
            model_ada =AdaBoostRegressor()
        model_df = self.data_prep(df)
        self.X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        model_ada.fit(self.X, self.y)
        return model_ada

    def forecast(self, model, n_ahead, x_test = None):
        x_dummy = self.data_prep(x_test)
#         max_lag = self.n_lag[-1]
#         lags = self.y[-max_lag:].tolist()
        lags = self.y.tolist()
        predictions = []
        for i in range(n_ahead):
            if x_test is not None:
                x_var = x_dummy.iloc[i, 0:].tolist()
            else:
                x_var = []
                
            if self.n_lag is not None:
                inp_lag = [lags[-l] for l in self.n_lag] # to get defined lagged variables 
            else:
                inp_lag = []
            if self.lag_transform is not None:
                lag_array = np.array(lags) # array is needed for transformation fuctions
                transform_lag = []
                for method, lag in self.lag_transform.items():
                    tl = method(lag_array, lag)[-1]
                    transform_lag.append(tl)
            else:
                transform_lag = []
                    
                    
            inp = x_var+inp_lag+transform_lag
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns

            pred = model.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)
#             lags = lags[-max_lag:]
        return np.array(predictions)

    
    def tune_model(self, df, cv_split, test_size, param_space, eval_metric, eval_num= 100):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            model =AdaBoostRegressor(**params)   
                
            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])
                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                model.fit(self.X, self.y)
                yhat = self.forecast(model, n_ahead =len(y_test), x_test=x_test)
                accuracy = eval_metric(y_test, yhat)*100
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
#         best_params = {i: int(best_hyperparams[i]) if i in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"] 
#                            else best_hyperparams[i] for i in best_hyperparams}
        return space_eval(param_space, best_hyperparams)
            