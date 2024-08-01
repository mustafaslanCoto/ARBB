import pandas as pd
import numpy as np
from numba import jit

from sklearn.model_selection import TimeSeriesSplit
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from cubist import Cubist
from sklearn.linear_model import LinearRegression
from arbb.stats import box_cox_transform, back_box_cox_transform
from datetime import timedelta

class cat_forecaster:
    def __init__(self, target_col, add_trend = False, trend_type ="component", n_lag = None, lag_transform = None, differencing_number = None, cat_variables = None,
                 box_cox = False, box_cox_lmda = None, box_cox_biasadj= False):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = CatBoostRegressor
        self.target_col = target_col
        self.cat_var = cat_variables
        self.n_lag = n_lag
        self.difference = differencing_number
        self.lag_transform = lag_transform
        self.trend = add_trend
        self.trend_type = trend_type
        self.box_cox = box_cox
        self.lmda = box_cox_lmda
        self.biasadj = box_cox_biasadj

    def data_prep(self, df):
        dfc = df.copy()
        if self.box_cox == True:
            self.is_zero = np.any(np.array(dfc[self.target_col]) < 1)
            trans_data, self.lmda = box_cox_transform(x = dfc[self.target_col], shift = self.is_zero, box_cox_lmda=self.lmda)
            dfc[self.target_col] = trans_data


        if (self.trend ==True):
            self.len = len(dfc)
            self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
            
            if (self.trend_type == "component"):
                dfc[self.target_col] = dfc[self.target_col]-self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))

        if self.difference is not None:
            if self.difference >1:
                self.last_train = df[self.target_col].tolist()[-self.difference:]
            else:
                self.last_train = df[self.target_col].tolist()[-1]
            dfc[self.target_col] = dfc[self.target_col].diff(self.difference)

        if self.cat_var is not None:
            for c in self.cat_var:
                dfc[c] = dfc[c].astype('str')
        if self.n_lag is not None:
            for i in self.n_lag:
                dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
            
        if self.lag_transform is not None:
            for n, k in self.lag_transform.items():
                df_array = np.array(dfc[self.target_col].shift(n))
                for i in k:
                    if i[0].__name__ == "rolling_quantile":
                        dfc["q_"+str(i[2])+"_"+str(n)+"_"+str(i[1])] = i[0](df_array, i[1], i[2])
                    else:
                        dfc[i[0].__name__+"_"+str(n)+"_"+str(i[1])] = i[0](df_array, i[1]) 

        if (self.trend ==True) & (self.trend_type == "feature"):
            # self.len = len(dfc)
            # self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
            if (self.target_col in dfc.columns):
                dfc["trend"] = self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
        dfc = dfc.dropna()
        # if self.target_col in dfc.columns:
        #     self.dfc = dfc
        #     self.df =df.loc[dfc.index]
        # else:
        return dfc
    
    def fit(self, df, param = None):
        if param is not None:
            model_cat = self.model(**param)
        else:
            model_cat = self.model()

        model_df = self.data_prep(df)
        self.X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        # self.data_prep(df)
        # self.X, self.y = self.dfc.drop(columns =self.target_col), self.dfc[self.target_col]
        self.model_cat = model_cat.fit(self.X, self.y, cat_features=self.cat_var, verbose = True)
    
    def forecast(self, n_ahead, x_test = None):
        lags = self.y.tolist()
        predictions = []

        if self.trend ==True:
            trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))

        for i in range(n_ahead):
            if x_test is not None:
                x_var = x_test.iloc[i, 0:].tolist()
            else:
                x_var = []
            if self.n_lag is not None:
                inp_lag = [lags[-j] for j in self.n_lag]
            else:
                inp_lag = []
                
            if self.lag_transform is not None:
                transform_lag = []    
                for n, k in self.lag_transform.items():
                    df_array = np.array(pd.Series(lags).shift(n-1))
                    for f in k:
                        if f[0].__name__ == "rolling_quantile":
                            t1 = f[0](df_array, f[1], f[2])[-1]
                        else:
                            t1 = f[0](df_array, f[1])[-1]
                        transform_lag.append(t1)
            else:
                transform_lag = []

            if (self.trend ==True) & (self.trend_type == "feature"):
                trend_var = [trend_pred[i]]
            else:
                trend_var = []

            inp = x_var + inp_lag+transform_lag+trend_var
            pred = self.model_cat.predict(inp)
            predictions.append(pred)
            lags.append(pred)

        if self.difference is not None:
            if self.difference>1:
                predictions_ = self.last_train+predictions
                for i in range(len(predictions_)):
                    if i<len(predictions_)-self.difference:
                        predictions_[i+self.difference] = predictions_[i]+predictions_[i+self.difference]
                        forecasts = predictions_[-n_ahead:]
            else:    
                predictions.insert(0, self.last_train)
                forecasts = np.cumsum(predictions)[-n_ahead:]
        else:
            forecasts = np.array(predictions)
        if (self.trend ==True)&(self.trend_type =="component"):
            forecasts = trend_pred+forecasts
        forecasts = np.array([max(0, x) for x in forecasts])

        if self.box_cox == True:
            forecasts = back_box_cox_transform(y_pred = forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)
        return forecasts
    
    def cv(self, df, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {i.__name__:[] for i in metrics}

        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})

    def tune_model(self, df, cv_split, test_size, param_space, eval_metric, opt_horizon = None, eval_num = 100):

        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            if ('n_lag' in params)|('box_cox' in params)|('box_cox_lmda' in params)|('box_cox_biasadj' in params):
                if ('n_lag' in params):
                    if type(params["n_lag"]) is tuple:
                        self.n_lag = list(params["n_lag"])
                    else:
                        self.n_lag = range(1, params["n_lag"]+1)
                if ('box_cox' in params):
                    self.box_cox = params["box_cox"]
                if ('box_cox_lmda' in params):
                    self.lmda = params["box_cox_lmda"]

                if ('box_cox_biasadj' in params):
                    self.biasadj = params["box_cox_biasadj"]

                # self.data_prep(df)
                model =self.model(**{k: v for k, v in params.items() if (k not in ["box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj"])})
            else:
                model =self.model(**params)  
            # model =self.model(**params)  

            
            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])

                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                # train_dfc = self.dfc.iloc[train_index]
                # self.X, self.y = train_dfc.drop(columns =self.target_col), train_dfc[self.target_col]
                self.model_cat = model.fit(self.X, self.y, cat_features=self.cat_var,
                            verbose = False)
                yhat = self.forecast(n_ahead =len(y_test), x_test=x_test)
                if eval_metric.__name__== 'mean_squared_error':
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:], squared=False)
                    else:
                        accuracy = eval_metric(y_test, yhat, squared=False)

                elif (eval_metric.__name__== 'MeanAbsoluteScaledError')|(eval_metric.__name__== 'MedianAbsoluteScaledError'):
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:], np.array(train[self.target_col]))
                    else:
                        accuracy = eval_metric(y_test, yhat, np.array(train[self.target_col]))
                else:
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:])
                    else:
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

class lightGBM_forecaster:
    def __init__(self, target_col, add_trend = False, trend_type ="component", n_lag = None, lag_transform = None,differencing_number = None, cat_variables = None,
                 box_cox = False, box_cox_lmda = None, box_cox_biasadj= False):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = LGBMRegressor
        self.target_col = target_col
        self.cat_var = cat_variables
        self.n_lag = n_lag
        self.difference = differencing_number
        self.lag_transform = lag_transform
        self.trend = add_trend
        self.trend_type = trend_type
        self.box_cox = box_cox
        self.lmda = box_cox_lmda
        self.biasadj = box_cox_biasadj
        
    def data_prep(self, df):
        dfc = df.copy()

        if self.box_cox == True:
            self.is_zero = np.any(np.array(dfc[self.target_col]) < 1)
            trans_data, self.lmda = box_cox_transform(x = dfc[self.target_col], shift = self.is_zero, box_cox_lmda=self.lmda)
            dfc[self.target_col] = trans_data

        if (self.trend ==True):
            self.len = len(dfc)
            self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
            
            if (self.trend_type == "component"):
                dfc[self.target_col] = dfc[self.target_col]-self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))

        if self.difference is not None:
            if self.difference > 1:
                self.last_train = dfc[self.target_col].tolist()[-self.difference:]
            else:
                self.last_train = dfc[self.target_col].tolist()[-1]
            dfc[self.target_col] = dfc[self.target_col].diff(self.difference)
        if self.cat_var is not None:
            for c in self.cat_var:
                dfc[c] = dfc[c].astype('category')
                
        if self.n_lag is not None:
            for i in self.n_lag:
                dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
            
        if self.lag_transform is not None:
            for n, k in self.lag_transform.items():
                df_array = np.array(dfc[self.target_col].shift(n))
                for f in k:
                    if f[0].__name__ == "rolling_quantile":
                        dfc["q_"+str(f[2])+"_"+str(n)+"_"+str(f[1])] = f[0](df_array, f[1], f[2])
                    else:
                        dfc[f[0].__name__+"_"+str(n)+"_"+str(f[1])] = f[0](df_array, f[1])

        if (self.trend ==True) & (self.trend_type == "feature"):
            # self.len = len(dfc)
            # self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
            if (self.target_col in dfc.columns):
                dfc["trend"] = self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
            
        dfc = dfc.dropna()

        return dfc
    
    def fit(self, df, param = None):

        if param is not None:
            model_lgb = self.model(**param, verbose=-1)
        else:
            model_lgb = self.model(verbose=-1)

        model_df = self.data_prep(df)
        self.X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        # self.data_prep(df)
        # self.X, self.y = self.dfc.drop(columns =self.target_col), self.dfc[self.target_col]
        self.model_lgb = model_lgb.fit(self.X, self.y, categorical_feature=self.cat_var)
    
    def forecast(self, n_ahead, x_test = None):
        lags = self.y.tolist()
        predictions = []

        if self.trend ==True:
            trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))

        for i in range(n_ahead):
            if x_test is not None:
                x_var = x_test.iloc[i, 0:].tolist()
            else:
                x_var = []
            if self.n_lag is not None:
                inp_lag = [lags[-i] for i in self.n_lag]
            else:
                inp_lag = []
                
            if self.lag_transform is not None:
                transform_lag = []    
                for n, k in self.lag_transform.items():
                    df_array = np.array(pd.Series(lags).shift(n-1))
                    for f in k:
                        if f[0].__name__ == "rolling_quantile":
                            t1 = f[0](df_array, f[1], f[2])[-1]
                        else:
                            t1 = f[0](df_array, f[1])[-1]
                        transform_lag.append(t1)
            else:
                transform_lag = []

            if (self.trend ==True) & (self.trend_type == "feature"):
                trend_var = [trend_pred[i]]
            else:
                trend_var = []
                    
                    
            inp = x_var+inp_lag+transform_lag+trend_var

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
            pred = self.model_lgb.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)
        if self.difference is not None:
            if self.difference>1:
                predictions_ = self.last_train+predictions
                for i in range(len(predictions_)):
                    if i<len(predictions_)-self.difference:
                        predictions_[i+self.difference] = predictions_[i]+predictions_[i+self.difference]
                        forecasts = predictions_[-n_ahead:]
            else:    
                predictions.insert(0, self.last_train)
                forecasts = np.cumsum(predictions)[-n_ahead:]
        else:
            forecasts = np.array(predictions)

        if (self.trend ==True)&(self.trend_type =="component"):
            forecasts = trend_pred+forecasts    
        forecasts = np.array([max(0, x) for x in forecasts])   
        if self.box_cox == True:
            forecasts = back_box_cox_transform(y_pred = forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)
        return forecasts
    
    def cv(self, df, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_df = pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

            cv_tr_df = pd.DataFrame({"feat_name":self.model_lgb.feature_name_, "importance":self.model_lgb.feature_importances_}).sort_values(by = "importance", ascending = False)
            cv_tr_df["fold"] = i
            self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})
    
    def tune_model(self, df, cv_split, test_size, param_space,eval_metric, opt_horizon = None, eval_num = 100):
        # if 'lags' not in param_space:
        #     self.data_prep(df)

        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            if ('n_lag' in params)|('box_cox' in params)|('box_cox_lmda' in params)|('box_cox_biasadj' in params):
                if ('n_lag' in params):
                    if type(params["n_lag"]) is tuple:
                        self.n_lag = list(params["n_lag"])
                    else:
                        self.n_lag = range(1, params["n_lag"]+1)
                if ('box_cox' in params):
                    self.box_cox = params["box_cox"]
                if ('box_cox_lmda' in params):
                    self.lmda = params["box_cox_lmda"]

                if ('box_cox_biasadj' in params):
                    self.biasadj = params["box_cox_biasadj"]

                # self.data_prep(df)
                model =self.model(**{k: v for k, v in params.items() if (k not in ["box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj"])}, verbose=-1)

            else:
                model =self.model(**params, verbose=-1)  
            # model =self.model(**params, verbose=-1)

            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])

                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]

                # train_dfc = self.dfc.iloc[train_index]
                # self.X, self.y = train_dfc.drop(columns =self.target_col), train_dfc[self.target_col]

                self.model_lgb = model.fit(self.X, self.y, categorical_feature=self.cat_var)
                yhat = self.forecast(n_ahead =len(y_test), x_test=x_test)
                if eval_metric.__name__== 'mean_squared_error':
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:], squared=False)
                    else:
                        accuracy = eval_metric(y_test, yhat, squared=False)

                elif (eval_metric.__name__== 'MeanAbsoluteScaledError')|(eval_metric.__name__== 'MedianAbsoluteScaledError'):
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:], np.array(train[self.target_col]))
                    else:
                        accuracy = eval_metric(y_test, yhat, np.array(train[self.target_col]))
                else:
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:])
                    else:
                        accuracy = eval_metric(y_test, yhat)
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
    
    def direct_forecast(self, H, x_test = None):
        if x_test is not None:
            if isinstance(x_test, pd.Series):
                x_test = x_test.to_frame().T

        lags = self.y.tolist()

        if self.trend ==True:
            trend_pred = self.lr_model.predict(np.array([self.len+H]).reshape(-1, 1))[0]

        if x_test is not None:
            x_var = x_test.iloc[0, 0:].tolist()
        else:
            x_var = []
            
        if self.n_lag is not None:
            new_lag = [i-self.n_lag[0]+1 for i in self.n_lag]
            inp_lag = [lags[-l] for l in new_lag] # to get defined lagged variables 
        else:
            inp_lag = []

        if self.lag_transform is not None:
            transform_lag = []    
            for n, k in self.lag_transform.items():
                df_array = np.array(pd.Series(lags).shift(n-1))
                for f in k:
                    if f[0].__name__ == "rolling_quantile":
                        t1 = f[0](df_array, f[1], f[2])[-1]
                    else:
                        t1 = f[0](df_array, f[1])[-1]
                    transform_lag.append(t1)
        else:
            transform_lag = []
            
        if (self.trend ==True) & (self.trend_type == "feature"):
            trend_var = [trend_pred]
        else:
            trend_var = []
                
                
        inp = x_var+inp_lag+transform_lag+trend_var
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
        pred = self.model_lgb.predict(df_inp)[0]

        if (self.trend ==True)&(self.trend_type =="component"):
            forecasts = trend_pred+pred
        else:
            forecasts = pred
            
        return forecasts

    def multiple_direct_forecats(self, train, test, H, param = None):
        idx = train[-H:].index # get index of last H values of train data to iterate over H horizons to forecast all values of 
        preds = []
        for i in range(H):
            train_direct= train[:idx[i]] # set new train data to forecast H step further
            test_direct = test.loc[idx[i]+timedelta(H)] # subset test row wich corresponds to H step further from the last value of train data 
            if param is not None:
                self.fit(train_direct, param)
            else:
                self.fit(train_direct)
            forecast = self.direct_forecast(H, x_test=test_direct)
            preds.append(forecast)
            
        if self.difference is not None:
            if self.difference>1:
                predictions_ = self.last_train+preds
                for i in range(len(predictions_)):
                    if i<len(predictions_)-self.difference:
                        predictions_[i+self.difference] = predictions_[i]+predictions_[i+self.difference]
                        multi_forecasts = predictions_[-H:]
            else:    
                preds.insert(0, self.last_train)
                multi_forecasts = np.cumsum(preds)[-H:]
        else:
            multi_forecasts = np.array(preds)

        if self.box_cox == True:
            multi_forecasts = back_box_cox_transform(y_pred = multi_forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)

        return multi_forecasts
            
            
    def cv_direct(self, df, cv_split, H, metrics, param):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=H)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_df = pd.DataFrame()
        self.cv_forecats_df = pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if param is not None:
                bb_forecast = self.multiple_direct_forecats(train, x_test, H, param)
            else:
                bb_forecast = self.multiple_direct_forecats(train, x_test, H)
            forecat_df = test[self.target_col].to_frame()
            forecat_df["forecasts"] = bb_forecast
            
            self.cv_forecats_df = pd.concat([self.cv_forecats_df, forecat_df], axis=0)

            cv_tr_df = pd.DataFrame({"feat_name":self.model_lgb.feature_name_, "importance":self.model_lgb.feature_importances_}).sort_values(by = "importance", ascending = False)
            cv_tr_df["fold"] = i
            self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)


            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})
    
    def tune_direct_model(self, df, cv_split, test_size, param_space, eval_metric, eval_num= 100):

        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            if ('n_lag' in params)|('box_cox' in params)|('box_cox_lmda' in params)|('box_cox_biasadj' in params):
                if ('n_lag' in params):
                    if type(params["n_lag"]) is tuple:
                        self.n_lag = list(params["n_lag"])
                    else:
                        self.n_lag = range(1, params["n_lag"]+1)
                if ('box_cox' in params):
                    self.box_cox = params["box_cox"]
                if ('box_cox_lmda' in params):
                    self.lmda = params["box_cox_lmda"]

                if ('box_cox_biasadj' in params):
                    self.biasadj = params["box_cox_biasadj"]

                # self.data_prep(df)
                param_model = {k: v for k, v in params.items() if (k not in ["box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj"])}
            else:
                param_model = params 
            # metric = []
            actuals = []
            predictions = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test = test.iloc[:, 1:]


                actuals += test[self.target_col].tolist()
                predictions += list(self.multiple_direct_forecats(train, x_test, test_size, param_model))

            if eval_metric.__name__== 'mean_squared_error':
                accuracy = eval_metric(np.array(actuals), np.array(predictions), squared=False)
            elif (eval_metric.__name__== 'MeanAbsoluteScaledError')|(eval_metric.__name__== 'MedianAbsoluteScaledError'):
                accuracy = eval_metric(np.array(actuals), np.array(predictions), np.array(train[self.target_col]))
            else:
                accuracy = eval_metric(np.array(actuals), np.array(predictions))
#                 print(str(accuracy)+" and len is "+str(len(test)))
            # metric.append(accuracy)
            score = np.mean(accuracy)

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
            
class xgboost_forecaster:
    def __init__(self, target_col, add_trend = False, trend_type ="component", n_lag = None, lag_transform = None, differencing_number = None, cat_variables = None,
                 box_cox = False, box_cox_lmda = None, box_cox_biasadj= False):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = XGBRegressor
        self.target_col = target_col
        self.n_lag = n_lag
        self.difference = differencing_number
        self.lag_transform = lag_transform
        self.cat_variables = cat_variables
        self.trend = add_trend
        self.trend_type = trend_type
        self.box_cox = box_cox
        self.lmda = box_cox_lmda
        self.biasadj = box_cox_biasadj

        
    def data_prep(self, df):
        dfc = df.copy()
        if self.cat_variables is not None:
            for col, cat in self.cat_var.items():
                dfc[col] = dfc[col].astype('category')
                dfc[col] = dfc[col].cat.set_categories(cat)
            dfc = pd.get_dummies(dfc)

            for i in self.drop_categ:
                dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)
                
        if (self.target_col in dfc.columns):
            if self.box_cox == True:
                self.is_zero = np.any(np.array(dfc[self.target_col]) < 1)
                trans_data, self.lmda = box_cox_transform(x = dfc[self.target_col], shift = self.is_zero, box_cox_lmda=self.lmda)
                dfc[self.target_col] = trans_data
            
            if (self.trend ==True):
                self.len = len(dfc)
                self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
                
                if (self.trend_type == "component"):
                    dfc[self.target_col] = dfc[self.target_col]-self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))

            if self.difference is not None:
                if self.difference >1:
                    self.last_train = dfc[self.target_col].tolist()[-self.difference:]
                else:
                    self.last_train = dfc[self.target_col].tolist()[-1]
                dfc[self.target_col] = dfc[self.target_col].diff(self.difference)

            if self.n_lag is not None:
                for i in self.n_lag:
                    dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
                
            if self.lag_transform is not None:
                for n, k in self.lag_transform.items():
                    df_array = np.array(dfc[self.target_col].shift(n))
                    for f in k:
                        if f[0].__name__ == "rolling_quantile":
                            dfc["q_"+str(f[2])+"_"+str(n)+"_"+str(f[1])] = f[0](df_array, f[1], f[2])
                        else:
                            dfc[f[0].__name__+"_"+str(n)+"_"+str(f[1])] = f[0](df_array, f[1]) 
                            
        if (self.trend ==True) & (self.trend_type == "feature"):

            if (self.target_col in dfc.columns):
                dfc["trend"] = self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
        dfc = dfc.dropna()


        # if self.target_col in dfc.columns:
        #     self.dfc = dfc
        #     self.df =df.loc[dfc.index]
        # else:
        return dfc

    def fit(self, df, param = None):
        if param is not None:
            model_xgb =self.model(**param)
        else:
            model_xgb =self.model()
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        
        model_train = self.data_prep(df)
        self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]

        self.model_xgb = model_xgb.fit(self.X, self.y, verbose = True)

    def forecast(self, n_ahead, x_test = None):
        if x_test is not None:
            x_dummy = self.data_prep(x_test)
        lags = self.y.tolist()
        predictions = []

        if self.trend ==True:
            trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))

        for i in range(n_ahead):
            if x_test is not None:
                x_var = x_dummy.iloc[i, 0:].tolist()
            else:
                x_var = []
                
            if self.n_lag is not None:
                inp_lag = [lags[-l] for l in self.n_lag] # to get defined lagged variables
                # if self.forecast_type =="recursive":
                #     inp_lag = [lags[-l] for l in self.n_lag] # to get defined lagged variables
                # elif self.forecast_type =="direct":
                #     lag_direct = [i-self.n_lag[0]+1 for i in self.n_lag]
                #     inp_lag = [lags[-l] for l in lag_direct] # to get defined lagged variables
            else:
                inp_lag = []

            if self.lag_transform is not None:
                transform_lag = []    
                for n, k in self.lag_transform.items():
                    df_array = np.array(pd.Series(lags).shift(n-1))
                    for f in k:
                        if f[0].__name__ == "rolling_quantile":
                            t1 = f[0](df_array, f[1], f[2])[-1]
                        else:
                            t1 = f[0](df_array, f[1])[-1]
                        transform_lag.append(t1)
            else:
                transform_lag = []
                
            if (self.trend ==True) & (self.trend_type == "feature"):
                trend_var = [trend_pred[i]]
            else:
                trend_var = [] 
                    
            inp = x_var+inp_lag+transform_lag+trend_var
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns

            pred = self.model_xgb.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)

        if self.difference is not None:
            if self.difference>1:
                predictions_ = self.last_train+predictions
                for i in range(len(predictions_)):
                    if i<len(predictions_)-self.difference:
                        predictions_[i+self.difference] = predictions_[i]+predictions_[i+self.difference]
                        forecasts = predictions_[-n_ahead:]
            else:    
                predictions.insert(0, self.last_train)
                forecasts = np.cumsum(predictions)[-n_ahead:]
        else:
            forecasts = np.array(predictions)

        if (self.trend ==True)&(self.trend_type =="component"):
            forecasts = trend_pred+forecasts    
        forecasts = np.array([max(0, x) for x in forecasts])    
        if self.box_cox == True:
            forecasts = back_box_cox_transform(y_pred = forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)
        return forecasts
    
    def cv(self, df, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_df = pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

            cv_tr_df = pd.DataFrame({"feat_name":self.model_xgb.feature_names_in_, "importance":self.model_xgb.feature_importances_}).sort_values(by = "importance", ascending = False)
            cv_tr_df["fold"] = i
            self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})

    
    def tune_model(self, df, cv_split, test_size, param_space, eval_metric,opt_horizon = None, eval_num= 100):
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        # if 'lags' not in param_space:
        #     self.data_prep(df)

        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            if ('n_lag' in params)|('box_cox' in params)|('box_cox_lmda' in params)|('box_cox_biasadj' in params):
                if ('n_lag' in params):
                    if type(params["n_lag"]) is tuple:
                        self.n_lag = list(params["n_lag"])
                    else:
                        self.n_lag = range(1, params["n_lag"]+1)
                if ('box_cox' in params):
                    self.box_cox = params["box_cox"]
                if ('box_cox_lmda' in params):
                    self.lmda = params["box_cox_lmda"]

                if ('box_cox_biasadj' in params):
                    self.biasadj = params["box_cox_biasadj"]

                # self.data_prep(df)
                model =self.model(**{k: v for k, v in params.items() if (k not in ["box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj"])})
            else:
                model =self.model(**params)  
            # model =self.model(**params)   
            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])

                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                # train_dfc = self.dfc.iloc[train_index]
                # self.X, self.y = train_dfc.drop(columns =self.target_col), train_dfc[self.target_col]

                self.model_xgb = model.fit(self.X, self.y, verbose = True)

                yhat = self.forecast(n_ahead =len(y_test), x_test=x_test)
                if eval_metric.__name__== 'mean_squared_error':
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:], squared=False)
                    else:
                        accuracy = eval_metric(y_test, yhat, squared=False)

                elif (eval_metric.__name__== 'MeanAbsoluteScaledError')|(eval_metric.__name__== 'MedianAbsoluteScaledError'):
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:], np.array(train[self.target_col]))
                    else:
                        accuracy = eval_metric(y_test, yhat, np.array(train[self.target_col]))
                else:
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:])
                    else:
                        accuracy = eval_metric(y_test, yhat)
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

    def direct_forecast(self, H, x_test = None):
        if x_test is not None:
            if isinstance(x_test, pd.Series):
                x_test = x_test.to_frame().T
            x_dummy = self.data_prep(x_test)
#         max_lag = self.n_lag[-1]
#         lags = self.y[-max_lag:].tolist()
        lags = self.y.tolist()

        if self.trend ==True:
            trend_pred = self.lr_model.predict(np.array([self.len+H]).reshape(-1, 1))[0]

        if x_test is not None:
            x_var = x_dummy.iloc[0, 0:].tolist()
        else:
            x_var = []
            
        if self.n_lag is not None:
            new_lag = [i-self.n_lag[0]+1 for i in self.n_lag]
            inp_lag = [lags[-l] for l in new_lag] # to get defined lagged variables 
        else:
            inp_lag = []

        if self.lag_transform is not None:
            transform_lag = []    
            for n, k in self.lag_transform.items():
                df_array = np.array(pd.Series(lags).shift(n-1))
                for f in k:
                    if f[0].__name__ == "rolling_quantile":
                        t1 = f[0](df_array, f[1], f[2])[-1]
                    else:
                        t1 = f[0](df_array, f[1])[-1]
                    transform_lag.append(t1)
        else:
            transform_lag = []
            
        if (self.trend ==True) & (self.trend_type == "feature"):
            trend_var = [trend_pred]
        else:
            trend_var = []
                
                
        inp = x_var+inp_lag+transform_lag+trend_var
        df_inp = pd.DataFrame(inp).T
        df_inp.columns = self.X.columns

        pred = self.model_xgb.predict(df_inp)[0]

        if (self.trend ==True)&(self.trend_type =="component"):
            forecasts = trend_pred+pred
        else:
            forecasts = pred
            
        return forecasts

    def multiple_direct_forecats(self, train, test, H, param = None):
        idx = train[-H:].index # get index of last H values of train data to iterate over H horizons to forecast all values of 
        preds = []
        for i in range(H):
            train_direct= train[:idx[i]] # set new train data to forecast H step further
            test_direct = test.loc[idx[i]+timedelta(H)] # subset test row wich corresponds to H step further from the last value of train data 
            if param is not None:
                self.fit(train_direct, param)
            else:
                self.fit(train_direct)
            forecast = self.direct_forecast(H, x_test=test_direct)
            preds.append(forecast)
            
        if self.difference is not None:
            if self.difference>1:
                predictions_ = self.last_train+preds
                for i in range(len(predictions_)):
                    if i<len(predictions_)-self.difference:
                        predictions_[i+self.difference] = predictions_[i]+predictions_[i+self.difference]
                        multi_forecasts = predictions_[-H:]
            else:    
                preds.insert(0, self.last_train)
                multi_forecasts = np.cumsum(preds)[-H:]
        else:
            multi_forecasts = np.array(preds)

        if self.box_cox == True:
            multi_forecasts = back_box_cox_transform(y_pred = multi_forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)

        return multi_forecasts
            
            
    def cv_direct(self, df, cv_split, H, metrics, param):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=H)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_df = pd.DataFrame()
        self.cv_forecats_df = pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if param is not None:
                bb_forecast = self.multiple_direct_forecats(train, x_test, H, param)
            else:
                bb_forecast = self.multiple_direct_forecats(train, x_test, H)
            forecat_df = test[self.target_col].to_frame()
            forecat_df["forecasts"] = bb_forecast
            
            self.cv_forecats_df = pd.concat([self.cv_forecats_df, forecat_df], axis=0)

            cv_tr_df = pd.DataFrame({"feat_name":self.model_xgb.feature_names_in_, "importance":self.model_xgb.feature_importances_}).sort_values(by = "importance", ascending = False)
            cv_tr_df["fold"] = i
            self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})
    
    def tune_direct_model(self, df, cv_split, test_size, param_space, eval_metric, eval_num= 100):
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        # if 'lags' not in param_space:
        #     self.data_prep(df)

        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            if ('n_lag' in params)|('box_cox' in params)|('box_cox_lmda' in params)|('box_cox_biasadj' in params):
                if ('n_lag' in params):
                    if type(params["n_lag"]) is tuple:
                        self.n_lag = list(params["n_lag"])
                    else:
                        self.n_lag = range(1, params["n_lag"]+1)
                if ('box_cox' in params):
                    self.box_cox = params["box_cox"]
                if ('box_cox_lmda' in params):
                    self.lmda = params["box_cox_lmda"]

                if ('box_cox_biasadj' in params):
                    self.biasadj = params["box_cox_biasadj"]

                # self.data_prep(df)
                param_model = {k: v for k, v in params.items() if (k not in ["box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj"])}
            else:
                param_model = params  
            # model =self.model(**params)   
            # metric = []
            actuals = []
            predictions = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test = test.iloc[:, 1:]


                actuals += test[self.target_col].tolist()
                predictions += list(self.multiple_direct_forecats(train, x_test, test_size, param_model))

            if eval_metric.__name__== 'mean_squared_error':
                accuracy = eval_metric(np.array(actuals), np.array(predictions), squared=False)
            elif (eval_metric.__name__== 'MeanAbsoluteScaledError')|(eval_metric.__name__== 'MedianAbsoluteScaledError'):
                accuracy = eval_metric(np.array(actuals), np.array(predictions), np.array(train[self.target_col]))
            else:
                accuracy = eval_metric(np.array(actuals), np.array(predictions))
#                 print(str(accuracy)+" and len is "+str(len(test)))
            # metric.append(accuracy)
            score = np.mean(accuracy)

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
    
class RandomForest_forecaster:
    def __init__(self, target_col,add_trend = False, trend_type ="component", n_lag=None, lag_transform = None, differencing_number = None, cat_variables = None,
                 box_cox = False, box_cox_lmda = None, box_cox_biasadj= False):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = RandomForestRegressor
        self.target_col = target_col
        self.n_lag = n_lag
        self.difference = differencing_number
        self.lag_transform = lag_transform
        self.cat_variables = cat_variables
        self.trend = add_trend
        self.trend_type = trend_type
        self.box_cox = box_cox
        self.lmda = box_cox_lmda
        self.biasadj = box_cox_biasadj
        
    
        
    def data_prep(self, df):
        dfc = df.copy()
        if self.cat_variables is not None:
            for col, cat in self.cat_var.items():
                dfc[col] = dfc[col].astype('category')
                dfc[col] = dfc[col].cat.set_categories(cat)
            dfc = pd.get_dummies(dfc)

            for i in self.drop_categ:
                dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)

        if (self.target_col in dfc.columns):

            if self.box_cox == True:
                self.is_zero = np.any(np.array(dfc[self.target_col]) < 1)
                trans_data, self.lmda = box_cox_transform(x = dfc[self.target_col], shift = self.is_zero, box_cox_lmda=self.lmda)
                dfc[self.target_col] = trans_data
            
            if (self.trend ==True):
                self.len = len(dfc)
                self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
                
                if (self.trend_type == "component"):
                    dfc[self.target_col] = dfc[self.target_col]-self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
            if self.difference is not None:
                if self.difference >1:
                    self.last_train = dfc[self.target_col].tolist()[-self.difference:]
                else:
                    self.last_train = dfc[self.target_col].tolist()[-1]
                dfc[self.target_col] = dfc[self.target_col].diff(self.difference)

            if self.n_lag is not None:
                for i in self.n_lag:
                    dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
                
            if self.lag_transform is not None:
                for n, k in self.lag_transform.items():
                    df_array = np.array(dfc[self.target_col].shift(n))
                    for f in k:
                        if f[0].__name__ == "rolling_quantile":
                            dfc["q_"+str(f[2])+"_"+str(n)+"_"+str(f[1])] = f[0](df_array, f[1], f[2])
                        else:
                            dfc[f[0].__name__+"_"+str(n)+"_"+str(f[1])] = f[0](df_array, f[1]) 
                            
        if (self.trend ==True) & (self.trend_type == "feature"):

            if (self.target_col in dfc.columns):
                dfc["trend"] = self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
        dfc = dfc.dropna()


        # if self.target_col in dfc.columns:
        #     self.dfc = dfc
        #     self.df =df.loc[dfc.index]
        # else:
        return dfc

    def fit(self, df, param = None):
        if param is not None:
            model_rf =self.model(**param)
        else:
            model_rf =self.model()
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        # model_df = self.data_prep(df)
        model_train = self.data_prep(df)
        self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
        self.model_rf = model_rf.fit(self.X, self.y)

    def forecast(self, n_ahead, x_test = None):
        if x_test is not None:
            x_dummy = self.data_prep(x_test)
#         max_lag = self.n_lag[-1]
#         lags = self.y[-max_lag:].tolist()
        lags = self.y.tolist()
        predictions = []

        if self.trend ==True:
            trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))

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
                transform_lag = []    
                for n, k in self.lag_transform.items():
                    df_array = np.array(pd.Series(lags).shift(n-1))
                    for f in k:
                        if f[0].__name__ == "rolling_quantile":
                            t1 = f[0](df_array, f[1], f[2])[-1]
                        else:
                            t1 = f[0](df_array, f[1])[-1]
                        transform_lag.append(t1)
            else:
                transform_lag = []
                
            if (self.trend ==True) & (self.trend_type == "feature"):
                trend_var = [trend_pred[i]]
            else:
                trend_var = []
                    
                    
            inp = x_var+inp_lag+transform_lag+trend_var
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns

            pred = self.model_rf.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)

        if self.difference is not None:
            if self.difference>1:
                predictions_ = self.last_train+predictions
                for i in range(len(predictions_)):
                    if i<len(predictions_)-self.difference:
                        predictions_[i+self.difference] = predictions_[i]+predictions_[i+self.difference]
                        forecasts = predictions_[-n_ahead:]
            else:    
                predictions.insert(0, self.last_train)
                forecasts = np.cumsum(predictions)[-n_ahead:]
        else:
            forecasts = np.array(predictions)
        if (self.trend ==True)&(self.trend_type =="component"):
            forecasts = trend_pred+forecasts    
        forecasts = np.array([max(0, x) for x in forecasts])      
        if self.box_cox == True:
            forecasts = back_box_cox_transform(y_pred = forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)
        return forecasts
    
    def cv(self, df, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_df = pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

            cv_tr_df = pd.DataFrame({"feat_name":self.model_rf.feature_names_in_, "importance":self.model_rf.feature_importances_}).sort_values(by = "importance", ascending = False)
            cv_tr_df["fold"] = i
            self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})
    
    def tune_model(self, df, cv_split, test_size, param_space, eval_metric, opt_horizon = None, eval_num= 100):
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        # if 'lags' not in param_space:
        #     self.data_prep(df)

        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            if ('n_lag' in params)|('box_cox' in params)|('box_cox_lmda' in params)|('box_cox_biasadj' in params):
                if ('n_lag' in params):
                    if type(params["n_lag"]) is tuple:
                        self.n_lag = list(params["n_lag"])
                    else:
                        self.n_lag = range(1, params["n_lag"]+1)
                if ('box_cox' in params):
                    self.box_cox = params["box_cox"]
                if ('box_cox_lmda' in params):
                    self.lmda = params["box_cox_lmda"]

                if ('box_cox_biasadj' in params):
                    self.biasadj = params["box_cox_biasadj"]

                # self.data_prep(df)
                model =self.model(**{k: v for k, v in params.items() if (k not in ["box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj"])})
            else:
                model =self.model(**params)  
            # model =self.model(**params)  
                
            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])

                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                # train_dfc = self.dfc.iloc[train_index]
                # self.X, self.y = train_dfc.drop(columns =self.target_col), train_dfc[self.target_col]

                self.model_rf = model.fit(self.X, self.y)
                yhat = self.forecast(n_ahead =len(y_test), x_test=x_test)
                if eval_metric.__name__== 'mean_squared_error':
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:], squared=False)
                    else:
                        accuracy = eval_metric(y_test, yhat, squared=False)

                elif (eval_metric.__name__== 'MeanAbsoluteScaledError')|(eval_metric.__name__== 'MedianAbsoluteScaledError'):
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:], np.array(train[self.target_col]))
                    else:
                        accuracy = eval_metric(y_test, yhat, np.array(train[self.target_col]))
                else:
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:])
                    else:
                        accuracy = eval_metric(y_test, yhat)
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
    def direct_forecast(self, H, x_test = None):
        if x_test is not None:
            if isinstance(x_test, pd.Series):
                x_test = x_test.to_frame().T
            x_dummy = self.data_prep(x_test)
#         max_lag = self.n_lag[-1]
#         lags = self.y[-max_lag:].tolist()
        lags = self.y.tolist()

        if self.trend ==True:
            trend_pred = self.lr_model.predict(np.array([self.len+H]).reshape(-1, 1))[0]

        if x_test is not None:
            x_var = x_dummy.iloc[0, 0:].tolist()
        else:
            x_var = []
            
        if self.n_lag is not None:
            new_lag = [i-self.n_lag[0]+1 for i in self.n_lag]
            inp_lag = [lags[-l] for l in new_lag] # to get defined lagged variables 
        else:
            inp_lag = []

        if self.lag_transform is not None:
            transform_lag = []    
            for n, k in self.lag_transform.items():
                df_array = np.array(pd.Series(lags).shift(n-1))
                for f in k:
                    if f[0].__name__ == "rolling_quantile":
                        t1 = f[0](df_array, f[1], f[2])[-1]
                    else:
                        t1 = f[0](df_array, f[1])[-1]
                    transform_lag.append(t1)
        else:
            transform_lag = []
            
        if (self.trend ==True) & (self.trend_type == "feature"):
            trend_var = [trend_pred]
        else:
            trend_var = []
                
                
        inp = x_var+inp_lag+transform_lag+trend_var
        df_inp = pd.DataFrame(inp).T
        df_inp.columns = self.X.columns

        pred = self.model_rf.predict(df_inp)[0]

        if (self.trend ==True)&(self.trend_type =="component"):
            forecasts = trend_pred+pred
        else:
            forecasts = pred
            
        return forecasts

    def multiple_direct_forecats(self, train, test, H, param = None):
        idx = train[-H:].index # get index of last H values of train data to iterate over H horizons to forecast all values of 
        preds = []
        for i in range(H):
            train_direct= train[:idx[i]] # set new train data to forecast H step further
            test_direct = test.loc[idx[i]+timedelta(H)] # subset test row wich corresponds to H step further from the last value of train data
            if param is not None:
                self.fit(train_direct, param)
            else:
                self.fit(train_direct)
            forecast = self.direct_forecast(H, x_test=test_direct)
            preds.append(forecast)
            
        if self.difference is not None:
            if self.difference>1:
                predictions_ = self.last_train+preds
                for i in range(len(predictions_)):
                    if i<len(predictions_)-self.difference:
                        predictions_[i+self.difference] = predictions_[i]+predictions_[i+self.difference]
                        multi_forecasts = predictions_[-H:]
            else:    
                preds.insert(0, self.last_train)
                multi_forecasts = np.cumsum(preds)[-H:]
        else:
            multi_forecasts = np.array(preds)

        if self.box_cox == True:
            multi_forecasts = back_box_cox_transform(y_pred = multi_forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)

        return multi_forecasts
            
            
    def cv_direct(self, df, cv_split, H, metrics, param):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=H)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_df = pd.DataFrame()
        self.cv_forecats_df = pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if param is not None:
                bb_forecast = self.multiple_direct_forecats(train, x_test, H, param)
            else:
                bb_forecast = self.multiple_direct_forecats(train, x_test, H)
            forecat_df = test[self.target_col].to_frame()
            forecat_df["forecasts"] = bb_forecast
            
            self.cv_forecats_df = pd.concat([self.cv_forecats_df, forecat_df], axis=0)

            cv_tr_df = pd.DataFrame({"feat_name":self.model_rf.feature_names_in_, "importance":self.model_rf.feature_importances_}).sort_values(by = "importance", ascending = False)
            cv_tr_df["fold"] = i
            self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})
    
    def tune_direct_model(self, df, cv_split, test_size, param_space, eval_metric, eval_num= 100):
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        # if 'lags' not in param_space:
        #     self.data_prep(df)

        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            if ('n_lag' in params)|('box_cox' in params)|('box_cox_lmda' in params)|('box_cox_biasadj' in params):
                if ('n_lag' in params):
                    if type(params["n_lag"]) is tuple:
                        self.n_lag = list(params["n_lag"])
                    else:
                        self.n_lag = range(1, params["n_lag"]+1)
                if ('box_cox' in params):
                    self.box_cox = params["box_cox"]
                if ('box_cox_lmda' in params):
                    self.lmda = params["box_cox_lmda"]

                if ('box_cox_biasadj' in params):
                    self.biasadj = params["box_cox_biasadj"]

                # self.data_prep(df)
                param_model = {k: v for k, v in params.items() if (k not in ["box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj"])}
            else:
                param_model = params  
            # metric = []
            actuals = []
            predictions = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test = test.iloc[:, 1:]


                actuals += test[self.target_col].tolist()
                predictions += list(self.multiple_direct_forecats(train, x_test, test_size, param_model))

            if eval_metric.__name__== 'mean_squared_error':
                accuracy = eval_metric(np.array(actuals), np.array(predictions), squared=False)
            elif (eval_metric.__name__== 'MeanAbsoluteScaledError')|(eval_metric.__name__== 'MedianAbsoluteScaledError'):
                accuracy = eval_metric(np.array(actuals), np.array(predictions), np.array(train[self.target_col]))
            else:
                accuracy = eval_metric(np.array(actuals), np.array(predictions))
#                 print(str(accuracy)+" and len is "+str(len(test)))
            # metric.append(accuracy)
            score = np.mean(accuracy)

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
    
class AdaBoost_forecaster:
    def __init__(self, target_col,add_trend = False, trend_type ="component", n_lag=None, lag_transform = None, differencing_number = None, cat_variables = None,
                 box_cox = False, box_cox_lmda = None, box_cox_biasadj= False):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = AdaBoostRegressor
        self.target_col = target_col
        self.n_lag = n_lag
        self.difference = differencing_number
        self.lag_transform = lag_transform
        self.cat_variables = cat_variables
        self.trend = add_trend
        self.trend_type = trend_type
        self.box_cox = box_cox
        self.lmda = box_cox_lmda
        self.biasadj = box_cox_biasadj


        
    def data_prep(self, df):
        dfc = df.copy()
            
        if self.cat_variables is not None:
            for col, cat in self.cat_var.items():
                dfc[col] = dfc[col].astype('category')
                dfc[col] = dfc[col].cat.set_categories(cat)
            dfc = pd.get_dummies(dfc)

            for i in self.drop_categ:
                dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)
                
        if (self.target_col in dfc.columns):

            if self.box_cox == True:
                self.is_zero = np.any(np.array(dfc[self.target_col]) < 1)
                trans_data, self.lmda = box_cox_transform(x = dfc[self.target_col], shift = self.is_zero, box_cox_lmda=self.lmda)
                dfc[self.target_col] = trans_data
            
            if (self.trend ==True):
                self.len = len(dfc)
                self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
                
                if (self.trend_type == "component"):
                    dfc[self.target_col] = dfc[self.target_col]-self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
                
            if self.difference is not None:
                if self.difference >1:
                    self.last_train = dfc[self.target_col].tolist()[-self.difference:]
                else:
                    self.last_train = dfc[self.target_col].tolist()[-1]
                dfc[self.target_col] = dfc[self.target_col].diff(self.difference)

            if self.n_lag is not None:
                for i in self.n_lag:
                    dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
                
            if self.lag_transform is not None:
                for n, k in self.lag_transform.items():
                    df_array = np.array(dfc[self.target_col].shift(n))
                    for f in k:
                        if f[0].__name__ == "rolling_quantile":
                            dfc["q_"+str(f[2])+"_"+str(n)+"_"+str(f[1])] = f[0](df_array, f[1], f[2])
                        else:
                            dfc[f[0].__name__+"_"+str(n)+"_"+str(f[1])] = f[0](df_array, f[1]) 
                            
        if (self.trend ==True) & (self.trend_type == "feature"):

            if (self.target_col in dfc.columns):
                dfc["trend"] = self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
        dfc = dfc.dropna()


        # if self.target_col in dfc.columns:
        #     self.dfc = dfc
        #     self.df =df.loc[dfc.index]
        # else:
        return dfc

    def fit(self, df, param = None):
        if param is not None:
            model_ada =self.model(**param)
        else:
            model_ada =self.model()
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        model_train = self.data_prep(df)
        self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
        self.model_ada = model_ada.fit(self.X, self.y)

    def forecast(self, n_ahead, x_test = None):
        self.H = n_ahead
        if x_test is not None:
            x_dummy = self.data_prep(x_test)
        lags = self.y.tolist()
        predictions = []
        
        if self.trend ==True:
            trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))
            
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
                transform_lag = []    
                for n, k in self.lag_transform.items():
                    df_array = np.array(pd.Series(lags).shift(n-1))
                    for f in k:
                        if f[0].__name__ == "rolling_quantile":
                            t1 = f[0](df_array, f[1], f[2])[-1]
                        else:
                            t1 = f[0](df_array, f[1])[-1]
                        transform_lag.append(t1)
            else:
                transform_lag = []
                
            if (self.trend ==True) & (self.trend_type == "feature"):
                trend_var = [trend_pred[i]]
            else:
                trend_var = []
                    
                    
            inp = x_var+inp_lag+transform_lag+trend_var
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns

            pred = self.model_ada.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)

        if self.difference is not None:
            if self.difference>1:
                predictions_ = self.last_train+predictions
                for i in range(len(predictions_)):
                    if i<len(predictions_)-self.difference:
                        predictions_[i+self.difference] = predictions_[i]+predictions_[i+self.difference]
                        forecasts = predictions_[-n_ahead:]
            else:    
                predictions.insert(0, self.last_train)
                forecasts = np.cumsum(predictions)[-n_ahead:]
        else:
            forecasts = np.array(predictions)
        if (self.trend ==True)&(self.trend_type =="component"):
            forecasts = trend_pred+forecasts    
        forecasts = np.array([max(0, x) for x in forecasts])  
        if self.box_cox == True:
            forecasts = back_box_cox_transform(y_pred = forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)
        return forecasts
    
    def cv(self, df, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_df = pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

            cv_tr_df = pd.DataFrame({"feat_name":self.model_ada.feature_names_in_, "importance":self.model_ada.feature_importances_}).sort_values(by = "importance", ascending = False)
            cv_tr_df["fold"] = i
            self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})

    
    def tune_model(self, df, cv_split, test_size, param_space, eval_metric, opt_horizon = None, eval_num= 100):
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        # if 'lags' not in param_space:
        #     self.data_prep(df)

        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            if ('n_lag' in params)|('box_cox' in params)|('box_cox_lmda' in params)|('box_cox_biasadj' in params):
                if ('n_lag' in params):
                    if type(params["n_lag"]) is tuple:
                        self.n_lag = list(params["n_lag"])
                    else:
                        self.n_lag = range(1, params["n_lag"]+1)
                if ('box_cox' in params):
                    self.box_cox = params["box_cox"]
                if ('box_cox_lmda' in params):
                    self.lmda = params["box_cox_lmda"]

                if ('box_cox_biasadj' in params):
                    self.biasadj = params["box_cox_biasadj"]

                # self.data_prep(df)
                model =self.model(**{k: v for k, v in params.items() if (k not in ["box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj"])})
            else:
                model =self.model(**params)  
            # model =self.model(**params)   
            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])

                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                # train_dfc = self.dfc.iloc[train_index]
                # self.X, self.y = train_dfc.drop(columns =self.target_col), train_dfc[self.target_col]

                self.model_ada = model.fit(self.X, self.y)


                yhat = self.forecast(n_ahead =len(y_test), x_test=x_test)
                if eval_metric.__name__== 'mean_squared_error':
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:], squared=False)
                    else:
                        accuracy = eval_metric(y_test, yhat, squared=False)

                elif (eval_metric.__name__== 'MeanAbsoluteScaledError')|(eval_metric.__name__== 'MedianAbsoluteScaledError'):
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:], np.array(train[self.target_col]))
                    else:
                        accuracy = eval_metric(y_test, yhat, np.array(train[self.target_col]))
                else:
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:])
                    else:
                        accuracy = eval_metric(y_test, yhat)
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
    
    def direct_forecast(self, H, x_test = None):
        if x_test is not None:
            if isinstance(x_test, pd.Series):
                x_test = x_test.to_frame().T
            x_dummy = self.data_prep(x_test)
#         max_lag = self.n_lag[-1]
#         lags = self.y[-max_lag:].tolist()
        lags = self.y.tolist()

        if self.trend ==True:
            trend_pred = self.lr_model.predict(np.array([self.len+H]).reshape(-1, 1))[0]

        if x_test is not None:
            x_var = x_dummy.iloc[0, 0:].tolist()
        else:
            x_var = []
            
        if self.n_lag is not None:
            new_lag = [i-self.n_lag[0]+1 for i in self.n_lag]
            inp_lag = [lags[-l] for l in new_lag] # to get defined lagged variables 
        else:
            inp_lag = []

        if self.lag_transform is not None:
            transform_lag = []    
            for n, k in self.lag_transform.items():
                df_array = np.array(pd.Series(lags).shift(n-1))
                for f in k:
                    if f[0].__name__ == "rolling_quantile":
                        t1 = f[0](df_array, f[1], f[2])[-1]
                    else:
                        t1 = f[0](df_array, f[1])[-1]
                    transform_lag.append(t1)
        else:
            transform_lag = []
            
        if (self.trend ==True) & (self.trend_type == "feature"):
            trend_var = [trend_pred]
        else:
            trend_var = []
                
                
        inp = x_var+inp_lag+transform_lag+trend_var
        df_inp = pd.DataFrame(inp).T
        df_inp.columns = self.X.columns

        pred = self.model_ada.predict(df_inp)[0]

        if (self.trend ==True)&(self.trend_type =="component"):
            forecasts = trend_pred+pred
        else:
            forecasts = pred
            
        return forecasts

    def multiple_direct_forecats(self, train, test, H, param = None):
        idx = train[-H:].index # get index of last H values of train data to iterate over H horizons to forecast all values of 
        preds = []
        for i in range(H):
            train_direct= train[:idx[i]] # set new train data to forecast H step further
            test_direct = test.loc[idx[i]+timedelta(H)] # subset test row wich corresponds to H step further from the last value of train data
            if param is not None:
                self.fit(train_direct, param)
            else:
                self.fit(train_direct)
            forecast = self.direct_forecast(H, x_test=test_direct)
            preds.append(forecast)
            
        if self.difference is not None:
            if self.difference>1:
                predictions_ = self.last_train+preds
                for i in range(len(predictions_)):
                    if i<len(predictions_)-self.difference:
                        predictions_[i+self.difference] = predictions_[i]+predictions_[i+self.difference]
                        multi_forecasts = predictions_[-H:]
            else:    
                preds.insert(0, self.last_train)
                multi_forecasts = np.cumsum(preds)[-H:]
        else:
            multi_forecasts = np.array(preds)

        if self.box_cox == True:
            multi_forecasts = back_box_cox_transform(y_pred = multi_forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)

        return multi_forecasts
            
            
    def cv_direct(self, df, cv_split, H, metrics, param):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=H)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_df = pd.DataFrame()
        self.cv_forecats_df = pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if param is not None:
                bb_forecast = self.multiple_direct_forecats(train, x_test, H, param)
            else:
                bb_forecast = self.multiple_direct_forecats(train, x_test, H)
            forecat_df = test[self.target_col].to_frame()
            forecat_df["forecasts"] = bb_forecast
            
            self.cv_forecats_df = pd.concat([self.cv_forecats_df, forecat_df], axis=0)

            cv_tr_df = pd.DataFrame({"feat_name":self.model_ada.feature_names_in_, "importance":self.model_ada.feature_importances_}).sort_values(by = "importance", ascending = False)
            cv_tr_df["fold"] = i
            self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})
    
    def tune_direct_model(self, df, cv_split, test_size, param_space, eval_metric, eval_num= 100):
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        # if 'lags' not in param_space:
        #     self.data_prep(df)

        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            if ('n_lag' in params)|('box_cox' in params)|('box_cox_lmda' in params)|('box_cox_biasadj' in params):
                if ('n_lag' in params):
                    if type(params["n_lag"]) is tuple:
                        self.n_lag = list(params["n_lag"])
                    else:
                        self.n_lag = range(1, params["n_lag"]+1)
                if ('box_cox' in params):
                    self.box_cox = params["box_cox"]
                if ('box_cox_lmda' in params):
                    self.lmda = params["box_cox_lmda"]

                if ('box_cox_biasadj' in params):
                    self.biasadj = params["box_cox_biasadj"]

                # self.data_prep(df)
                param_model = {k: v for k, v in params.items() if (k not in ["box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj"])}
            else:
                param_model = params  
            # metric = []
            actuals = []
            predictions = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test = test.iloc[:, 1:]


                actuals += test[self.target_col].tolist()
                predictions += list(self.multiple_direct_forecats(train, x_test, test_size, param_model))

            if eval_metric.__name__== 'mean_squared_error':
                accuracy = eval_metric(np.array(actuals), np.array(predictions), squared=False)
            elif (eval_metric.__name__== 'MeanAbsoluteScaledError')|(eval_metric.__name__== 'MedianAbsoluteScaledError'):
                accuracy = eval_metric(np.array(actuals), np.array(predictions), np.array(train[self.target_col]))
            else:
                accuracy = eval_metric(np.array(actuals), np.array(predictions))
#                 print(str(accuracy)+" and len is "+str(len(test)))
            # metric.append(accuracy)
            score = np.mean(accuracy)

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
    
class Cubist_forecaster:
    def __init__(self, target_col, add_trend = False, trend_type ="component", n_lag = None, lag_transform = None, differencing_number = None, cat_variables = None,
                 box_cox = False, box_cox_lmda = None, box_cox_biasadj= False):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = Cubist
        self.target_col = target_col
        self.n_lag = n_lag
        self.difference = differencing_number
        self.lag_transform = lag_transform
        self.cat_variables = cat_variables
        self.trend = add_trend
        self.trend_type = trend_type
        self.box_cox = box_cox
        self.lmda = box_cox_lmda
        self.biasadj = box_cox_biasadj

        
    def data_prep(self, df):
        dfc = df.copy()
            
        if self.cat_variables is not None:
            for col, cat in self.cat_var.items():
                dfc[col] = dfc[col].astype('category')
                dfc[col] = dfc[col].cat.set_categories(cat)
            dfc = pd.get_dummies(dfc)

            for i in self.drop_categ:
                dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)
                
        if (self.target_col in dfc.columns):

            if self.box_cox == True:
                self.is_zero = np.any(np.array(dfc[self.target_col]) < 1)
                trans_data, self.lmda = box_cox_transform(x = dfc[self.target_col], shift = self.is_zero, box_cox_lmda=self.lmda)
                dfc[self.target_col] = trans_data
            
            if (self.trend ==True):
                self.len = len(dfc)
                self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
                
                if (self.trend_type == "component"):
                    dfc[self.target_col] = dfc[self.target_col]-self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
                
            if self.difference is not None:
                if self.difference >1:
                    self.last_train = dfc[self.target_col].tolist()[-self.difference:]
                else:
                    self.last_train = dfc[self.target_col].tolist()[-1]
                dfc[self.target_col] = dfc[self.target_col].diff(self.difference)

            if self.n_lag is not None:
                for i in self.n_lag:
                    dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
                
            if self.lag_transform is not None:
                for n, k in self.lag_transform.items():
                    df_array = np.array(dfc[self.target_col].shift(n))
                    for f in k:
                        if f[0].__name__ == "rolling_quantile":
                            dfc["q_"+str(f[2])+"_"+str(n)+"_"+str(f[1])] = f[0](df_array, f[1], f[2])
                        else:
                            dfc[f[0].__name__+"_"+str(n)+"_"+str(f[1])] = f[0](df_array, f[1]) 
                            
        if (self.trend ==True) & (self.trend_type == "feature"):

            if (self.target_col in dfc.columns):
                dfc["trend"] = self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
        dfc = dfc.dropna()


        # if self.target_col in dfc.columns:
        #     self.dfc = dfc
        #     self.df =df.loc[dfc.index]
        # else:
        return dfc

    def fit(self, df, param = None):
        if param is not None:
            model_cub =self.model(**param)
        else:
            model_cub =self.model()
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        model_train = self.data_prep(df)
        self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
        self.model_cub = model_cub.fit(self.X, self.y)

    def forecast(self, n_ahead, x_test = None):
        self.H = n_ahead
        if x_test is not None:
            x_dummy = self.data_prep(x_test)
        lags = self.y.tolist()
        predictions = []
        
        if self.trend ==True:
            trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))
            
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
                transform_lag = []    
                for n, k in self.lag_transform.items():
                    df_array = np.array(pd.Series(lags).shift(n-1))
                    for f in k:
                        if f[0].__name__ == "rolling_quantile":
                            t1 = f[0](df_array, f[1], f[2])[-1]
                        else:
                            t1 = f[0](df_array, f[1])[-1]
                        transform_lag.append(t1)
            else:
                transform_lag = []
                
            if (self.trend ==True) & (self.trend_type == "feature"):
                trend_var = [trend_pred[i]]
            else:
                trend_var = []
                    
                    
            inp = x_var+inp_lag+transform_lag+trend_var
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns

            pred = self.model_cub.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)

        if self.difference is not None:
            if self.difference>1:
                predictions_ = self.last_train+predictions
                for i in range(len(predictions_)):
                    if i<len(predictions_)-self.difference:
                        predictions_[i+self.difference] = predictions_[i]+predictions_[i+self.difference]
                        forecasts = predictions_[-n_ahead:]
            else:    
                predictions.insert(0, self.last_train)
                forecasts = np.cumsum(predictions)[-n_ahead:]
        else:
            forecasts = np.array(predictions)
        if (self.trend ==True)&(self.trend_type =="component"):
            forecasts = trend_pred+forecasts  
        forecasts = np.array([max(0, x) for x in forecasts])        
        if self.box_cox == True:
            forecasts = back_box_cox_transform(y_pred = forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)
        return forecasts

    def cv(self, df, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {i.__name__:[] for i in metrics}
        # self.cv_df = pd.DataFrame()

        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

            # cv_tr_df = pd.DataFrame({"feat_name":self.model_ada.feature_names_in_, "importance":self.model_ada.feature_importances_}).sort_values(by = "importance", ascending = False)
            # cv_tr_df["fold"] = i
            # self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})

    def tune_model(self, df, cv_split, test_size, param_space, eval_metric, opt_horizon = None, eval_num= 100):
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        # if 'lags' not in param_space:
        #     self.data_prep(df)

        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            if ('n_lag' in params)|('box_cox' in params)|('box_cox_lmda' in params)|('box_cox_biasadj' in params):
                if ('n_lag' in params):
                    if type(params["n_lag"]) is tuple:
                        self.n_lag = list(params["n_lag"])
                    else:
                        self.n_lag = range(1, params["n_lag"]+1)
                if ('box_cox' in params):
                    self.box_cox = params["box_cox"]
                if ('box_cox_lmda' in params):
                    self.lmda = params["box_cox_lmda"]

                if ('box_cox_biasadj' in params):
                    self.biasadj = params["box_cox_biasadj"]

                # self.data_prep(df)
                model =self.model(**{k: v for k, v in params.items() if (k not in ["box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj"])})
            else:
                model =self.model(**params)  
            # model =self.model(**params)   
            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])

                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                # train_dfc = self.dfc.iloc[train_index]
                # self.X, self.y = train_dfc.drop(columns =self.target_col), train_dfc[self.target_col]

                self.model_cub = model.fit(self.X, self.y)


                yhat = self.forecast(n_ahead =len(y_test), x_test=x_test)
                if eval_metric.__name__== 'mean_squared_error':
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:], squared=False)
                    else:
                        accuracy = eval_metric(y_test, yhat, squared=False)

                elif (eval_metric.__name__== 'MeanAbsoluteScaledError')|(eval_metric.__name__== 'MedianAbsoluteScaledError'):
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:], np.array(train[self.target_col]))
                    else:
                        accuracy = eval_metric(y_test, yhat, np.array(train[self.target_col]))
                else:
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:])
                    else:
                        accuracy = eval_metric(y_test, yhat)
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
    
    def direct_forecast(self, H, x_test = None):
        if x_test is not None:
            if isinstance(x_test, pd.Series):
                x_test = x_test.to_frame().T
            x_dummy = self.data_prep(x_test)
#         max_lag = self.n_lag[-1]
#         lags = self.y[-max_lag:].tolist()
        lags = self.y.tolist()

        if self.trend ==True:
            trend_pred = self.lr_model.predict(np.array([self.len+H]).reshape(-1, 1))[0]

        if x_test is not None:
            x_var = x_dummy.iloc[0, 0:].tolist()
        else:
            x_var = []
            
        if self.n_lag is not None:
            new_lag = [i-self.n_lag[0]+1 for i in self.n_lag]
            inp_lag = [lags[-l] for l in new_lag] # to get defined lagged variables 
        else:
            inp_lag = []

        if self.lag_transform is not None:
            transform_lag = []    
            for n, k in self.lag_transform.items():
                df_array = np.array(pd.Series(lags).shift(n-1))
                for f in k:
                    if f[0].__name__ == "rolling_quantile":
                        t1 = f[0](df_array, f[1], f[2])[-1]
                    else:
                        t1 = f[0](df_array, f[1])[-1]
                    transform_lag.append(t1)
        else:
            transform_lag = []
            
        if (self.trend ==True) & (self.trend_type == "feature"):
            trend_var = [trend_pred]
        else:
            trend_var = []
                
                
        inp = x_var+inp_lag+transform_lag+trend_var
        df_inp = pd.DataFrame(inp).T
        df_inp.columns = self.X.columns

        pred = self.model_cub.predict(df_inp)[0]

        if (self.trend ==True)&(self.trend_type =="component"):
            forecasts = trend_pred+pred
        else:
            forecasts = pred
            
        return forecasts

    def multiple_direct_forecats(self, train, test, H, param = None):
        idx = train[-H:].index # get index of last H values of train data to iterate over H horizons to forecast all values of 
        preds = []
        for i in range(H):
            train_direct= train[:idx[i]] # set new train data to forecast H step further
            test_direct = test.loc[idx[i]+timedelta(H)] # subset test row wich corresponds to H step further from the last value of train data 
            if param is not None:
                self.fit(train_direct, param)
            else:
                self.fit(train_direct)
            forecast = self.direct_forecast(H, x_test=test_direct)
            preds.append(forecast)
            
        if self.difference is not None:
            if self.difference>1:
                predictions_ = self.last_train+preds
                for i in range(len(predictions_)):
                    if i<len(predictions_)-self.difference:
                        predictions_[i+self.difference] = predictions_[i]+predictions_[i+self.difference]
                        multi_forecasts = predictions_[-H:]
            else:    
                preds.insert(0, self.last_train)
                multi_forecasts = np.cumsum(preds)[-H:]
        else:
            multi_forecasts = np.array(preds)

        if self.box_cox == True:
            multi_forecasts = back_box_cox_transform(y_pred = multi_forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)

        return multi_forecasts
            
            
    def cv_direct(self, df, cv_split, H, metrics, param):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=H)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_forecats_df = pd.DataFrame()

        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if param is not None:
                bb_forecast = self.multiple_direct_forecats(train, x_test, H, param)
            else:
                bb_forecast = self.multiple_direct_forecats(train, x_test, H)
            forecat_df = test[self.target_col].to_frame()
            forecat_df["forecasts"] = bb_forecast
            
            self.cv_forecats_df = pd.concat([self.cv_forecats_df, forecat_df], axis=0)


            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})
    
    def tune_direct_model(self, df, cv_split, test_size, param_space, eval_metric, eval_num= 100):
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        # if 'lags' not in param_space:
        #     self.data_prep(df)

        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            if ('n_lag' in params)|('box_cox' in params)|('box_cox_lmda' in params)|('box_cox_biasadj' in params):
                if ('n_lag' in params):
                    if type(params["n_lag"]) is tuple:
                        self.n_lag = list(params["n_lag"])
                    else:
                        self.n_lag = range(1, params["n_lag"]+1)
                if ('box_cox' in params):
                    self.box_cox = params["box_cox"]
                if ('box_cox_lmda' in params):
                    self.lmda = params["box_cox_lmda"]

                if ('box_cox_biasadj' in params):
                    self.biasadj = params["box_cox_biasadj"]

                # self.data_prep(df)
                param_model = {k: v for k, v in params.items() if (k not in ["box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj"])}
            else:
                param_model = params  
            # metric = []
            actuals = []
            predictions = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test = test.iloc[:, 1:]


                actuals += test[self.target_col].tolist()
                predictions += list(self.multiple_direct_forecats(train, x_test, test_size, param_model))

            if eval_metric.__name__== 'mean_squared_error':
                accuracy = eval_metric(np.array(actuals), np.array(predictions), squared=False)
            elif (eval_metric.__name__== 'MeanAbsoluteScaledError')|(eval_metric.__name__== 'MedianAbsoluteScaledError'):
                accuracy = eval_metric(np.array(actuals), np.array(predictions), np.array(train[self.target_col]))
            else:
                accuracy = eval_metric(np.array(actuals), np.array(predictions))
#                 print(str(accuracy)+" and len is "+str(len(test)))
            # metric.append(accuracy)
            score = np.mean(accuracy)

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
    
class HistGradientBoosting_forecaster:
    def __init__(self, target_col, add_trend = False, trend_type ="component", n_lag = None, lag_transform = None, differencing_number = None, cat_variables = None,
                 box_cox = False, box_cox_lmda = None, box_cox_biasadj= False):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = HistGradientBoostingRegressor
        self.target_col = target_col
        self.n_lag = n_lag
        self.difference = differencing_number
        self.lag_transform = lag_transform
        self.cat_variables = cat_variables
        self.trend = add_trend
        self.trend_type = trend_type
        self.box_cox = box_cox
        self.lmda = box_cox_lmda
        self.biasadj = box_cox_biasadj

        
    def data_prep(self, df):
        dfc = df.copy()
        if self.cat_variables is not None:
            for col, cat in self.cat_var.items():
                dfc[col] = dfc[col].astype('category')
                dfc[col] = dfc[col].cat.set_categories(cat)
            dfc = pd.get_dummies(dfc)

            for i in self.drop_categ:
                dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)
                
        if (self.target_col in dfc.columns):
            if self.box_cox == True:
                self.is_zero = np.any(np.array(dfc[self.target_col]) < 1)
                trans_data, self.lmda = box_cox_transform(x = dfc[self.target_col], shift = self.is_zero, box_cox_lmda=self.lmda)
                dfc[self.target_col] = trans_data
            
            if (self.trend ==True):
                self.len = len(dfc)
                self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
                
                if (self.trend_type == "component"):
                    dfc[self.target_col] = dfc[self.target_col]-self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))

            if self.difference is not None:
                if self.difference >1:
                    self.last_train = dfc[self.target_col].tolist()[-self.difference:]
                else:
                    self.last_train = dfc[self.target_col].tolist()[-1]
                dfc[self.target_col] = dfc[self.target_col].diff(self.difference)

            if self.n_lag is not None:
                for i in self.n_lag:
                    dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
                
            if self.lag_transform is not None:
                for n, k in self.lag_transform.items():
                    df_array = np.array(dfc[self.target_col].shift(n))
                    for f in k:
                        if f[0].__name__ == "rolling_quantile":
                            dfc["q_"+str(f[2])+"_"+str(n)+"_"+str(f[1])] = f[0](df_array, f[1], f[2])
                        else:
                            dfc[f[0].__name__+"_"+str(n)+"_"+str(f[1])] = f[0](df_array, f[1]) 
                            
        if (self.trend ==True) & (self.trend_type == "feature"):

            if (self.target_col in dfc.columns):
                dfc["trend"] = self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
        dfc = dfc.dropna()

        return dfc

    def fit(self, df, param = None):
        if param is not None:
            model_hist =self.model(**param)
        else:
            model_hist =self.model()
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        
        model_train = self.data_prep(df)
        self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]

        self.model_hist = model_hist.fit(self.X, self.y)

    def forecast(self, n_ahead, x_test = None):
        if x_test is not None:
            x_dummy = self.data_prep(x_test)
        lags = self.y.tolist()
        predictions = []

        if self.trend ==True:
            trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))

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
                transform_lag = []    
                for n, k in self.lag_transform.items():
                    df_array = np.array(pd.Series(lags).shift(n-1))
                    for f in k:
                        if f[0].__name__ == "rolling_quantile":
                            t1 = f[0](df_array, f[1], f[2])[-1]
                        else:
                            t1 = f[0](df_array, f[1])[-1]
                        transform_lag.append(t1)
            else:
                transform_lag = []
                
            if (self.trend ==True) & (self.trend_type == "feature"):
                trend_var = [trend_pred[i]]
            else:
                trend_var = [] 
                    
            inp = x_var+inp_lag+transform_lag+trend_var
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns

            pred = self.model_hist.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)

        if self.difference is not None:
            if self.difference>1:
                predictions_ = self.last_train+predictions
                for i in range(len(predictions_)):
                    if i<len(predictions_)-self.difference:
                        predictions_[i+self.difference] = predictions_[i]+predictions_[i+self.difference]
                        forecasts = predictions_[-n_ahead:]
            else:    
                predictions.insert(0, self.last_train)
                forecasts = np.cumsum(predictions)[-n_ahead:]
        else:
            forecasts = np.array(predictions)

        if (self.trend ==True)&(self.trend_type =="component"):
            forecasts = trend_pred+forecasts    
        forecasts = np.array([max(0, x) for x in forecasts])      
        if self.box_cox == True:
            forecasts = back_box_cox_transform(y_pred = forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)
        return forecasts

    def cv(self, df, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        # self.cv_df = pd.DataFrame()

        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

            # cv_tr_df = pd.DataFrame({"feat_name":self.model_hist.feature_names_in_, "importance":self.model_hist.feature_importances_}).sort_values(by = "importance", ascending = False)
            # cv_tr_df["fold"] = i
            # self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})

    
    def tune_model(self, df, cv_split, test_size, param_space, eval_metric, opt_horizon = None, eval_num= 100):
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]


        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            if ('n_lag' in params)|('box_cox' in params)|('box_cox_lmda' in params)|('box_cox_biasadj' in params):
                if ('n_lag' in params):
                    if type(params["n_lag"]) is tuple:
                        self.n_lag = list(params["n_lag"])
                    else:
                        self.n_lag = range(1, params["n_lag"]+1)
                if ('box_cox' in params):
                    self.box_cox = params["box_cox"]
                if ('box_cox_lmda' in params):
                    self.lmda = params["box_cox_lmda"]

                if ('box_cox_biasadj' in params):
                    self.biasadj = params["box_cox_biasadj"]

                model =self.model(**{k: v for k, v in params.items() if (k not in ["box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj"])})
            else:
                model =self.model(**params)   
            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])

                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]

                self.model_hist = model.fit(self.X, self.y)

                yhat = self.forecast(n_ahead =len(y_test), x_test=x_test)
                if eval_metric.__name__== 'mean_squared_error':
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:], squared=False)
                    else:
                        accuracy = eval_metric(y_test, yhat, squared=False)

                elif (eval_metric.__name__== 'MeanAbsoluteScaledError')|(eval_metric.__name__== 'MedianAbsoluteScaledError'):
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:], np.array(train[self.target_col]))
                    else:
                        accuracy = eval_metric(y_test, yhat, np.array(train[self.target_col]))
                else:
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:])
                    else:
                        accuracy = eval_metric(y_test, yhat)
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
    
    def direct_forecast(self, H, x_test = None):
        if x_test is not None:
            if isinstance(x_test, pd.Series):
                x_test = x_test.to_frame().T
            x_dummy = self.data_prep(x_test)
#         max_lag = self.n_lag[-1]
#         lags = self.y[-max_lag:].tolist()
        lags = self.y.tolist()

        if self.trend ==True:
            trend_pred = self.lr_model.predict(np.array([self.len+H]).reshape(-1, 1))[0]

        if x_test is not None:
            x_var = x_dummy.iloc[0, 0:].tolist()
        else:
            x_var = []
            
        if self.n_lag is not None:
            new_lag = [i-self.n_lag[0]+1 for i in self.n_lag]
            inp_lag = [lags[-l] for l in new_lag] # to get defined lagged variables 
        else:
            inp_lag = []

        if self.lag_transform is not None:
            transform_lag = []    
            for n, k in self.lag_transform.items():
                df_array = np.array(pd.Series(lags).shift(n-1))
                for f in k:
                    if f[0].__name__ == "rolling_quantile":
                        t1 = f[0](df_array, f[1], f[2])[-1]
                    else:
                        t1 = f[0](df_array, f[1])[-1]
                    transform_lag.append(t1)
        else:
            transform_lag = []
            
        if (self.trend ==True) & (self.trend_type == "feature"):
            trend_var = [trend_pred]
        else:
            trend_var = []
                
                
        inp = x_var+inp_lag+transform_lag+trend_var
        df_inp = pd.DataFrame(inp).T
        df_inp.columns = self.X.columns

        pred = self.model_hist.predict(df_inp)[0]

        if (self.trend ==True)&(self.trend_type =="component"):
            forecasts = trend_pred+pred
        else:
            forecasts = pred
            
        return forecasts

    def multiple_direct_forecats(self, train, test, H, param = None):
        idx = train[-H:].index # get index of last H values of train data to iterate over H horizons to forecast all values of 
        preds = []
        for i in range(H):
            train_direct= train[:idx[i]] # set new train data to forecast H step further
            test_direct = test.loc[idx[i]+timedelta(H)] # subset test row wich corresponds to H step further from the last value of train data 
            if param is not None:
                self.fit(train_direct, param)
            else:
                self.fit(train_direct)
            forecast = self.direct_forecast(H, x_test=test_direct)
            preds.append(forecast)
            
        if self.difference is not None:
            if self.difference>1:
                predictions_ = self.last_train+preds
                for i in range(len(predictions_)):
                    if i<len(predictions_)-self.difference:
                        predictions_[i+self.difference] = predictions_[i]+predictions_[i+self.difference]
                        multi_forecasts = predictions_[-H:]
            else:    
                preds.insert(0, self.last_train)
                multi_forecasts = np.cumsum(preds)[-H:]
        else:
            multi_forecasts = np.array(preds)

        if self.box_cox == True:
            multi_forecasts = back_box_cox_transform(y_pred = multi_forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)

        return multi_forecasts
            
            
    def cv_direct(self, df, cv_split, H, metrics, param):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=H)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_forecats_df = pd.DataFrame()

        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if param is not None:
                bb_forecast = self.multiple_direct_forecats(train, x_test, H, param)
            else:
                bb_forecast = self.multiple_direct_forecats(train, x_test, H)
            forecat_df = test[self.target_col].to_frame()
            forecat_df["forecasts"] = bb_forecast
            
            self.cv_forecats_df = pd.concat([self.cv_forecats_df, forecat_df], axis=0)


            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})
    
    def tune_direct_model(self, df, cv_split, test_size, param_space, eval_metric, eval_num= 100):
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        # if 'lags' not in param_space:
        #     self.data_prep(df)

        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            if ('n_lag' in params)|('box_cox' in params)|('box_cox_lmda' in params)|('box_cox_biasadj' in params):
                if ('n_lag' in params):
                    if type(params["n_lag"]) is tuple:
                        self.n_lag = list(params["n_lag"])
                    else:
                        self.n_lag = range(1, params["n_lag"]+1)
                if ('box_cox' in params):
                    self.box_cox = params["box_cox"]
                if ('box_cox_lmda' in params):
                    self.lmda = params["box_cox_lmda"]

                if ('box_cox_biasadj' in params):
                    self.biasadj = params["box_cox_biasadj"]

                # self.data_prep(df)
                param_model = {k: v for k, v in params.items() if (k not in ["box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj"])}
            else:
                param_model = params  
            # model =self.model(**params)   
            # metric = []
            actuals = []
            predictions = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test = test.iloc[:, 1:]


                actuals += test[self.target_col].tolist()
                predictions += list(self.multiple_direct_forecats(train, x_test, test_size, param_model))

            if eval_metric.__name__== 'mean_squared_error':
                accuracy = eval_metric(np.array(actuals), np.array(predictions), squared=False)
            elif (eval_metric.__name__== 'MeanAbsoluteScaledError')|(eval_metric.__name__== 'MedianAbsoluteScaledError'):
                accuracy = eval_metric(np.array(actuals), np.array(predictions), np.array(train[self.target_col]))
            else:
                accuracy = eval_metric(np.array(actuals), np.array(predictions))
#                 print(str(accuracy)+" and len is "+str(len(test)))
            # metric.append(accuracy)
            score = np.mean(accuracy)

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

class LR_forecaster:
    def __init__(self, target_col,add_trend = False, trend_type ="component", n_lag=None, lag_transform = None, differencing_number = None, cat_variables = None,
                 box_cox = False, box_cox_lmda = None, box_cox_biasadj= False):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = LinearRegression()
        self.target_col = target_col
        self.n_lag = n_lag
        self.difference = differencing_number
        self.lag_transform = lag_transform
        self.cat_variables = cat_variables
        self.trend = add_trend
        self.trend_type = trend_type
        self.box_cox = box_cox
        self.lmda = box_cox_lmda
        self.biasadj = box_cox_biasadj
        
    
        
    def data_prep(self, df):
        dfc = df.copy()
        if self.cat_variables is not None:
            for col, cat in self.cat_var.items():
                dfc[col] = dfc[col].astype('category')
                dfc[col] = dfc[col].cat.set_categories(cat)
            dfc = pd.get_dummies(dfc)

            for i in self.drop_categ:
                dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)

        if (self.target_col in dfc.columns):

            if self.box_cox == True:
                self.is_zero = np.any(np.array(dfc[self.target_col]) < 1)
                trans_data, self.lmda = box_cox_transform(x = dfc[self.target_col], shift = self.is_zero, box_cox_lmda=self.lmda)
                dfc[self.target_col] = trans_data
            
            if (self.trend ==True):
                self.len = len(dfc)
                self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
                
                if (self.trend_type == "component"):
                    dfc[self.target_col] = dfc[self.target_col]-self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
            if self.difference is not None:
                if self.difference >1:
                    self.last_train = dfc[self.target_col].tolist()[-self.difference:]
                else:
                    self.last_train = dfc[self.target_col].tolist()[-1]
                dfc[self.target_col] = dfc[self.target_col].diff(self.difference)

            if self.n_lag is not None:
                for i in self.n_lag:
                    dfc["lag"+"_"+str(i)] = dfc[self.target_col].shift(i)
                
            if self.lag_transform is not None:
                for n, k in self.lag_transform.items():
                    df_array = np.array(dfc[self.target_col].shift(n))
                    for f in k:
                        if f[0].__name__ == "rolling_quantile":
                            dfc["q_"+str(f[2])+"_"+str(n)+"_"+str(f[1])] = f[0](df_array, f[1], f[2])
                        else:
                            dfc[f[0].__name__+"_"+str(n)+"_"+str(f[1])] = f[0](df_array, f[1]) 
                            
        if (self.trend ==True) & (self.trend_type == "feature"):

            if (self.target_col in dfc.columns):
                dfc["trend"] = self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
        dfc = dfc.dropna()

        return dfc

    def fit(self, df):

        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        model_train = self.data_prep(df)
        
        self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
        self.model_LR = self.model.fit(np.array(self.X), self.y)

    def forecast(self, n_ahead, x_test = None):
        if x_test is not None:
            if isinstance(x_test, pd.Series):
                x_test = x_test.to_frame().T
            x_dummy = self.data_prep(x_test)
        lags = self.y.tolist()
        predictions = []

        if self.trend ==True:
            trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))

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
                transform_lag = []    
                for n, k in self.lag_transform.items():
                    df_array = np.array(pd.Series(lags).shift(n-1))
                    for f in k:
                        if f[0].__name__ == "rolling_quantile":
                            t1 = f[0](df_array, f[1], f[2])[-1]
                        else:
                            t1 = f[0](df_array, f[1])[-1]
                        transform_lag.append(t1)
            else:
                transform_lag = []
                
            if (self.trend ==True) & (self.trend_type == "feature"):
                trend_var = [trend_pred[i]]
            else:
                trend_var = []
                    
                    
            inp = x_var+inp_lag+transform_lag+trend_var
            # df_inp = pd.DataFrame(inp).T
            # df_inp.columns = self.X.columns

            pred = self.model_LR.predict(np.array(inp).reshape(1,-1))[0]
            predictions.append(pred)
            lags.append(pred)

        if self.difference is not None:
            if self.difference>1:
                predictions_ = self.last_train+predictions
                for i in range(len(predictions_)):
                    if i<len(predictions_)-self.difference:
                        predictions_[i+self.difference] = predictions_[i]+predictions_[i+self.difference]
                        forecasts = predictions_[-n_ahead:]
            else:    
                predictions.insert(0, self.last_train)
                forecasts = np.cumsum(predictions)[-n_ahead:]
        else:
            forecasts = np.array(predictions)
        if (self.trend ==True)&(self.trend_type =="component"):
            forecasts = trend_pred+forecasts    
        forecasts = np.array([max(0, x) for x in forecasts])      
        if self.box_cox == True:
            forecasts = back_box_cox_transform(y_pred = forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)
        return forecasts
        
    
    def cv(self, df, cv_split, test_size, metrics):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_forecats_df = pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

            forecat_df = test[self.target_col].to_frame()
            forecat_df["forecasts"] = bb_forecast
            
            self.cv_forecats_df = pd.concat([self.cv_forecats_df, forecat_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})
    
    def tune_model(self, df, cv_split, test_size, param_space, eval_metric,opt_horizon =None, eval_num= 100):
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        # if 'lags' not in param_space:
        #     self.data_prep(df)

        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            if ('n_lag' in params)|('box_cox' in params)|('box_cox_lmda' in params)|('box_cox_biasadj' in params):
                if ('n_lag' in params):
                    if type(params["n_lag"]) is tuple:
                        self.n_lag = list(params["n_lag"])
                    else:
                        self.n_lag = range(1, params["n_lag"]+1)
                if ('box_cox' in params):
                    self.box_cox = params["box_cox"]
                if ('box_cox_lmda' in params):
                    self.lmda = params["box_cox_lmda"]

                if ('box_cox_biasadj' in params):
                    self.biasadj = params["box_cox_biasadj"]

                
            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])

                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]

                self.model_LR = self.model.fit(np.array(self.X), self.y)
                yhat = self.forecast(n_ahead =len(y_test), x_test=x_test)
                if eval_metric.__name__== 'mean_squared_error':
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:], squared=False)
                    else:
                        accuracy = eval_metric(y_test, yhat, squared=False)

                elif (eval_metric.__name__== 'MeanAbsoluteScaledError')|(eval_metric.__name__== 'MedianAbsoluteScaledError'):
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:], np.array(train[self.target_col]))
                    else:
                        accuracy = eval_metric(y_test, yhat, np.array(train[self.target_col]))
                else:
                    if opt_horizon is not None:
                        accuracy = eval_metric(y_test[-opt_horizon:], yhat[-opt_horizon:])
                    else:
                        accuracy = eval_metric(y_test, yhat)
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

    def direct_forecast(self, H, x_test = None):
        if x_test is not None:
            if isinstance(x_test, pd.Series):
                x_test = x_test.to_frame().T
            x_dummy = self.data_prep(x_test)
#         max_lag = self.n_lag[-1]
#         lags = self.y[-max_lag:].tolist()
        lags = self.y.tolist()

        if self.trend ==True:
            trend_pred = self.lr_model.predict(np.array([self.len+H]).reshape(-1, 1))[0]

        if x_test is not None:
            x_var = x_dummy.iloc[0, 0:].tolist()
        else:
            x_var = []
            
        if self.n_lag is not None:
            new_lag = [i-self.n_lag[0]+1 for i in self.n_lag]
            inp_lag = [lags[-l] for l in new_lag] # to get defined lagged variables 
        else:
            inp_lag = []

        if self.lag_transform is not None:
            transform_lag = []    
            for n, k in self.lag_transform.items():
                df_array = np.array(pd.Series(lags).shift(n-1))
                for f in k:
                    if f[0].__name__ == "rolling_quantile":
                        t1 = f[0](df_array, f[1], f[2])[-1]
                    else:
                        t1 = f[0](df_array, f[1])[-1]
                    transform_lag.append(t1)
        else:
            transform_lag = []
            
        if (self.trend ==True) & (self.trend_type == "feature"):
            trend_var = [trend_pred]
        else:
            trend_var = []
                
                
        inp = x_var+inp_lag+transform_lag+trend_var
        # df_inp = pd.DataFrame(inp).T
        # df_inp.columns = self.X.columns

        pred = self.model_LR.predict(np.array(inp).reshape(1,-1))[0]

        if (self.trend ==True)&(self.trend_type =="component"):
            forecasts = trend_pred+pred
        else:
            forecasts = pred
            
        return forecasts

    def multiple_direct_forecats(self, train, test, H):
        idx = train[-H:].index # get index of last H values of train data to iterate over H horizons to forecast all values of 
        preds = []
        for i in range(H):
            train_direct= train[:idx[i]] # set new train data to forecast H step further
            test_direct = test.loc[idx[i]+timedelta(H)] # subset test row wich corresponds to H step further from the last value of train data 
            self.fit(train_direct)
            forecast = self.direct_forecast(H, x_test=test_direct)
            preds.append(forecast)
            
        if self.difference is not None:
            if self.difference>1:
                predictions_ = self.last_train+preds
                for i in range(len(predictions_)):
                    if i<len(predictions_)-self.difference:
                        predictions_[i+self.difference] = predictions_[i]+predictions_[i+self.difference]
                        multi_forecasts = predictions_[-H:]
            else:    
                preds.insert(0, self.last_train)
                multi_forecasts = np.cumsum(preds)[-H:]
        else:
            multi_forecasts = np.array(preds)

        if self.box_cox == True:
            forecasts = back_box_cox_transform(y_pred = multi_forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)

        return multi_forecasts
            
            
    def cv_direct(self, df, cv_split, H, metrics):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=H)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        # self.cv_df_direct = pd.DataFrame()
        self.cv_forecats_df = pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            bb_forecast = self.multiple_direct_forecats(train, x_test, H)
            forecat_df = test[self.target_col].to_frame()
            forecat_df["forecasts"] = bb_forecast
            
            self.cv_forecats_df = pd.concat([self.cv_forecats_df, forecat_df], axis=0)

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})

    def tune_direct_model(self, df, cv_split, test_size, param_space, eval_metric, eval_num= 100):
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        # if 'lags' not in param_space:
        #     self.data_prep(df)

        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            if ('n_lag' in params)|('box_cox' in params)|('box_cox_lmda' in params)|('box_cox_biasadj' in params):
                if ('n_lag' in params):
                    if type(params["n_lag"]) is tuple:
                        self.n_lag = list(params["n_lag"])
                    else:
                        self.n_lag = range(1, params["n_lag"]+1)
                if ('box_cox' in params):
                    self.box_cox = params["box_cox"]
                if ('box_cox_lmda' in params):
                    self.lmda = params["box_cox_lmda"]

                if ('box_cox_biasadj' in params):
                    self.biasadj = params["box_cox_biasadj"]
                
            # metric = []
            actuals = []
            predictions = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test = test.iloc[:, 1:]


                actuals += test[self.target_col].tolist()
                predictions += list(self.multiple_direct_forecats(train, x_test, test_size, param_model))

            if eval_metric.__name__== 'mean_squared_error':
                accuracy = eval_metric(np.array(actuals), np.array(predictions), squared=False)
            elif (eval_metric.__name__== 'MeanAbsoluteScaledError')|(eval_metric.__name__== 'MedianAbsoluteScaledError'):
                accuracy = eval_metric(np.array(actuals), np.array(predictions), np.array(train[self.target_col]))
            else:
                accuracy = eval_metric(np.array(actuals), np.array(predictions))
#                 print(str(accuracy)+" and len is "+str(len(test)))
            # metric.append(accuracy)
            score = np.mean(accuracy)

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
    
class lightGBM_bidirect_forecaster:
    def __init__(self, target_col, n_lag = None, difference_1 = None, difference_2 = None, lag_transform = None, cat_variables = None,
                 trend1 = False, trend2 = False, trend_type = "component"):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = LGBMRegressor
        self.target_col = target_col
        self.cat_var = cat_variables
        self.n_lag = n_lag
        self.difference1 = difference_1
        self.difference2 = difference_2
        self.lag_transform = lag_transform
        self.trend1 = trend1
        self.trend2 = trend2
        self.trend_type = trend_type
        
    def data_prep(self, df):
        dfc = df.copy()

        if (self.trend1 ==True):
            self.len = len(dfc)
            self.lr_model1 = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col[0]])
            
            if (self.trend_type == "component"):
                dfc[self.target_col[0]] = dfc[self.target_col[0]]-self.lr_model1.predict(np.array(range(self.len)).reshape(-1, 1))

        if (self.trend2 ==True):
            self.len = len(dfc)
            self.lr_model2 = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col[1]])
            
            if (self.trend_type == "component"):
                dfc[self.target_col[1]] = dfc[self.target_col[1]]-self.lr_model2.predict(np.array(range(self.len)).reshape(-1, 1))

        if self.difference1 is not None:
            self.last_tar1 = df[self.target_col[0]].tolist()[-1]
            for i in range(1, self.difference1+1):
                dfc[self.target_col[0]] = dfc[self.target_col[0]].diff(1)
                
        if self.difference2 is not None:
            self.last_tar2 = df[self.target_col[1]].tolist()[-1]
            for i in range(1, self.difference2+1):
                dfc[self.target_col[1]] = dfc[self.target_col[1]].diff(1)
        
        if self.cat_var is not None:
            for c in self.cat_var:
                dfc[c] = dfc[c].astype('category')
                
        if self.n_lag is not None:
            for i in self.n_lag:
                dfc[self.target_col[0]+"_lag"+"_"+str(i)] = dfc[self.target_col[0]].shift(i)
                
            for i in self.n_lag:
                dfc[self.target_col[1]+"_lag"+"_"+str(i)] = dfc[self.target_col[1]].shift(i)
            
        if self.lag_transform is not None:
            df_array = np.array(dfc[self.target_col])
            for i, j in self.lag_transform.items():
                dfc[i.__name__+"_"+str(j)] = i(df_array, j)
            
        dfc = dfc.dropna()
        return dfc
    
    def fit(self, df, param = None):

        if param is not None:
            model_lgb1 = self.model(**param, verbose=-1)
            model_lgb2 = self.model(**param, verbose=-1)
        else:
            model_lgb1 = self.model(verbose=-1)
            model_lgb2 = self.model(verbose=-1)

        model_df = self.data_prep(df)
        
        self.X, self.y1, self.y2 = model_df.drop(columns =self.target_col), model_df[self.target_col[0]], model_df[self.target_col[1]]
        self.model_lgb1 = model_lgb1.fit(self.X, self.y1, categorical_feature=self.cat_var)
        self.model_lgb2 = model_lgb2.fit(self.X, self.y2, categorical_feature=self.cat_var)
    
    def forecast(self, n_ahead, x_test = None):
        tar1_lags = self.y1.tolist()
        tar2_lags = self.y2.tolist()
        tar1_predictions = []
        tar2_predictions = []

        if self.trend1 ==True:
            trend_pred1= self.lr_model1.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))

        if self.trend2 ==True:
            trend_pred2= self.lr_model2.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))
        
        for i in range(n_ahead):
            if x_test is not None:
                x_var = x_test.iloc[i, 0:].tolist()       
            else:
                x_var = []
            if self.n_lag is not None:
                inp_lag1 = [tar1_lags[-i] for i in self.n_lag]
                inp_lag2 = [tar2_lags[-i] for i in self.n_lag]
            else:
                inp_lag1 = []
                inp_lag2 = []
                
            if self.lag_transform is not None:
                lag_array = np.array(tar1_lags) # array is needed for transformation fuctions
                transform_lag = []
                for method, lag in self.lag_transform.items():
                    tl = method(lag_array, lag)[-1]
                    transform_lag.append(tl)
            else:
                transform_lag = []

            inp = x_var + inp_lag1+inp_lag2+transform_lag
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

            pred1 = self.model_lgb1.predict(df_inp)[0]
            tar1_predictions.append(pred1)
            tar1_lags.append(pred1)
            
            pred2 = self.model_lgb2.predict(df_inp)[0]
            tar2_predictions.append(pred2)
            tar2_lags.append(pred2)
            
        if self.difference1 is not None:
            tar1_predictions.insert(0, self.last_tar1)
            forecast1 = np.cumsum(tar1_predictions)[-n_ahead:]
        else:
            forecast1 = np.array(tar1_predictions)
            
        if self.difference2 is not None:
            tar2_predictions.insert(0, self.last_tar2)
            forecast2 = np.cumsum(tar2_predictions)[-n_ahead:]
        else:
            forecast2 = np.array(tar2_predictions)

        if (self.trend1 ==True)&(self.trend_type =="component"):
            forecast1 = trend_pred1+forecast1

        if (self.trend2 ==True)&(self.trend_type =="component"):
            forecast2 = trend_pred2+forecast2

        forecast1 = np.array([max(0, x) for x in forecast1])  
        forecast2 = np.array([max(0, x) for x in forecast2])  
            
        return forecast1, forecast2
    
    def cv(self, df, forecast_idx, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_df = pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col[forecast_idx]])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)[forecast_idx]

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col[forecast_idx]]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

            if forecast_idx == 0:
                cv_tr_df = pd.DataFrame({"feat_name":self.model_lgb1.feature_name_, "importance":self.model_lgb1.feature_importances_}).sort_values(by = "importance", ascending = False)
            else:
                cv_tr_df = pd.DataFrame({"feat_name":self.model_lgb2.feature_name_, "importance":self.model_lgb2.feature_importances_}).sort_values(by = "importance", ascending = False)
            cv_tr_df["fold"] = i
            self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})
    
    def tune_model(self, df, forecast_col, cv_split, test_size, param_space, eval_metric, eval_num = 100):
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
    
class xgboost_bidirect_forecaster:
    def __init__(self, target_col, n_lag = None, difference_1 = None, difference_2 = None, lag_transform = None, 
                 cat_variables = None, trend1  = False, trend2= False, trend_type ="component"):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = XGBRegressor
        self.target_col = target_col
        self.n_lag = n_lag
        self.difference1 = difference_1
        self.difference2 = difference_2
        self.lag_transform = lag_transform
        self.cat_variables = cat_variables
        self.trend1 = trend1
        self.trend2 = trend2
        self.trend_type = trend_type
    
        
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
                
        if (self.target_col[0] in dfc.columns) | (self.target_col[1] in dfc.columns):

            if (self.trend1 ==True):
                self.len = len(dfc)
                self.lr_model1 = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col[0]])
                
                if (self.trend_type == "component"):
                    dfc[self.target_col[0]] = dfc[self.target_col[0]]-self.lr_model1.predict(np.array(range(self.len)).reshape(-1, 1))

            if (self.trend2 ==True):
                self.len = len(dfc)
                self.lr_model2 = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col[1]])
                
                if (self.trend_type == "component"):
                    dfc[self.target_col[1]] = dfc[self.target_col[1]]-self.lr_model2.predict(np.array(range(self.len)).reshape(-1, 1))
                    
            if self.difference1 is not None:
                self.last_tar1 = df[self.target_col[0]].tolist()[-1]
                for i in range(1, self.difference1+1):
                    dfc[self.target_col[0]] = dfc[self.target_col[0]].diff(1)
                    
            if self.difference2 is not None:
                self.last_tar2 = df[self.target_col[1]].tolist()[-1]
                for i in range(1, self.difference2+1):
                    dfc[self.target_col[1]] = dfc[self.target_col[1]].diff(1)
                    
            if self.n_lag is not None:
                for i in self.n_lag:
                    dfc[self.target_col[0]+"_lag"+"_"+str(i)] = dfc[self.target_col[0]].shift(i)
                    
                for i in self.n_lag:
                    dfc[self.target_col[1]+"_lag"+"_"+str(i)] = dfc[self.target_col[1]].shift(i)

                
            if self.lag_transform is not None:
                df_array = np.array(dfc[self.target_col])
                for i, j in self.lag_transform.items():
                    dfc[i.__name__+"_"+str(j)] = i(df_array, j)    
        dfc = dfc.dropna()
        return dfc

    def fit(self, df, param = None):
        if param is not None:
            model_xgb1 =self.model(**param)
            model_xgb2 =self.model(**param)
        else:
            model_xgb1 =self.model()
            model_xgb2 =self.model()
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        model_df = self.data_prep(df)
        self.X, self.y1, self.y2 = model_df.drop(columns =self.target_col), model_df[self.target_col[0]], model_df[self.target_col[1]]
        self.model_xgb1 = model_xgb1.fit(self.X, self.y1, verbose = True)
        self.model_xgb2 = model_xgb2.fit(self.X, self.y2, verbose = True)

    def forecast(self, n_ahead, x_test = None):
        if x_test is not None:
            x_dummy = self.data_prep(x_test)
        tar1_lags = self.y1.tolist()
        tar2_lags = self.y2.tolist()
        tar1_predictions = []
        tar2_predictions = []

        if self.trend1 ==True:
            trend_pred1= self.lr_model1.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))

        if self.trend2 ==True:
            trend_pred2= self.lr_model2.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))
        
        for i in range(n_ahead):
            if x_test is not None:
                x_var = x_dummy.iloc[i, 0:].tolist()
            else:
                x_var = []
                
            if self.n_lag is not None:
                inp_lag1 = [tar1_lags[-i] for i in self.n_lag]
                inp_lag2 = [tar2_lags[-i] for i in self.n_lag]
            else:
                inp_lag1 = []
                inp_lag2 = []
            if self.lag_transform is not None:
                lag_array = np.array(tar1_lags) # array is needed for transformation fuctions
                transform_lag = []
                for method, lag in self.lag_transform.items():
                    tl = method(lag_array, lag)[-1]
                    transform_lag.append(tl)
            else:
                transform_lag = []
                    
                    
            inp = x_var + inp_lag1+inp_lag2+transform_lag
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns

            pred1 = self.model_xgb1.predict(df_inp)[0]
            tar1_predictions.append(pred1)
            tar1_lags.append(pred1)

            pred2 = self.model_xgb2.predict(df_inp)[0]
            tar2_predictions.append(pred2)
            tar2_lags.append(pred2)

        if self.difference1 is not None:
            tar1_predictions.insert(0, self.last_tar1)
            forecast1 = np.cumsum(tar1_predictions)[-n_ahead:]
        else:
            forecast1 = np.array(tar1_predictions)
            
        if self.difference2 is not None:
            tar2_predictions.insert(0, self.last_tar2)
            forecast2 = np.cumsum(tar2_predictions)[-n_ahead:]
        else:
            forecast2 = np.array(tar2_predictions)

        if (self.trend1 ==True)&(self.trend_type =="component"):
            forecast1 = trend_pred1+forecast1

        if (self.trend2 ==True)&(self.trend_type =="component"):
            forecast2 = trend_pred2+forecast2

        forecast1 = np.array([max(0, x) for x in forecast1])  
        forecast2 = np.array([max(0, x) for x in forecast2])  

        return forecast1, forecast2
    
    def cv(self, df, forecast_idx, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {i.__name__:[] for i in metrics}
        self.cv_df = pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col[forecast_idx]])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)[forecast_idx]

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col[forecast_idx]]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

            if forecast_idx == 0:
                cv_tr_df = pd.DataFrame({"feat_name":self.model_xgb1.feature_names_in_, "importance":self.model_xgb1.feature_importances_}).sort_values(by = "importance", ascending = False)
            else:
                cv_tr_df = pd.DataFrame({"feat_name":self.model_xgb2.feature_names_in_, "importance":self.model_xgb2.feature_importances_}).sort_values(by = "importance", ascending = False)
            cv_tr_df["fold"] = i
            self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})

    
    def tune_model(self, df, forecast_col, cv_split, test_size, param_space, eval_metric, eval_num = 100):

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
                self.model_xgb1 = model1.fit(self.X, self.y1, verbose = True)
                self.model_xgb2 = model2.fit(self.X, self.y2, verbose = True)
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
    
class RandomForest_bidirect_forecaster:
    def __init__(self, target_col, n_lag = None, difference_1 = None, difference_2 = None, lag_transform = None, 
                 cat_variables = None, trend1  = False, trend2= False, trend_type ="component"):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = RandomForestRegressor
        self.target_col = target_col
        self.n_lag = n_lag
        self.difference1 = difference_1
        self.difference2 = difference_2
        self.lag_transform = lag_transform
        self.cat_variables = cat_variables
        self.trend1 = trend1
        self.trend2 = trend2
        self.trend_type = trend_type

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
                
        if (self.target_col[0] in dfc.columns) | (self.target_col[1] in dfc.columns):

            if (self.trend1 ==True):
                self.len = len(dfc)
                self.lr_model1 = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col[0]])
                
                if (self.trend_type == "component"):
                    dfc[self.target_col[0]] = dfc[self.target_col[0]]-self.lr_model1.predict(np.array(range(self.len)).reshape(-1, 1))

            if (self.trend2 ==True):
                self.len = len(dfc)
                self.lr_model2 = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col[1]])
                
                if (self.trend_type == "component"):
                    dfc[self.target_col[1]] = dfc[self.target_col[1]]-self.lr_model2.predict(np.array(range(self.len)).reshape(-1, 1))
                    
            if self.difference1 is not None:
                self.last_tar1 = df[self.target_col[0]].tolist()[-1]
                for i in range(1, self.difference1+1):
                    dfc[self.target_col[0]] = dfc[self.target_col[0]].diff(1)
                    
            if self.difference2 is not None:
                self.last_tar2 = df[self.target_col[1]].tolist()[-1]
                for i in range(1, self.difference2+1):
                    dfc[self.target_col[1]] = dfc[self.target_col[1]].diff(1)
                    
            if self.n_lag is not None:
                for i in self.n_lag:
                    dfc[self.target_col[0]+"_lag"+"_"+str(i)] = dfc[self.target_col[0]].shift(i)
                    
                for i in self.n_lag:
                    dfc[self.target_col[1]+"_lag"+"_"+str(i)] = dfc[self.target_col[1]].shift(i)

                
            if self.lag_transform is not None:
                df_array = np.array(dfc[self.target_col])
                for i, j in self.lag_transform.items():
                    dfc[i.__name__+"_"+str(j)] = i(df_array, j)    
        dfc = dfc.dropna()
        return dfc

    def fit(self, df, param = None):
        if param is not None:
            model_rf1 =self.model(**param)
            model_rf2 =self.model(**param)
        else:
            model_rf1 =self.model()
            model_rf2 =self.model()
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        model_df = self.data_prep(df)
        self.X, self.y1, self.y2 = model_df.drop(columns =self.target_col), model_df[self.target_col[0]], model_df[self.target_col[1]]
        self.model_rf1 = model_rf1.fit(self.X, self.y1)
        self.model_rf2 = model_rf2.fit(self.X, self.y2)

    def forecast(self, n_ahead, x_test = None):
        if x_test is not None:
            x_dummy = self.data_prep(x_test)
        tar1_lags = self.y1.tolist()
        tar2_lags = self.y2.tolist()
        tar1_predictions = []
        tar2_predictions = []

        if self.trend1 ==True:
            trend_pred1= self.lr_model1.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))

        if self.trend2 ==True:
            trend_pred2= self.lr_model2.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))
        
        for i in range(n_ahead):
            if x_test is not None:
                x_var = x_dummy.iloc[i, 0:].tolist()
            else:
                x_var = []
                
            if self.n_lag is not None:
                inp_lag1 = [tar1_lags[-i] for i in self.n_lag]
                inp_lag2 = [tar2_lags[-i] for i in self.n_lag]
            else:
                inp_lag1 = []
                inp_lag2 = []
            if self.lag_transform is not None:
                lag_array = np.array(tar1_lags) # array is needed for transformation fuctions
                transform_lag = []
                for method, lag in self.lag_transform.items():
                    tl = method(lag_array, lag)[-1]
                    transform_lag.append(tl)
            else:
                transform_lag = []
                    
                    
            inp = x_var + inp_lag1+inp_lag2+transform_lag
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns

            pred1 = self.model_rf1.predict(df_inp)[0]
            tar1_predictions.append(pred1)
            tar1_lags.append(pred1)

            pred2 = self.model_rf2.predict(df_inp)[0]
            tar2_predictions.append(pred2)
            tar2_lags.append(pred2)

        if self.difference1 is not None:
            tar1_predictions.insert(0, self.last_tar1)
            forecast1 = np.cumsum(tar1_predictions)[-n_ahead:]
        else:
            forecast1 = np.array(tar1_predictions)
            
        if self.difference2 is not None:
            tar2_predictions.insert(0, self.last_tar2)
            forecast2 = np.cumsum(tar2_predictions)[-n_ahead:]
        else:
            forecast2 = np.array(tar2_predictions)

        if (self.trend1 ==True)&(self.trend_type =="component"):
            forecast1 = trend_pred1+forecast1

        if (self.trend2 ==True)&(self.trend_type =="component"):
            forecast2 = trend_pred2+forecast2

        forecast1 = np.array([max(0, x) for x in forecast1])  
        forecast2 = np.array([max(0, x) for x in forecast2])  

        return forecast1, forecast2
    
    def cv(self, df, forecast_idx, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {i.__name__:[] for i in metrics}
        self.cv_df = pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col[forecast_idx]])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)[forecast_idx]

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col[forecast_idx]]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

            if forecast_idx == 0:
                cv_tr_df = pd.DataFrame({"feat_name":self.model_rf1.feature_names_in_, "importance":self.model_rf1.feature_importances_}).sort_values(by = "importance", ascending = False)
            else:
                cv_tr_df = pd.DataFrame({"feat_name":self.model_rf2.feature_names_in_, "importance":self.model_rf2.feature_importances_}).sort_values(by = "importance", ascending = False)
            cv_tr_df["fold"] = i
            self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})
    
    def tune_model(self, df, forecast_col, cv_split, test_size, param_space, eval_metric, eval_num = 100):

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
                self.model_rf1 = model1.fit(self.X, self.y1)
                self.model_rf2 = model2.fit(self.X, self.y2)
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
    
class cat_bidirect_forecaster:
    def __init__(self, target_col, n_lag = None, difference_1 = None, difference_2 = None, lag_transform = None, cat_variables = None,
                 trend1  = False, trend2= False, trend_type ="component"):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = CatBoostRegressor
        self.target_col = target_col
        self.cat_var = cat_variables
        self.n_lag = n_lag
        self.difference1 = difference_1
        self.difference2 = difference_2
        self.lag_transform = lag_transform
        self.trend1 = trend1
        self.trend2 = trend2
        self.trend_type = trend_type
        
    def data_prep(self, df):
        dfc = df.copy()
        # self.raw_df = df.copy()
        if (self.trend1 ==True):
            self.len = len(dfc)
            self.lr_model1 = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col[0]])
            
            if (self.trend_type == "component"):
                dfc[self.target_col[0]] = dfc[self.target_col[0]]-self.lr_model1.predict(np.array(range(self.len)).reshape(-1, 1))

        if (self.trend2 ==True):
            self.len = len(dfc)
            self.lr_model2 = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col[1]])
            
            if (self.trend_type == "component"):
                dfc[self.target_col[1]] = dfc[self.target_col[1]]-self.lr_model2.predict(np.array(range(self.len)).reshape(-1, 1))

        if self.difference1 is not None:
            self.last_tar1 = df[self.target_col[0]].tolist()[-1]
            for i in range(1, self.difference1+1):
                dfc[self.target_col[0]] = dfc[self.target_col[0]].diff(1)
                
        if self.difference2 is not None:
            self.last_tar2 = df[self.target_col[1]].tolist()[-1]
            for i in range(1, self.difference2+1):
                dfc[self.target_col[1]] = dfc[self.target_col[1]].diff(1)
        
        if self.cat_var is not None:
            for c in self.cat_var:
                dfc[c] = dfc[c].astype('category')
                
        if self.n_lag is not None:
            for i in self.n_lag:
                dfc[self.target_col[0]+"_lag"+"_"+str(i)] = dfc[self.target_col[0]].shift(i)
                
            for i in self.n_lag:
                dfc[self.target_col[1]+"_lag"+"_"+str(i)] = dfc[self.target_col[1]].shift(i)
            
        if self.lag_transform is not None:
            df_array = np.array(dfc[self.target_col])
            for i, j in self.lag_transform.items():
                dfc[i.__name__+"_"+str(j)] = i(df_array, j)
            
        dfc = dfc.dropna()
        return dfc

    def fit(self, df, param = None):

        if param is not None:
            model_cat1 = self.model(**param)
            model_cat2 = self.model(**param)
        else:
            model_cat1 = self.model()
            model_cat2 = self.model()

        model_df = self.data_prep(df)
        
        self.X, self.y1, self.y2 = model_df.drop(columns =self.target_col), model_df[self.target_col[0]], model_df[self.target_col[1]]
        self.model_cat1 = model_cat1.fit(self.X, self.y1, cat_features=self.cat_var)
        self.model_cat2 = model_cat2.fit(self.X, self.y2, cat_features=self.cat_var)
    

    def forecast(self, n_ahead, x_test = None):
        tar1_lags = self.y1.tolist()
        tar2_lags = self.y2.tolist()
        tar1_predictions = []
        tar2_predictions = []

        if self.trend1 ==True:
            trend_pred1= self.lr_model1.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))

        if self.trend2 ==True:
            trend_pred2= self.lr_model2.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))
        
        for i in range(n_ahead):
            if x_test is not None:
                x_var = x_test.iloc[i, 0:].tolist()       
            else:
                x_var = []
            if self.n_lag is not None:
                inp_lag1 = [tar1_lags[-i] for i in self.n_lag]
                inp_lag2 = [tar2_lags[-i] for i in self.n_lag]
            else:
                inp_lag1 = []
                inp_lag2 = []
                
            if self.lag_transform is not None:
                lag_array = np.array(lags) # array is needed for transformation fuctions
                transform_lag = []
                for method, lag in self.lag_transform.items():
                    tl = method(lag_array, lag)[-1]
                    transform_lag.append(tl)
            else:
                transform_lag = []

            inp = x_var + inp_lag1+inp_lag2+transform_lag
            
            pred1 = self.model_cat1.predict(inp)
            tar1_predictions.append(pred1)
            tar1_lags.append(pred1)
            
            pred2 = self.model_cat2.predict(inp)
            tar2_predictions.append(pred2)
            tar2_lags.append(pred2)
            
        if self.difference1 is not None:
            tar1_predictions.insert(0, self.last_tar1)
            forecast1 = np.cumsum(tar1_predictions)[-n_ahead:]
        else:
            forecast1 = np.array(tar1_predictions)
            
        if self.difference2 is not None:
            tar2_predictions.insert(0, self.last_tar2)
            forecast2 = np.cumsum(tar2_predictions)[-n_ahead:]
        else:
            forecast2 = np.array(tar2_predictions)

        if (self.trend1 ==True)&(self.trend_type =="component"):
            forecast1 = trend_pred1+forecast1

        if (self.trend2 ==True)&(self.trend_type =="component"):
            forecast2 = trend_pred2+forecast2

        forecast1 = np.array([max(0, x) for x in forecast1])  
        forecast2 = np.array([max(0, x) for x in forecast2])  
            
        return forecast1, forecast2
    
    def cv(self, df, forecast_idx, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {i.__name__:[] for i in metrics}
        # self.cv_df = pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col[forecast_idx]])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)[forecast_idx]

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col[forecast_idx]]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

            # if forecast_idx == 0:
            #     cv_tr_df = pd.DataFrame({"feat_name":self.model_lgb1.feature_name_, "importance":self.model_lgb1.feature_importances_}).sort_values(by = "importance", ascending = False)
            # else:
            #     cv_tr_df = pd.DataFrame({"feat_name":self.model_lgb2.feature_name_, "importance":self.model_lgb2.feature_importances_}).sort_values(by = "importance", ascending = False)
            # cv_tr_df["fold"] = i
            # self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})

            
    def tune_model(self, df, forecast_col, cv_split, test_size, param_space, eval_metric, eval_num = 100):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)

        def objective(params):
            model1 =self.model(**params)
            model2 =self.model(**params)

            
            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.drop(columns = self.target_col), np.array(test[forecast_col])
                model_train = self.data_prep(train)
                self.X, self.y1, self.y2 = model_train.drop(columns =self.target_col), model_train[self.target_col[0]], model_train[self.target_col[1]]
                self.model_cat1 = model1.fit(self.X, self.y1, cat_features=self.cat_var,
                            verbose = False)
                self.model_cat2 = model2.fit(self.X, self.y2, cat_features=self.cat_var,
                            verbose = False)
                
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
    
class Cubist_bidirect_forecaster:
    def __init__(self, target_col, n_lag = None, difference_1 = None, difference_2 = None, lag_transform = None, 
                 cat_variables = None, trend1  = False, trend2= False, trend_type ="component"):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = Cubist
        self.target_col = target_col
        self.n_lag = n_lag
        self.difference1 = difference_1
        self.difference2 = difference_2
        self.lag_transform = lag_transform
        self.cat_variables = cat_variables
        self.trend1 = trend1
        self.trend2 = trend2
        self.trend_type = trend_type

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
                
        if (self.target_col[0] in dfc.columns) | (self.target_col[1] in dfc.columns):

            if (self.trend1 ==True):
                self.len = len(dfc)
                self.lr_model1 = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col[0]])
                
                if (self.trend_type == "component"):
                    dfc[self.target_col[0]] = dfc[self.target_col[0]]-self.lr_model1.predict(np.array(range(self.len)).reshape(-1, 1))

            if (self.trend2 ==True):
                self.len = len(dfc)
                self.lr_model2 = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col[1]])
                
                if (self.trend_type == "component"):
                    dfc[self.target_col[1]] = dfc[self.target_col[1]]-self.lr_model2.predict(np.array(range(self.len)).reshape(-1, 1))
                    
            if self.difference1 is not None:
                self.last_tar1 = df[self.target_col[0]].tolist()[-1]
                for i in range(1, self.difference1+1):
                    dfc[self.target_col[0]] = dfc[self.target_col[0]].diff(1)
                    
            if self.difference2 is not None:
                self.last_tar2 = df[self.target_col[1]].tolist()[-1]
                for i in range(1, self.difference2+1):
                    dfc[self.target_col[1]] = dfc[self.target_col[1]].diff(1)
                    
            if self.n_lag is not None:
                for i in self.n_lag:
                    dfc[self.target_col[0]+"_lag"+"_"+str(i)] = dfc[self.target_col[0]].shift(i)
                    
                for i in self.n_lag:
                    dfc[self.target_col[1]+"_lag"+"_"+str(i)] = dfc[self.target_col[1]].shift(i)

                
            if self.lag_transform is not None:
                df_array = np.array(dfc[self.target_col])
                for i, j in self.lag_transform.items():
                    dfc[i.__name__+"_"+str(j)] = i(df_array, j)    
        dfc = dfc.dropna()
        return dfc

    def fit(self, df, param = None):
        if param is not None:
            model_cub1 =self.model(**param)
            model_cub2 =self.model(**param)
        else:
            model_cub1 =self.model()
            model_cub2 =self.model()
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        model_df = self.data_prep(df)
        self.X, self.y1, self.y2 = model_df.drop(columns =self.target_col), model_df[self.target_col[0]], model_df[self.target_col[1]]
        self.model_cub1 = model_cub1.fit(self.X, self.y1)
        self.model_cub2 = model_cub2.fit(self.X, self.y2)

    def forecast(self, n_ahead, x_test = None):
        if x_test is not None:
            x_dummy = self.data_prep(x_test)
        tar1_lags = self.y1.tolist()
        tar2_lags = self.y2.tolist()
        tar1_predictions = []
        tar2_predictions = []

        if self.trend1 ==True:
            trend_pred1= self.lr_model1.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))

        if self.trend2 ==True:
            trend_pred2= self.lr_model2.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))
        
        for i in range(n_ahead):
            if x_test is not None:
                x_var = x_dummy.iloc[i, 0:].tolist()
            else:
                x_var = []
                
            if self.n_lag is not None:
                inp_lag1 = [tar1_lags[-i] for i in self.n_lag]
                inp_lag2 = [tar2_lags[-i] for i in self.n_lag]
            else:
                inp_lag1 = []
                inp_lag2 = []
            if self.lag_transform is not None:
                lag_array = np.array(tar1_lags) # array is needed for transformation fuctions
                transform_lag = []
                for method, lag in self.lag_transform.items():
                    tl = method(lag_array, lag)[-1]
                    transform_lag.append(tl)
            else:
                transform_lag = []
                    
                    
            inp1 = x_var + inp_lag1+inp_lag2+transform_lag
            df_inp = pd.DataFrame(inp1).T
            df_inp.columns = self.X.columns

            pred1 = self.model_cub1.predict(df_inp)[0]
            tar1_predictions.append(pred1)
            tar1_lags.append(pred1)

            pred2 = self.model_cub2.predict(df_inp)[0]
            tar2_predictions.append(pred2)
            tar2_lags.append(pred2)

        if self.difference1 is not None:
            tar1_predictions.insert(0, self.last_tar1)
            forecast1 = np.cumsum(tar1_predictions)[-n_ahead:]
        else:
            forecast1 = np.array(tar1_predictions)
            
        if self.difference2 is not None:
            tar2_predictions.insert(0, self.last_tar2)
            forecast2 = np.cumsum(tar2_predictions)[-n_ahead:]
        else:
            forecast2 = np.array(tar2_predictions)

        if (self.trend1 ==True)&(self.trend_type =="component"):
            forecast1 = trend_pred1+forecast1

        if (self.trend2 ==True)&(self.trend_type =="component"):
            forecast2 = trend_pred2+forecast2

        forecast1 = np.array([max(0, x) for x in forecast1])  
        forecast2 = np.array([max(0, x) for x in forecast2])  

        return forecast1, forecast2
    
    def cv(self, df, forecast_idx, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {i.__name__:[] for i in metrics}
        # self.cv_df = pd.DataFrame()

        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col[forecast_idx]])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)[forecast_idx]

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col[forecast_idx]]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

            # if forecast_idx == 0:
            #     cv_tr_df = pd.DataFrame({"feat_name":self.model_rf1.feature_names_in_, "importance":self.model_rf1.feature_importances_}).sort_values(by = "importance", ascending = False)
            # else:
            #     cv_tr_df = pd.DataFrame({"feat_name":self.model_rf2.feature_names_in_, "importance":self.model_rf2.feature_importances_}).sort_values(by = "importance", ascending = False)
            # cv_tr_df["fold"] = i
            # self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})

    
    def tune_model(self, df, forecast_col, cv_split, test_size, param_space, eval_metric, eval_num = 100):

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
                
                self.X, self.y1,  self.y2 = model_train.drop(columns =self.target_col), model_train[self.target_col[0]], model_train[self.target_col[1]]
                self.model_cub1 = model1.fit(self.X, self.y1)
                self.model_cub2 = model2.fit(self.X, self.y2)
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

class AdaBoost_bidirect_forecaster:
    def __init__(self, target_col, n_lag = None, difference_1 = None, difference_2 = None, lag_transform = None, 
                 cat_variables = None,  trend1  = False, trend2= False, trend_type ="component"):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = AdaBoostRegressor
        self.target_col = target_col
        self.n_lag = n_lag
        self.difference1 = difference_1
        self.difference2 = difference_2
        self.lag_transform = lag_transform
        self.cat_variables=cat_variables
        self.trend1 = trend1
        self.trend2 = trend2
        self.trend_type = trend_type

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
                
        if (self.target_col[0] in dfc.columns) | (self.target_col[1] in dfc.columns):

            if (self.trend1 ==True):
                self.len = len(dfc)
                self.lr_model1 = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col[0]])
                
                if (self.trend_type == "component"):
                    dfc[self.target_col[0]] = dfc[self.target_col[0]]-self.lr_model1.predict(np.array(range(self.len)).reshape(-1, 1))

            if (self.trend2 ==True):
                self.len = len(dfc)
                self.lr_model2 = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col[1]])
                
                if (self.trend_type == "component"):
                    dfc[self.target_col[1]] = dfc[self.target_col[1]]-self.lr_model2.predict(np.array(range(self.len)).reshape(-1, 1))
                    
            if self.difference1 is not None:
                self.last_tar1 = df[self.target_col[0]].tolist()[-1]
                for i in range(1, self.difference1+1):
                    dfc[self.target_col[0]] = dfc[self.target_col[0]].diff(1)
                    
            if self.difference2 is not None:
                self.last_tar2 = df[self.target_col[1]].tolist()[-1]
                for i in range(1, self.difference2+1):
                    dfc[self.target_col[1]] = dfc[self.target_col[1]].diff(1)
                    
            if self.n_lag is not None:
                for i in self.n_lag:
                    dfc[self.target_col[0]+"_lag"+"_"+str(i)] = dfc[self.target_col[0]].shift(i)
                    
                for i in self.n_lag:
                    dfc[self.target_col[1]+"_lag"+"_"+str(i)] = dfc[self.target_col[1]].shift(i)

                
            if self.lag_transform is not None:
                df_array = np.array(dfc[self.target_col])
                for i, j in self.lag_transform.items():
                    dfc[i.__name__+"_"+str(j)] = i(df_array, j)    
        dfc = dfc.dropna()
        return dfc

    def fit(self, df, param = None):
        if param is not None:
            model_ada1 =self.model(**param)
            model_ada2 =self.model(**param)
        else:
            model_ada1 =self.model()
            model_ada2 =self.model()
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        model_df = self.data_prep(df)
        self.X, self.y1, self.y2 = model_df.drop(columns =self.target_col), model_df[self.target_col[0]], model_df[self.target_col[1]]
        self.model_ada1 = model_ada1.fit(self.X, self.y1)
        self.model_ada2 = model_ada2.fit(self.X, self.y2)

    def forecast(self, n_ahead, x_test = None):
        if x_test is not None:
            x_dummy = self.data_prep(x_test)
        tar1_lags = self.y1.tolist()
        tar2_lags = self.y2.tolist()
        tar1_predictions = []
        tar2_predictions = []

        if self.trend1 ==True:
            trend_pred1= self.lr_model1.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))

        if self.trend2 ==True:
            trend_pred2= self.lr_model2.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))
        
        for i in range(n_ahead):
            if x_test is not None:
                x_var = x_dummy.iloc[i, 0:].tolist()
            else:
                x_var = []
                
            if self.n_lag is not None:
                inp_lag1 = [tar1_lags[-i] for i in self.n_lag]
                inp_lag2 = [tar2_lags[-i] for i in self.n_lag]
            else:
                inp_lag1 = []
                inp_lag2 = []
            if self.lag_transform is not None:
                lag_array = np.array(tar1_lags) # array is needed for transformation fuctions
                transform_lag = []
                for method, lag in self.lag_transform.items():
                    tl = method(lag_array, lag)[-1]
                    transform_lag.append(tl)
            else:
                transform_lag = []
                    
                    
            inp = x_var + inp_lag1+inp_lag2+transform_lag
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns

            pred1 = self.model_ada1.predict(df_inp)[0]
            tar1_predictions.append(pred1)
            tar1_lags.append(pred1)

            pred2 = self.model_ada2.predict(df_inp)[0]
            tar2_predictions.append(pred2)
            tar2_lags.append(pred2)

        if self.difference1 is not None:
            tar1_predictions.insert(0, self.last_tar1)
            forecast1 = np.cumsum(tar1_predictions)[-n_ahead:]
        else:
            forecast1 = np.array(tar1_predictions)
            
        if self.difference2 is not None:
            tar2_predictions.insert(0, self.last_tar2)
            forecast2 = np.cumsum(tar2_predictions)[-n_ahead:]
        else:
            forecast2 = np.array(tar2_predictions)

        if (self.trend1 ==True)&(self.trend_type =="component"):
            forecast1 = trend_pred1+forecast1

        if (self.trend2 ==True)&(self.trend_type =="component"):
            forecast2 = trend_pred2+forecast2

        forecast1 = np.array([max(0, x) for x in forecast1])  
        forecast2 = np.array([max(0, x) for x in forecast2])  

        return forecast1, forecast2
    
    def cv(self, df, forecast_idx, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {i.__name__:[] for i in metrics}
        self.cv_df = pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col[forecast_idx]])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)[forecast_idx]

            for m in metrics:
                if m.__name__== 'mean_squared_error':
                    eval = m(y_test, bb_forecast, squared=False)
                elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                    eval = m(y_test, bb_forecast, np.array(train[self.target_col[forecast_idx]]))
                else:
                    eval = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval)

            if forecast_idx == 0:
                cv_tr_df = pd.DataFrame({"feat_name":self.model_ada1.feature_names_in_, "importance":self.model_ada1.feature_importances_}).sort_values(by = "importance", ascending = False)
            else:
                cv_tr_df = pd.DataFrame({"feat_name":self.model_ada2.feature_names_in_, "importance":self.model_ada2.feature_importances_}).sort_values(by = "importance", ascending = False)
            cv_tr_df["fold"] = i
            self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})

    
    def tune_model(self, df, forecast_col, cv_split, test_size, param_space, eval_metric, eval_num = 100):
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