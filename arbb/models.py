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
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from arbb.stats import box_cox_transform, back_box_cox_transform, undiff_ts, seasonal_diff, invert_seasonal_diff
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta

class cat_forecaster:
    def __init__(self, target_col, add_trend = False, trend_type ="linear", ets_params = None, n_lag = None, lag_transform = None,
                 differencing_number = None, seasonal_length = None, cat_variables = None,
                 box_cox = False, box_cox_lmda = None, box_cox_biasadj= False):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = CatBoostRegressor
        self.target_col = target_col
        self.cat_var = cat_variables
        self.n_lag = n_lag
        self.difference = differencing_number
        self.season_diff = seasonal_length
        self.lag_transform = lag_transform
        self.trend = add_trend
        self.trend_type = trend_type
        if ets_params is not None:
            self.ets_model = ets_params[0]
            self.ets_fit = ets_params[1]
        self.box_cox = box_cox
        self.lmda = box_cox_lmda
        self.biasadj = box_cox_biasadj
        self.tuned_params = None
        self.actuals = None
        self.prob_forecasts = None


    def data_prep(self, df):
        dfc = df.copy()
        if self.box_cox == True:
            self.is_zero = np.any(np.array(dfc[self.target_col]) < 1)
            trans_data, self.lmda = box_cox_transform(x = dfc[self.target_col], shift = self.is_zero, box_cox_lmda=self.lmda)
            dfc[self.target_col] = trans_data


        if self.trend ==True:
            self.len = len(dfc)
            if (self.trend_type == "linear") | (self.trend_type == "feature_lr"):
                self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
                if self.trend_type == "linear":
                    dfc[self.target_col] = dfc[self.target_col]-self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))

            if (self.trend_type == "ses")|(self.trend_type == "feature_ses"):
                self.ses_model = ExponentialSmoothing(dfc[self.target_col], **self.ets_model).fit(**self.ets_fit)
                if self.trend_type == "ses":
                    dfc[self.target_col] = dfc[self.target_col]-self.ses_model.fittedvalues.values

        if (self.difference is not None)|(self.season_diff is not None):
            self.orig = dfc[self.target_col].tolist()
            if self.difference is not None:
                dfc[self.target_col] = np.diff(dfc[self.target_col], n= self.difference, prepend=np.repeat(np.nan, self.difference))
                if self.season_diff is not None:
                    self.orig_d = dfc[self.target_col].tolist()
                    dfc[self.target_col] = seasonal_diff(dfc[self.target_col], self.season_diff)

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

        if self.trend ==True:
            if (self.target_col in dfc.columns):
                if self.trend_type == "feature_lr":
                    dfc["trend"] = self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))

                if self.trend_type == "feature_ses":
                    dfc["trend"] = self.ses_model.fittedvalues.values
                
        dfc = dfc.dropna()
        # if self.target_col in dfc.columns:
        #     self.dfc = dfc
        #     self.df =df.loc[dfc.index]
        # else:
        return dfc


    
    def fit(self, df, param = None):
        if param is not None:
            model_ = self.model(**param)
        else:
            model_ = self.model()

        model_df = self.data_prep(df)
        self.X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        # self.data_prep(df)
        # self.X, self.y = self.dfc.drop(columns =self.target_col), self.dfc[self.target_col]
        self.model_fit = model_.fit(self.X, self.y, cat_features=self.cat_var, verbose = True)
    
    def forecast(self, n_ahead, x_test = None):
        lags = self.y.tolist()
        predictions = []

        if self.trend ==True:
            if (self.trend_type == "linear") | (self.trend_type == "feature_lr"):
                trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))

            if (self.trend_type == "ses") | (self.trend_type == "feature_ses"):
                trend_pred = self.ses_model.forecast(n_ahead).values

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

            if (self.trend ==True):
                if (self.trend_type == "feature_lr")|(self.trend_type == "feature_ses"):
                    trend_var = [trend_pred[i]]
                else:
                    trend_var = []
            else:
                trend_var = []

            inp = x_var + inp_lag+transform_lag+trend_var
            pred = self.model_fit.predict(inp)
            predictions.append(pred)
            lags.append(pred)

        
        forecasts = np.array(predictions)
        if self.season_diff is not None:
            forecasts = invert_seasonal_diff(self.orig_d, forecasts, self.season_diff)

        if self.difference is not None:
            forecasts = undiff_ts(self.orig, forecasts, self.difference)

        if (self.trend ==True):
            if (self.trend_type =="linear")|(self.trend_type =="ses"):
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

class lightGBM_forecaster:
    def __init__(self, target_col, add_trend = False, trend_type ="linear", ets_params = None, n_lag = None, lag_transform = None,
                 differencing_number = None,seasonal_length = None, cat_variables = None,
                 box_cox = False, box_cox_lmda = None, box_cox_biasadj= False):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = LGBMRegressor
        self.target_col = target_col
        self.cat_var = cat_variables
        self.n_lag = n_lag
        self.difference = differencing_number
        self.season_diff = seasonal_length
        self.lag_transform = lag_transform
        self.trend = add_trend
        self.trend_type = trend_type
        if ets_params is not None:
            self.ets_model = ets_params[0]
            self.ets_fit = ets_params[1]
        self.box_cox = box_cox
        self.lmda = box_cox_lmda
        self.biasadj = box_cox_biasadj
        self.tuned_params = None
        self.actuals = None
        self.prob_forecasts = None
        
    def data_prep(self, df):
        dfc = df.copy()

        if self.box_cox == True:
            self.is_zero = np.any(np.array(dfc[self.target_col]) < 1)
            trans_data, self.lmda = box_cox_transform(x = dfc[self.target_col], shift = self.is_zero, box_cox_lmda=self.lmda)
            dfc[self.target_col] = trans_data

        if (self.trend ==True):
            self.len = len(dfc)
            if (self.trend_type == "linear") | (self.trend_type == "feature_lr"):
                self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
                if (self.trend_type == "linear"):
                    dfc[self.target_col] = dfc[self.target_col]-self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))

            if (self.trend_type == "ses")|(self.trend_type == "feature_ses"):
                self.ses_model = ExponentialSmoothing(dfc[self.target_col], **self.ets_model).fit(**self.ets_fit)
                if (self.trend_type == "ses"):
                    dfc[self.target_col] = dfc[self.target_col]-self.ses_model.fittedvalues.values

        if (self.difference is not None)|(self.season_diff is not None):
            self.orig = dfc[self.target_col].tolist()
            if self.difference is not None:
                dfc[self.target_col] = np.diff(dfc[self.target_col], n= self.difference, prepend=np.repeat(np.nan, self.difference))
            if self.season_diff is not None:
                self.orig_d = dfc[self.target_col].tolist()
                dfc[self.target_col] = seasonal_diff(dfc[self.target_col], self.season_diff)

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

        if self.trend ==True:
            if (self.target_col in dfc.columns):
                if self.trend_type == "feature_lr":
                    dfc["trend"] = self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))

                if self.trend_type == "feature_ses":
                    dfc["trend"] = self.ses_model.fittedvalues.values
            
        dfc = dfc.dropna()

        return dfc
    
    def fit(self, df, param = None):

        if param is not None:
            model_ = self.model(**param, verbose=-1)
        else:
            model_ = self.model(verbose=-1)

        model_df = self.data_prep(df)
        self.X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        # self.data_prep(df)
        # self.X, self.y = self.dfc.drop(columns =self.target_col), self.dfc[self.target_col]
        self.model_fit = model_.fit(self.X, self.y, categorical_feature=self.cat_var)
  
    
    def forecast(self, n_ahead, x_test = None):
        lags = self.y.tolist()
        predictions = []

        if self.trend ==True:
            if (self.trend_type == "linear") | (self.trend_type == "feature_lr"):
                trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))
            if (self.trend_type == "ses") | (self.trend_type == "feature_ses"):
                trend_pred = self.ses_model.forecast(n_ahead).values

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

            if (self.trend ==True):
                if (self.trend_type == "feature_lr")|(self.trend_type == "feature_ses"):
                    trend_var = [trend_pred[i]]
                else:
                    trend_var = []
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
            pred = self.model_fit.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)

        forecasts = np.array(predictions)
        if self.season_diff is not None:
            forecasts = invert_seasonal_diff(self.orig_d, forecasts, self.season_diff)

        if self.difference is not None:
            forecasts = undiff_ts(self.orig, forecasts, self.difference)

        if (self.trend ==True):
            if (self.trend_type =="linear")|(self.trend_type =="ses"):
                forecasts = trend_pred+forecasts

        forecasts = np.array([max(0, x) for x in forecasts])   
        if self.box_cox == True:
            forecasts = back_box_cox_transform(y_pred = forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)
        return forecasts
    
    def cv(self, df, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_df = pd.DataFrame()
        self.cv_forecats_df= pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)

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

            cv_tr_df = pd.DataFrame({"feat_name":self.model_fit.feature_name_, "importance":self.model_fit.feature_importances_}).sort_values(by = "importance", ascending = False)
            cv_tr_df["fold"] = i
            self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})
            
class xgboost_forecaster:
    def __init__(self, target_col, add_trend = False, trend_type ="linear", ets_params = None, n_lag = None, lag_transform = None,
                 differencing_number = None, seasonal_length = None, cat_variables = None,
                 box_cox = False, box_cox_lmda = None, box_cox_biasadj= False):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = XGBRegressor
        self.target_col = target_col
        self.n_lag = n_lag
        self.difference = differencing_number
        self.season_diff = seasonal_length
        self.lag_transform = lag_transform
        self.cat_variables = cat_variables
        self.trend = add_trend
        self.trend_type = trend_type
        if ets_params is not None:
            self.ets_model = ets_params[0]
            self.ets_fit = ets_params[1]
        self.box_cox = box_cox
        self.lmda = box_cox_lmda
        self.biasadj = box_cox_biasadj
        self.tuned_params = None
        self.actuals = None
        self.prob_forecasts = None

        
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
                if (self.trend_type == "linear") | (self.trend_type == "feature_lr"):
                    self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
                    if (self.trend_type == "linear"):
                        dfc[self.target_col] = dfc[self.target_col]-self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))

                if (self.trend_type == "ses")|(self.trend_type == "feature_ses"):
                    self.ses_model = ExponentialSmoothing(dfc[self.target_col], **self.ets_model).fit(**self.ets_fit)
                    if (self.trend_type == "ses"):
                        dfc[self.target_col] = dfc[self.target_col]-self.ses_model.fittedvalues.values

            if (self.difference is not None)|(self.season_diff is not None):
                self.orig = dfc[self.target_col].tolist()
                if self.difference is not None:
                    dfc[self.target_col] = np.diff(dfc[self.target_col], n= self.difference, prepend=np.repeat(np.nan, self.difference))
                if self.season_diff is not None:
                    self.orig_d = dfc[self.target_col].tolist()
                    dfc[self.target_col] = seasonal_diff(dfc[self.target_col], self.season_diff)

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
                            
        if self.trend ==True:
            if (self.target_col in dfc.columns):
                if self.trend_type == "feature_lr":
                    dfc["trend"] = self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))

                if self.trend_type == "feature_ses":
                    dfc["trend"] = self.ses_model.fittedvalues.values
        dfc = dfc.dropna()


        # if self.target_col in dfc.columns:
        #     self.dfc = dfc
        #     self.df =df.loc[dfc.index]
        # else:
        return dfc


    def fit(self, df, param = None):
        if param is not None:
            model_ =self.model(**param)
        else:
            model_ =self.model()
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        
        model_train = self.data_prep(df)
        self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]

        self.model_fit = model_.fit(self.X, self.y, verbose = True)

    def forecast(self, n_ahead, x_test = None):
        if x_test is not None:
            x_dummy = self.data_prep(x_test)
        lags = self.y.tolist()
        predictions = []

        if self.trend ==True:
            if (self.trend_type == "linear") | (self.trend_type == "feature_lr"):
                trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))
            if (self.trend_type == "ses") | (self.trend_type == "feature_ses"):
                trend_pred = self.ses_model.forecast(n_ahead).values

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
                
            if (self.trend ==True):
                if (self.trend_type == "feature_lr")|(self.trend_type == "feature_ses"):
                    trend_var = [trend_pred[i]]
                else:
                    trend_var = []
            else:
                trend_var = [] 
                    
            inp = x_var+inp_lag+transform_lag+trend_var
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns

            pred = self.model_fit.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)

        forecasts = np.array(predictions)
        if self.season_diff is not None:
            forecasts = invert_seasonal_diff(self.orig_d, forecasts, self.season_diff)

        if self.difference is not None:
            forecasts = undiff_ts(self.orig, forecasts, self.difference)

        if (self.trend ==True):
            if (self.trend_type =="linear")|(self.trend_type =="ses"):
                forecasts = trend_pred+forecasts   
        forecasts = np.array([max(0, x) for x in forecasts])    
        if self.box_cox == True:
            forecasts = back_box_cox_transform(y_pred = forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)
        return forecasts
    
    def cv(self, df, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_df = pd.DataFrame()
        self.cv_forecats_df= pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)

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

            cv_tr_df = pd.DataFrame({"feat_name":self.model_fit.feature_names_in_, "importance":self.model_fit.feature_importances_}).sort_values(by = "importance", ascending = False)
            cv_tr_df["fold"] = i
            self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})

    
class RandomForest_forecaster:
    def __init__(self, target_col,add_trend = False, trend_type ="linear", ets_params = None, n_lag=None, lag_transform = None,
                 differencing_number = None, seasonal_length = None, cat_variables = None,
                 box_cox = False, box_cox_lmda = None, box_cox_biasadj= False):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = RandomForestRegressor
        self.target_col = target_col
        self.n_lag = n_lag
        self.difference = differencing_number
        self.season_diff = seasonal_length
        self.lag_transform = lag_transform
        self.cat_variables = cat_variables
        self.trend = add_trend
        self.trend_type = trend_type
        if ets_params is not None:
            self.ets_model = ets_params[0]
            self.ets_fit = ets_params[1]
        self.box_cox = box_cox
        self.lmda = box_cox_lmda
        self.biasadj = box_cox_biasadj
        self.tuned_params = None
        self.actuals = None
        self.prob_forecasts = None
        
    
        
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
                if (self.trend_type == "linear") | (self.trend_type == "feature_lr"):
                    self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
                    if (self.trend_type == "linear"):
                        dfc[self.target_col] = dfc[self.target_col]-self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
    
                if (self.trend_type == "ses")|(self.trend_type == "feature_ses"):
                    self.ses_model = ExponentialSmoothing(dfc[self.target_col], **self.ets_model).fit(**self.ets_fit)
                    if (self.trend_type == "ses"):
                        dfc[self.target_col] = dfc[self.target_col]-self.ses_model.fittedvalues.values
                        
            if (self.difference is not None)|(self.season_diff is not None):
                self.orig = dfc[self.target_col].tolist()
                if self.difference is not None:
                    dfc[self.target_col] = np.diff(dfc[self.target_col], n= self.difference, prepend=np.repeat(np.nan, self.difference))
                if self.season_diff is not None:
                    self.orig_d = dfc[self.target_col].tolist()
                    dfc[self.target_col] = seasonal_diff(dfc[self.target_col], self.season_diff)

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
                            
        if self.trend ==True:
            if (self.target_col in dfc.columns):
                if self.trend_type == "feature_lr":
                    dfc["trend"] = self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))

                if self.trend_type == "feature_ses":
                    dfc["trend"] = self.ses_model.fittedvalues.values
        dfc = dfc.dropna()


        # if self.target_col in dfc.columns:
        #     self.dfc = dfc
        #     self.df =df.loc[dfc.index]
        # else:
        return dfc

    def fit(self, df, param = None):
        if param is not None:
            model_ =self.model(**param)
        else:
            model_ =self.model()
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        # model_df = self.data_prep(df)
        model_train = self.data_prep(df)
        self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
        self.model_fit = model_.fit(self.X, self.y)

    def forecast(self, n_ahead, x_test = None):
        if x_test is not None:
            x_dummy = self.data_prep(x_test)
#         max_lag = self.n_lag[-1]
#         lags = self.y[-max_lag:].tolist()
        lags = self.y.tolist()
        predictions = []

        if self.trend ==True:
            if (self.trend_type == "linear") | (self.trend_type == "feature_lr"):
                trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))
            if (self.trend_type == "ses") | (self.trend_type == "feature_ses"):
                trend_pred = self.ses_model.forecast(n_ahead).values

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
                
            if (self.trend ==True):
                if (self.trend_type == "feature_lr")|(self.trend_type == "feature_ses"):
                    trend_var = [trend_pred[i]]
                else:
                    trend_var = []

            else:
                trend_var = []
                    
                    
            inp = x_var+inp_lag+transform_lag+trend_var
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns

            pred = self.model_fit.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)

        forecasts = np.array(predictions)
        if self.season_diff is not None:
            forecasts = invert_seasonal_diff(self.orig_d, forecasts, self.season_diff)

        if self.difference is not None:
            forecasts = undiff_ts(self.orig, forecasts, self.difference)

        if (self.trend ==True):
            if (self.trend_type =="linear")|(self.trend_type =="ses"):
                forecasts = trend_pred+forecasts

        forecasts = np.array([max(0, x) for x in forecasts])      
        if self.box_cox == True:
            forecasts = back_box_cox_transform(y_pred = forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)
        return forecasts
    
    def cv(self, df, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_df = pd.DataFrame()
        self.cv_forecats_df= pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)

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

            cv_tr_df = pd.DataFrame({"feat_name":self.model_fit.feature_names_in_, "importance":self.model_fit.feature_importances_}).sort_values(by = "importance", ascending = False)
            cv_tr_df["fold"] = i
            self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})
    
class AdaBoost_forecaster:
    def __init__(self, target_col,add_trend = False, trend_type ="linear", ets_params = None, n_lag=None, lag_transform = None,
                 differencing_number = None, seasonal_length = None, cat_variables = None,
                 box_cox = False, box_cox_lmda = None, box_cox_biasadj= False):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = AdaBoostRegressor
        self.target_col = target_col
        self.n_lag = n_lag
        self.difference = differencing_number
        self.season_diff = seasonal_length
        self.lag_transform = lag_transform
        self.cat_variables = cat_variables
        self.trend = add_trend
        self.trend_type = trend_type
        if ets_params is not None:
            self.ets_model = ets_params[0]
            self.ets_fit = ets_params[1]
        self.box_cox = box_cox
        self.lmda = box_cox_lmda
        self.biasadj = box_cox_biasadj
        self.tuned_params = None
        self.actuals = None
        self.prob_forecasts = None


        
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
                if (self.trend_type == "linear") | (self.trend_type == "feature_lr"):
                    self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
                    if (self.trend_type == "linear"):
                        dfc[self.target_col] = dfc[self.target_col]-self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
    
                if (self.trend_type == "ses")|(self.trend_type == "feature_ses"):
                    self.ses_model = ExponentialSmoothing(dfc[self.target_col], **self.ets_model).fit(**self.ets_fit)
                    if (self.trend_type == "ses"):
                        dfc[self.target_col] = dfc[self.target_col]-self.ses_model.fittedvalues.values
                
            if (self.difference is not None)|(self.season_diff is not None):
                self.orig = dfc[self.target_col].tolist()
                if self.difference is not None:
                    dfc[self.target_col] = np.diff(dfc[self.target_col], n= self.difference, prepend=np.repeat(np.nan, self.difference))
                if self.season_diff is not None:
                    self.orig_d = dfc[self.target_col].tolist()
                    dfc[self.target_col] = seasonal_diff(dfc[self.target_col], self.season_diff)

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
                            
        if self.trend ==True:
            if (self.target_col in dfc.columns):
                if self.trend_type == "feature_lr":
                    dfc["trend"] = self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))

                if self.trend_type == "feature_ses":
                    dfc["trend"] = self.ses_model.fittedvalues.values
        dfc = dfc.dropna()


        # if self.target_col in dfc.columns:
        #     self.dfc = dfc
        #     self.df =df.loc[dfc.index]
        # else:
        return dfc

    def fit(self, df, param = None):
        if param is not None:
            model_ =self.model(**param)
        else:
            model_ =self.model()
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        model_train = self.data_prep(df)
        self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
        self.model_fit = model_.fit(self.X, self.y)

    def forecast(self, n_ahead, x_test = None):
        self.H = n_ahead
        if x_test is not None:
            x_dummy = self.data_prep(x_test)
        lags = self.y.tolist()
        predictions = []
        
        if self.trend ==True:
            if (self.trend_type == "linear") | (self.trend_type == "feature_lr"):
                trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))
            if (self.trend_type == "ses") | (self.trend_type == "feature_ses"):
                trend_pred = self.ses_model.forecast(n_ahead).values
            
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
                
            if (self.trend ==True):
                if (self.trend_type == "feature_lr")|(self.trend_type == "feature_ses"):
                    trend_var = [trend_pred[i]]
                else:
                    trend_var = []
            else:
                trend_var = []
                    
                    
            inp = x_var+inp_lag+transform_lag+trend_var
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns

            pred = self.model_fit.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)

        forecasts = np.array(predictions)
        if self.season_diff is not None:
            forecasts = invert_seasonal_diff(self.orig_d, forecasts, self.season_diff)

        if self.difference is not None:
            forecasts = undiff_ts(self.orig, forecasts, self.difference)

        if (self.trend ==True):
            if (self.trend_type =="linear")|(self.trend_type =="ses"):
                forecasts = trend_pred+forecasts

        forecasts = np.array([max(0, x) for x in forecasts])  
        if self.box_cox == True:
            forecasts = back_box_cox_transform(y_pred = forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)
        return forecasts
    
    def cv(self, df, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_df = pd.DataFrame()
        self.cv_forecats_df= pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)

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

            cv_tr_df = pd.DataFrame({"feat_name":self.model_fit.feature_names_in_, "importance":self.model_fit.feature_importances_}).sort_values(by = "importance", ascending = False)
            cv_tr_df["fold"] = i
            self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})

    
class Cubist_forecaster:
    def __init__(self, target_col, add_trend = False, trend_type ="linear", ets_params = None, n_lag = None, lag_transform = None,
                 differencing_number = None,seasonal_length = None, cat_variables = None,
                 box_cox = False, box_cox_lmda = None, box_cox_biasadj= False):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = Cubist
        self.target_col = target_col
        self.n_lag = n_lag
        self.difference = differencing_number
        self.season_diff = seasonal_length
        self.lag_transform = lag_transform
        self.cat_variables = cat_variables
        self.trend = add_trend
        self.trend_type = trend_type
        if ets_params is not None:
            self.ets_model = ets_params[0]
            self.ets_fit = ets_params[1]
        self.box_cox = box_cox
        self.lmda = box_cox_lmda
        self.biasadj = box_cox_biasadj
        self.tuned_params = None
        self.actuals = None
        self.prob_forecasts = None

        
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
                if (self.trend_type == "linear") | (self.trend_type == "feature_lr"):
                    self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
                    if (self.trend_type == "linear"):
                        dfc[self.target_col] = dfc[self.target_col]-self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
    
                if (self.trend_type == "ses")|(self.trend_type == "feature_ses"):
                    self.ses_model = ExponentialSmoothing(dfc[self.target_col], **self.ets_model).fit(**self.ets_fit)
                    if (self.trend_type == "ses"):
                        dfc[self.target_col] = dfc[self.target_col]-self.ses_model.fittedvalues.values
                
            if (self.difference is not None)|(self.season_diff is not None):
                self.orig = dfc[self.target_col].tolist()
                if self.difference is not None:
                    dfc[self.target_col] = np.diff(dfc[self.target_col], n= self.difference, prepend=np.repeat(np.nan, self.difference))
                if self.season_diff is not None:
                    self.orig_d = dfc[self.target_col].tolist()
                    dfc[self.target_col] = seasonal_diff(dfc[self.target_col], self.season_diff)

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
                            
        if self.trend ==True:
            if (self.target_col in dfc.columns):
                if self.trend_type == "feature_lr":
                    dfc["trend"] = self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))

                if self.trend_type == "feature_ses":
                    dfc["trend"] = self.ses_model.fittedvalues.values
        dfc = dfc.dropna()


        # if self.target_col in dfc.columns:
        #     self.dfc = dfc
        #     self.df =df.loc[dfc.index]
        # else:
        return dfc

    def fit(self, df, param = None):
        if param is not None:
            model_ =self.model(**param)
        else:
            model_ =self.model()
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        model_train = self.data_prep(df)
        self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
        self.model_fit = model_.fit(self.X, self.y)

    def forecast(self, n_ahead, x_test = None):
        self.H = n_ahead
        if x_test is not None:
            x_dummy = self.data_prep(x_test)
        lags = self.y.tolist()
        predictions = []
        
        if self.trend ==True:
            if (self.trend_type == "linear") | (self.trend_type == "feature_lr"):
                trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))
            if (self.trend_type == "ses") | (self.trend_type == "feature_ses"):
                trend_pred = self.ses_model.forecast(n_ahead).values
            
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
                
            if (self.trend ==True):
                if (self.trend_type == "feature_lr")|(self.trend_type == "feature_ses"):
                    trend_var = [trend_pred[i]]
                else:
                    trend_var = []
            else:
                trend_var = []
                    
                    
            inp = x_var+inp_lag+transform_lag+trend_var
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns

            pred = self.model_fit.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)

        forecasts = np.array(predictions)
        if self.season_diff is not None:
            forecasts = invert_seasonal_diff(self.orig_d, forecasts, self.season_diff)

        if self.difference is not None:
            forecasts = undiff_ts(self.orig, forecasts, self.difference)

        if (self.trend ==True):
            if (self.trend_type =="linear")|(self.trend_type =="ses"):
                forecasts = trend_pred+forecasts

        forecasts = np.array([max(0, x) for x in forecasts])        
        if self.box_cox == True:
            forecasts = back_box_cox_transform(y_pred = forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)
        return forecasts

    def cv(self, df, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {i.__name__:[] for i in metrics}
        self.cv_forecats_df= pd.DataFrame()

        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)

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

            # cv_tr_df = pd.DataFrame({"feat_name":self.model_ada.feature_names_in_, "importance":self.model_ada.feature_importances_}).sort_values(by = "importance", ascending = False)
            # cv_tr_df["fold"] = i
            # self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})

    
class HistGradientBoosting_forecaster:
    def __init__(self, target_col, add_trend = False, trend_type ="linear", ets_params = None, n_lag = None, lag_transform = None, 
                 differencing_number = None, seasonal_length = None, cat_variables = None,
                 box_cox = False, box_cox_lmda = None, box_cox_biasadj= False):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = HistGradientBoostingRegressor
        self.target_col = target_col
        self.n_lag = n_lag
        self.difference = differencing_number
        self.season_diff = seasonal_length
        self.lag_transform = lag_transform
        self.cat_variables = cat_variables
        self.trend = add_trend
        self.trend_type = trend_type
        if ets_params is not None:
            self.ets_model = ets_params[0]
            self.ets_fit = ets_params[1]
        self.box_cox = box_cox
        self.lmda = box_cox_lmda
        self.biasadj = box_cox_biasadj
        self.tuned_params = None
        self.actuals = None
        self.prob_forecasts = None

        
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
                if (self.trend_type == "linear") | (self.trend_type == "feature_lr"):
                    self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
                    if (self.trend_type == "linear"):
                        dfc[self.target_col] = dfc[self.target_col]-self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
    
                if (self.trend_type == "ses")|(self.trend_type == "feature_ses"):
                    self.ses_model = ExponentialSmoothing(dfc[self.target_col], **self.ets_model).fit(**self.ets_fit)
                    if (self.trend_type == "ses"):
                        dfc[self.target_col] = dfc[self.target_col]-self.ses_model.fittedvalues.values

            if (self.difference is not None)|(self.season_diff is not None):
                self.orig = dfc[self.target_col].tolist()
                if self.difference is not None:
                    dfc[self.target_col] = np.diff(dfc[self.target_col], n= self.difference, prepend=np.repeat(np.nan, self.difference))
                if self.season_diff is not None:
                    self.orig_d = dfc[self.target_col].tolist()
                    dfc[self.target_col] = seasonal_diff(dfc[self.target_col], self.season_diff)

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
                            
        if self.trend ==True:
            if (self.target_col in dfc.columns):
                if self.trend_type == "feature_lr":
                    dfc["trend"] = self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))

                if self.trend_type == "feature_ses":
                    dfc["trend"] = self.ses_model.fittedvalues.values
        dfc = dfc.dropna()

        return dfc

    def fit(self, df, param = None):
        if param is not None:
            model_ =self.model(**param)
        else:
            model_ =self.model()
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        
        model_train = self.data_prep(df)
        self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]

        self.model_fit = model_.fit(self.X, self.y)

    def forecast(self, n_ahead, x_test = None):
        if x_test is not None:
            x_dummy = self.data_prep(x_test)
        lags = self.y.tolist()
        predictions = []

        if self.trend ==True:
            if (self.trend_type == "linear") | (self.trend_type == "feature_lr"):
                trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))
            if (self.trend_type == "ses") | (self.trend_type == "feature_ses"):
                trend_pred = self.ses_model.forecast(n_ahead).values

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
                
            if (self.trend ==True):
                if (self.trend_type == "feature_lr")|(self.trend_type == "feature_ses"):
                    trend_var = [trend_pred[i]]
                else:
                    trend_var = []
            else:
                trend_var = [] 
                    
            inp = x_var+inp_lag+transform_lag+trend_var
            df_inp = pd.DataFrame(inp).T
            df_inp.columns = self.X.columns

            pred = self.model_fit.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)

        forecasts = np.array(predictions)
        if self.season_diff is not None:
            forecasts = invert_seasonal_diff(self.orig_d, forecasts, self.season_diff)

        if self.difference is not None:
            forecasts = undiff_ts(self.orig, forecasts, self.difference)

        if (self.trend ==True):
            if (self.trend_type =="linear")|(self.trend_type =="ses"):
                forecasts = trend_pred+forecasts   
        forecasts = np.array([max(0, x) for x in forecasts])      
        if self.box_cox == True:
            forecasts = back_box_cox_transform(y_pred = forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)
        return forecasts

    def cv(self, df, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_forecats_df= pd.DataFrame()

        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)

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

            # cv_tr_df = pd.DataFrame({"feat_name":self.model_hist.feature_names_in_, "importance":self.model_hist.feature_importances_}).sort_values(by = "importance", ascending = False)
            # cv_tr_df["fold"] = i
            # self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})

class LR_forecaster:
    def __init__(self, target_col, model = LinearRegression, add_trend = False, trend_type ="linear", ets_params = None,
                 n_lag = None, lag_transform = None, differencing_number = None, seasonal_length = None,
                 cat_variables = None,
                 box_cox = False, box_cox_lmda = None, box_cox_biasadj= False):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = model
        self.target_col = target_col
        self.n_lag = n_lag
        self.difference = differencing_number
        self.season_diff = seasonal_length
        self.lag_transform = lag_transform
        self.cat_variables = cat_variables
        self.trend = add_trend
        self.trend_type = trend_type
        if ets_params is not None:
            self.ets_model = ets_params[0]
            self.ets_fit = ets_params[1]
        self.box_cox = box_cox
        self.lmda = box_cox_lmda
        self.biasadj = box_cox_biasadj
        self.tuned_params = None
        self.actuals = None
        self.prob_forecasts = None

        
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
                if (self.trend_type == "linear") | (self.trend_type == "feature_lr"):
                    self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[self.target_col])
                    if (self.trend_type == "linear"):
                        dfc[self.target_col] = dfc[self.target_col]-self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
    
                if (self.trend_type == "ses")|(self.trend_type == "feature_ses"):
                    self.ses_model = ExponentialSmoothing(dfc[self.target_col], **self.ets_model).fit(**self.ets_fit)
                    if (self.trend_type == "ses"):
                        dfc[self.target_col] = dfc[self.target_col]-self.ses_model.fittedvalues.values

            if (self.difference is not None)|(self.season_diff is not None):
                self.orig = dfc[self.target_col].tolist()
                if self.difference is not None:
                    dfc[self.target_col] = np.diff(dfc[self.target_col], n= self.difference, prepend=np.repeat(np.nan, self.difference))
                if self.season_diff is not None:
                    self.orig_d = dfc[self.target_col].tolist()
                    dfc[self.target_col] = seasonal_diff(dfc[self.target_col], self.season_diff)

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
                            
        if self.trend ==True:
            if (self.target_col in dfc.columns):
                if self.trend_type == "feature_lr":
                    dfc["trend"] = self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))

                if self.trend_type == "feature_ses":
                    dfc["trend"] = self.ses_model.fittedvalues.values
        dfc = dfc.dropna()

        return dfc

    def fit(self, df, param = None):
        if param is not None:
            model =self.model(**param)
        else:
            model =self.model()
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
        
        model_train = self.data_prep(df)
        self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]

        self.model_fit = model.fit(self.X, self.y)

    def forecast(self, n_ahead, x_test = None):
        if x_test is not None:
            x_dummy = self.data_prep(x_test)
        lags = self.y.tolist()
        predictions = []

        if self.trend ==True:
            if (self.trend_type == "linear") | (self.trend_type == "feature_lr"):
                trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))
            if (self.trend_type == "ses") | (self.trend_type == "feature_ses"):
                trend_pred = self.ses_model.forecast(n_ahead).values

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
                
            if (self.trend ==True):
                if (self.trend_type == "feature_lr")|(self.trend_type == "feature_ses"):
                    trend_var = [trend_pred[i]]
                else:
                    trend_var = []
            else:
                trend_var = [] 
                    
            inp = x_var+inp_lag+transform_lag+trend_var

            pred = self.model_fit.predict(np.array(inp).reshape(1,-1))[0]

            predictions.append(pred)
            lags.append(pred)

        forecasts = np.array(predictions)
        if self.season_diff is not None:
            forecasts = invert_seasonal_diff(self.orig_d, forecasts, self.season_diff)

        if self.difference is not None:
            forecasts = undiff_ts(self.orig, forecasts, self.difference)

        if (self.trend ==True):
            if (self.trend_type =="linear")|(self.trend_type =="ses"):
                forecasts = trend_pred+forecasts 
        forecasts = np.array([max(0, x) for x in forecasts])      
        if self.box_cox == True:
            forecasts = back_box_cox_transform(y_pred = forecasts, lmda = self.lmda, shift= self.is_zero, box_cox_biasadj=self.biasadj)
        return forecasts

    def cv(self, df, cv_split, test_size, metrics, params = None):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_impotance = pd.DataFrame()
        self.cv_forecats_df= pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_col), np.array(test[self.target_col])
            
            if params is not None:
                self.fit(train, param = params)
            else:
                self.fit(train)
            
            bb_forecast = self.forecast(test_size, x_test=x_test)

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

            cv_tr_df = pd.DataFrame({"features":self.model_fit.feature_names_in_, "coefs":self.model_fit.coef_}).sort_values(by = "coefs", ascending = False)
            cv_tr_df["fold"] = i
            self.cv_impotance = pd.concat([self.cv_impotance, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})

    
def cv_tune(model, df, cv_split, test_size, param_space,eval_metric, opt_horizon = None,
                eval_num = 100, verbose= False):

    tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
    
    def objective(params):
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


        if (model.trend_type == "ses")|(model.trend_type == "feature_ses"):
            model.ets_model = {}
            for ets_param1 in ["trend", "damped_trend", "seasonal", "seasonal_periods"]:
                if ets_param1 in params:
                    model.ets_model[ets_param1] = params[ets_param1]
            model.ets_fit = {}
            for ets_param2 in ["smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend"]:
                if ets_param2 in params:
                    model.ets_fit[ets_param2] = params[ets_param2]
            # model.ets_model = ets_params[0]

            # self.data_prep(df)
        if model.model.__name__ != 'LinearRegression':
            model_params = {k: v for k, v in params.items() if (k not in ["box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj", "trend", "damped_trend", "seasonal", "seasonal_periods", "smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend"])}

        metric = []
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.iloc[:, 1:], np.array(test[model.target_col])

            if model.model.__name__ != 'LinearRegression':
                model.fit(train, model_params)
            else:
                model.fit(train)
            yhat = model.forecast(n_ahead =len(y_test), x_test=x_test)
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

        if verbose ==True:
            print ("SCORE:", score)
        return {'loss':score, 'status':STATUS_OK}
        
    trials = Trials()

    best_hyperparams = fmin(fn = objective,
                    space = param_space,
                    algo = tpe.suggest,
                    max_evals = eval_num,
                    trials = trials)
    model.tuned_params = [space_eval(param_space, {k: v[0] for k, v in t['misc']['vals'].items()}) 
                for t in trials.trials]
    
    return space_eval(param_space, best_hyperparams)

def backtest_tune(model, df, n_windows, H, param_space,eval_metric, opt_horizon = None,
                eval_num = 100, verbose= False):
    
    def objective(params):
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

            # self.data_prep(df)
        if model.model.__name__ != 'LinearRegression':
            model_params = {k: v for k, v in params.items() if (k not in ["box_cox", "n_lag", "box_cox_lmda", "box_cox_biasadj"])}

        metric = []
        fix_idx = n_windows+H
        for i in range(1, n_windows+1):
            train = df[:-(fix_idx-i)]
            if i ==n_windows:
                test = df[-H:]
            else:
                test= df[-(fix_idx-i):-(n_windows-i)]
                
            x_test, y_test = test.iloc[:, 1:], np.array(test[model.target_col])

            if model.model.__name__ != 'LinearRegression':
                model.fit(train, model_params)
            else:
                model.fit(train)

            yhat = model.forecast(n_ahead =len(y_test), x_test=x_test)
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

        if verbose ==True:
            print ("SCORE:", score)
        return {'loss':score, 'status':STATUS_OK}
        
    trials = Trials()

    best_hyperparams = fmin(fn = objective,
                    space = param_space,
                    algo = tpe.suggest,
                    max_evals = eval_num,
                    trials = trials)
    model.tuned_params = [space_eval(param_space, {k: v[0] for k, v in t['misc']['vals'].items()}) 
                for t in trials.trials]
    
    return space_eval(param_space, best_hyperparams)

def backtest_model(model, df, n_windows, H, metrics, model_params=None):
    
    metrics_dict = {m.__name__: [] for m in metrics}
    prob_forecasts = []
    actuals = []
    fix_idx = n_windows+H
    for i in range(1, n_windows+1):
        train = df[:-(fix_idx-i)]
        if i ==n_windows:
            test = df[-H:]
        else:
            test= df[-(fix_idx-i):-(n_windows-i)]
            
        x_test, y_test = test.iloc[:, 1:], np.array(test[model.target_col])

        if model_params is not None:
            model.fit(train, model_params)
        else:
            model.fit(train)
        yhat = model.forecast(n_ahead =len(y_test), x_test=x_test)
        actuals.append(y_test)
        prob_forecasts.append(yhat)
        for m in metrics:
            if m.__name__== 'mean_squared_error':
                eval = m(y_test, yhat, squared=False)
            elif (m.__name__== 'MeanAbsoluteScaledError')|(m.__name__== 'MedianAbsoluteScaledError'):
                eval = m(y_test, yhat, np.array(train[model.target_col]))
            else:
                eval = m(y_test, yhat)
            metrics_dict[m.__name__].append(eval)
            
    overal_perform = [[m.__name__, np.mean(metrics_dict[m.__name__])] for m in metrics]
    
    prob_forecasts = np.row_stack(prob_forecasts)
    prob_forecasts=pd.DataFrame(prob_forecasts)
    prob_forecasts.columns = ["horizon_"+str(i+1) for i in prob_forecasts.columns]
    model.prob_forecasts = prob_forecasts
    
    actuals = np.row_stack(actuals)
    actuals=pd.DataFrame(actuals)
    actuals.columns = ["horizon_"+str(i+1) for i in actuals.columns]
    model.actuals = actuals
    
    return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})

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


import statsmodels.api as sm
class VAR_model:
    def __init__(self, target_cols, lag_dict, lag_transform = None, diff_dict = None, seasonal_diff =None,
                 trend = None,trend_types=None,ets_params = None, box_cox = None, box_cox_lmda = None, box_cox_biasadj= False, add_constant = True,
                 cat_variables = None, verbose = False):
        
        self.target_cols = target_cols
        self.cat_variables = cat_variables
        
        self.box_cox = box_cox
        if self.box_cox is not None:
            if not isinstance(self.box_cox, dict):
                raise TypeError("box_cox must be a dictionary of target values")
            
        self.lamdas = box_cox_lmda
        self.biasadj = box_cox_biasadj
        if (self.biasadj ==False) & (self.box_cox is not None): #define default value of box_cox_biasadj
            self.biasadj= {}
            for i in self.box_cox:
                self.biasadj[i] = False

        self.trend = trend
        
        if self.trend is not None:
            if not isinstance(self.trend, dict):
                raise TypeError("trend must be a dictionary of target values")
        
        self.trend_types = trend_types
        self.ets_params = ets_params
        self.lag_dict = lag_dict
        self.diffs = diff_dict
        self.season_diffs = seasonal_diff
        self.lag_transform = lag_transform
        self.cons = add_constant

        if (self.trend is not None) & (self.trend_types is None): #definin default trend type when trend type is not passed
            self.trend_types = {}
            for i in self.trend:
                self.trend_types[i] = "linear"

    def data_prep(self, df):
        dfc = df.copy()
        if self.cat_variables is not None:
            for col, cat in self.cat_var.items():
                dfc[col] = dfc[col].astype('category')
                dfc[col] = dfc[col].cat.set_categories(cat)
            dfc = pd.get_dummies(dfc)
            
            for i in self.drop_categ:
                dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)
                
        if all(elem in dfc.columns for elem in self.target_cols):

            if self.box_cox is not None:
                if self.lamdas is None:
                    self.lamdas = {i: None for i in self.box_cox}
                    
                self.is_zeros = {i: None for i in self.lamdas}
                
                for k, lm in self.lamdas.items():
                    
                    self.is_zeros[k] = np.any(np.array(dfc[k]) < 1)
                    trans_data, self.lamdas[k] = box_cox_transform(x = dfc[k], shift = self.is_zeros[k], box_cox_lmda=lm)
                    if self.box_cox[k] ==True:
                        dfc[k] = trans_data

            if self.trend is not None:
                self.tr_models = {i: None for i in self.trend_types}
                self.len = len(dfc)
                for k, tr in self.trend_types.items():
                    if (tr == "linear") | (tr == "feature_lr"):
                        self.tr_models[k] = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1), dfc[k])
                        if (tr == "linear"):
                            dfc[k] = dfc[k]-self.tr_models[k].predict(np.array(range(self.len)).reshape(-1, 1))

                    if (tr == "ses")|(tr == "feature_ses"):
                        self.tr_models[k] = ExponentialSmoothing(dfc[k], **self.ets_param[k][0]).fit(**self.ets_param[k][1])
                        if (tr == "ses"):
                            dfc[k] = dfc[k]-self.tr_models[k].fittedvalues.values
            
            if self.diffs is not None:
                self.origs = {i: None for i in self.diffs}
                for x, d in self.diffs.items():
                    self.origs[x] = dfc[x].tolist()
                    dfc[x] = np.diff(dfc[x], n= d, prepend=np.repeat(np.nan, d))
                    
            if self.season_diffs is not None:
                self.orig_ds = {i: None for i in self.season_diffs}
                for w, s in self.diffs.items():
                    self.orig_ds[w] = dfc[w].tolist()
                    dfc[w] = seasonal_diff(dfc[w], s)
            
            for a, lags in self.lag_dict.items():
                if len(lags)>0: # to ensure that given lags are not empty for target variables
                    for lg in lags:
                        dfc[str(a)+"_lag"+"_"+str(lg)] = dfc[a].shift(lg)
                    
            if self.lag_transform is not None:
                for n, k in self.lag_transform.items():
                    for f in k:
                        df_array = np.array(dfc[n].shift(f[0]))
                        if f[1].__name__ == "rolling_quantile":
                            dfc["q_"+str(f[3])+"_"+str(n)+"_"+str(f[0])+"_w"+str(f[2])] = f[1](df_array, f[2], f[3])
                        else:
                            dfc[f[1].__name__+"_"+str(n)+"_"+str(f[0])+"_"+str(f[2])] = f[1](df_array, f[2])
                            
                
            # for i in self.lag_list2:
            #     dfc[self.target_col[1]+"_lag"+"_"+str(i)] = dfc[self.target_col[1]].shift(i)
        dfc = dfc.dropna()
        return dfc

    # def summary(self):
    #     from scipy.stats import t
    #     output = pd.DataFrame()
    #     for i in range(self.N):
    #         wi = self.posterior[i]
    #         W = np.diag(wi)

    #         X_weight = np.dot(W, self.Xs)
    #         y_weight= np.dot(W, self.ys)
    #         coeff_state = np.linalg.lstsq(X_weight, y_weight, rcond=None)[0]

            
    #         res= y_weight-(coeff_state.T @ X_weight.T).T
    #         kp = self.Xs.shape[1]
    #         dfg = self.ys.shape[0]-kp
            
            
    #         state_index = self.ys.shape[1]*self.Xs.shape[1]*[f"state_{i}"]
    #         endog_index = []
    #         coef_index = self.ys.shape[1]*self.col_names
    #         endogs = pd.DataFrame()
    #         for k in range(self.ys.shape[1]):
                
    #             endog_idx = [f"results for {self.target_col[k]}" for i in range(self.Xs.shape[1])]
    #             endog_index+=endog_idx
                
                
    #             RSS = np.sum(res[:,k]**2)
    #             SEs = np.sqrt((RSS/dfg) * np.diag(np.linalg.inv(X_weight.T @ X_weight)))
    #             t_vals = coeff_state[:, k]/SEs
    #             p_values = (1 - t.cdf(np.abs(t_vals), dfg)) * 2
    #             result = np.round(np.column_stack((coeff_state[:, k], SEs, t_vals, p_values)), 4)
    #             result = pd.DataFrame(result)
    #             # coef_names = [f"state_{i}_{item}" for item in self.col_names]
                
    #             endogs = pd.concat([endogs, result])
    #         new_idx = [state_index, endog_index, coef_index]
    #         # print(len(state_index), len(endog_index), len(coef_index))
    #         endogs.index = pd.MultiIndex.from_arrays(new_idx)
    #         endogs.columns = ["coefficients", "SE", "t-values", "p-values"]
    #         output = pd.concat([output, endogs])

    #     return output

        

    def forecast(self, H, exog=None):
        y_lists = {j: self.y[:,i].tolist() for i,j in enumerate(self.target_cols)}
        if exog is not None:
            if self.cons == True:
                if exog.shape[0] == 1:
                    exog.insert(0, 'const', 1)
                else:
                    exog = sm.add_constant(exog)
            exog = np.array(self.data_prep(exog))

        forecasts = {i: [] for i in self.target_cols}

        if self.trend is not None:
            # self.tr_models = {f"{i}": None for i in self.trend_types}
            # self.len = len(dfc)
            trend_preds = {i: [] for i in self.trend_types}
            for k, tr in self.trend_types.items():
                if (tr == "linear") | (tr == "feature_lr"):
                    trend_preds[k] = self.tr_models[k].predict(np.array(range(self.len, self.len+H)).reshape(-1, 1))
    
                if (tr == "ses")|(tr == "feature_ses"):
                    trend_preds[k] = self.tr_models[k].forecast(n_ahead).values

        for t in range(H): # recursion step
            if exog is not None:
                exo_inp = exog[t].tolist()
            else:
                if self.cons == True:
                    exo_inp = [1]
                else:
                    exo_inp = []


            if self.lag_dict is not None:
                lags = []
                for tr, v in y_lists.items():
                    if len(self.lag_dict[tr])>0: # to ensure that given lags are not empty for target variables
                        ys = [v[-x] for x in self.lag_dict[tr]]
                        lags+=ys
            else:
                lags = []

                # inp = exo_inp+lags1+lags2
            
            if self.lag_transform is not None:
                transform_lag = []    
                for n, k in self.lag_transform.items():
                    for f in k:
                        df_array = np.array(pd.Series(lags).shift(f[0]-1))
                        if f[1].__name__ == "rolling_quantile":
                            t1 = f[1](df_array, f[2], f[3])[-1]
                        else:
                            t1 = f[1](df_array, f[2])[-1]
                        transform_lag.append(t1)
            else:
                transform_lag = []

            
            if self.trend is not None:
                trend_var = []
                for k, tr in self.trend_types.items():
                    if (tr == "feature_lr")|(tr == "feature_ses"):
                        trend_var.append(trend_preds[k][t])
                    else:
                        trend_var = []
            else:
                trend_var = [] 
                
            
            inp = exo_inp+lags+transform_lag+trend_var

            pred = self.predict(inp)
            for id, ff in enumerate(forecasts):
                forecasts[ff].append(pred[id])
                y_lists[ff].append(pred[id])


        if self.season_diffs is not None:
            for s in self.orig_ds:
                forecasts[s] = invert_seasonal_diff(self.orig_ds[s], np.array(forecasts[s]), self.season_diffs[s])
                
        if self.diffs is not None:
            for d in self.diffs:
                forecasts[d] = undiff_ts(self.origs[d], np.array(forecasts[d]), self.diffs[d])
                
        if self.trend is not None:
            for k, tr in self.trend_types.items():
                if (tr =="linear")|(tr =="ses"):
                    forecasts[k] = trend_preds[k]+forecasts[k]
        for f in forecasts:    
            forecasts[f] = np.array([max(0, x) for x in forecasts[f]])

        if self.box_cox is not None:
            for k, lmd in self.lamdas.items():
                if self.box_cox[k]==True:
                    forecasts[k] = back_box_cox_transform(y_pred = forecasts[k], lmda = lmd, shift= self.is_zeros[k], box_cox_biasadj=self.biasadj[k])
      
        return forecasts
        
    def predict(self, X):
        arr = np.array(X) 
        return np.dot(self.coeffs.T, arr.T)
        

    def fit(self, df_train):
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df_train[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
            self.drop_categ= [sorted(df_train[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
            
        df = self.data_prep(df_train)
        self.X = np.array(df.drop(columns = self.target_cols))
        if self.cons == True:
            self.X = sm.add_constant(self.X)
        self.y = np.array(df[self.target_cols])
        self.coeffs = np.linalg.lstsq(self.X, self.y, rcond=None)[0]

    def cv_var(self, df, target_col, cv_split, test_size, metrics):
    
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        self.metrics_dict = {m.__name__: [] for m in metrics}
        # self.cv_impotance = pd.DataFrame()
        self.cv_forecats_df= pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns = self.target_cols), np.array(test[target_col])
            
            self.fit(train)
            
            bb_forecast = self.forecast(H=test_size, exog=x_test)[target_col]

            forecat_df = test[target_col].to_frame()
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

            # cv_tr_df = pd.DataFrame({"features":self.model_fit.feature_names_in_, "coefs":self.model_fit.coef_}).sort_values(by = "coefs", ascending = False)
            # cv_tr_df["fold"] = i
            # self.cv_impotance = pd.concat([self.cv_impotance, cv_tr_df], axis=0)

        overal_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]  
        
        return pd.DataFrame(overal_perform).rename(columns = {0:"eval_metric", 1:"score"})


def cv_tune_var(model, df, target_col, cv_split, test_size, param_space,eval_metric, opt_horizon = None,
                eval_num = 100, verbose= False):

    tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
    
    def check_pattern(pattern, param):
        return any(pattern in text for text in param)
    
    def objective(params):
        
        if check_pattern("lag_dict", params)|check_pattern("diff_dict", params)|check_pattern("box_cox", params):
            if check_pattern("lag_dict", params):
                model.lag_dict = {}
                for i, j in enumerate(model.target_cols):
                    lag_str = j+str(i)
                    if type(params[lag_str]) is tuple:
                        model.lag_dict[j] = list(params[lag_str])
                    else:
                        model.lag_dict[j] = range(1, params[lag_str]+1)
                
            if check_pattern("diff_dict", params):
                model.diffs = {}
                for i, j in enumerate(model.target_cols):
                    diff_str = j+str(i)
                    model.diffs[j] = params[diff_str]
            
            if check_pattern("box_cox_lmda", params):
                model.lamdas = {}
                for i, j in enumerate(model.target_cols):
                    lamd_str = "box_cox_lmda"+str(i)
                    model.lamdas[j] = params[lamd_str]

            if check_pattern("box_cox_biasadj", params):
                model.box_cox_biasadj = {}
                for i, j in enumerate(model.target_cols):
                    bias_str = "box_cox_biasadj"+str(i)
                    model.box_cox_biasadj[j] = params[bias_str]

        metric = []
        for train_index, test_index in tscv.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.iloc[:, 2:], np.array(test[target_col])

            # if model.model.__name__ != 'LinearRegression':
            #     model.fit(train, model_params)
            # else:
            model.fit(train)
            yhat = model.forecast(H =len(y_test), exog=x_test)[target_col]
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

        if verbose ==True:
            print ("SCORE:", score)
        return {'loss':score, 'status':STATUS_OK}
        
    trials = Trials()

    best_hyperparams = fmin(fn = objective,
                    space = param_space,
                    algo = tpe.suggest,
                    max_evals = eval_num,
                    trials = trials)
    model.tuned_params = [space_eval(param_space, {k: v[0] for k, v in t['misc']['vals'].items()}) 
                for t in trials.trials]
    
    return space_eval(param_space, best_hyperparams)