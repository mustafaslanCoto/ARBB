#!/usr/bin/env python3
"""
ML Forecasting Package
=======================

This module contains various forecasting classes (using CatBoost, LightGBM, XGBoost,
RandomForest, etc.) and utility functions for cross-validation and hyperparameter
tuning for time-series forecasting.
"""

from tabnanny import verbose
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit, KFold
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from arbb.utils import box_cox_transform, back_box_cox_transform, undiff_ts, seasonal_diff, invert_seasonal_diff, kfold_target_encoder, target_encoder_for_test
from catboost import CatBoostRegressor
from cubist import Cubist


class ml_forecaster:
    """
    ml Forecaster for time series forecasting.

    Args:
        model (class): Machine learning model class (e.g., CatBoostRegressor, LGBMRegressor).
        target_col (str): Name of the target variable.
        cat_variables (list, optional): List of categorical features.
        n_lag (list or int, optional): Lag(s) to include as features.
        difference (int, optional): Order of difference (e.g. 1 for first difference).
        seasonal_length (int, optional): Seasonal period for seasonal differencing.
        trend (bool, optional): Whether to remove trend.
        trend_type (str, optional): Type of trend removal ('linear', 'feature_lr', 'ses', 'feature_ses').
        ets_params (tuple, optional): A tuple (model_params, fit_params) for exponential smoothing. Ex.g. ({'trend': 'add', 'seasonal': 'add'}, {'damped_trend': True}).
        box_cox (bool, optional): Whether to perform a Box–Cox transformation.
        box_cox_lmda (float, optional): The lambda value for Box–Cox.
        box_cox_biasadj (bool, optional): If True, adjust bias after Box–Cox inversion. Default is False.
        lag_transform (dict, optional): Dictionary specifying additional lag transformations.
    """
    def __init__(self, model, target_col, cat_variables=None, n_lag=None, difference=None, seasonal_diff=None,
                 trend=False, trend_type="linear", ets_params=None, box_cox=False, box_cox_lmda=None,
                 box_cox_biasadj=False, lag_transform=None):
        # Validate that either n_lag or lag_transform is provided
        if n_lag is None and lag_transform is None:
            raise ValueError("You must supply either n_lag or lag_transform parameters")
            
        self.target_col = target_col
        self.cat_var = cat_variables
        self.n_lag = n_lag  # single integer or list of lags
        self.difference = difference
        self.season_diff = seasonal_diff
        self.trend = trend
        self.trend_type = trend_type
        if ets_params is not None:
            self.ets_model = ets_params[0]
            self.ets_fit = ets_params[1]
        self.box_cox = box_cox
        self.lamda = box_cox_lmda
        self.biasadj = box_cox_biasadj
        self.lag_transform = lag_transform
        
        # Set default tuned parameters and placeholders for fitted attributes
        self.tuned_params = None
        self.actuals = None
        self.prob_forecasts = None
        self.model = model  # the chosen ML model

        if self.model.__name__ in ["CatBoostRegressor", "LGBMRegressor"]:
        
            def data_prep(self, df):
                """
                Prepare the data with lag features, differencing, trend-removal, and Box–Cox transformation.

                Args:
                    df (pd.DataFrame): Input dataframe.

                Returns:
                    pd.DataFrame: Dataframe ready for model fitting.
                """
                dfc = df.copy()
                # Box-Cox transformation if flag is set
                if self.box_cox:
                    self.is_zero = np.any(np.array(dfc[self.target_col]) < 1)
                    trans_data, self.lamda = box_cox_transform(x=dfc[self.target_col],
                                                                shift=self.is_zero,
                                                                box_cox_lmda=self.lamda)
                    dfc[self.target_col] = trans_data

                # Detrend if required
                if self.trend:
                    self.len = len(dfc)
                    if self.trend_type in ["linear", "feature_lr"]:
                        self.lr_model = LinearRegression().fit(np.array(range(self.len)).reshape(-1, 1),
                                                                dfc[self.target_col])
                        if self.trend_type == "linear":
                            # Remove trend component directly from target
                            dfc[self.target_col] = dfc[self.target_col] - self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
                    elif self.trend_type in ["ses", "feature_ses"]:
                        # Apply simple exponential smoothing detrending
                        self.ses_model = ExponentialSmoothing(dfc[self.target_col], **self.ets_model).fit(**self.ets_fit)
                        if self.trend_type == "ses":
                            dfc[self.target_col] = dfc[self.target_col] - self.ses_model.fittedvalues.values

                # Apply ordinary differencing if needed
                if self.difference is not None:
                    self.orig = dfc[self.target_col].tolist()
                    dfc[self.target_col] = np.diff(dfc[self.target_col], n=self.difference,
                                                prepend=np.repeat(np.nan, self.difference))
                # Apply seasonal differencing if needed
                if self.season_diff is not None:
                    self.orig_d = dfc[self.target_col].tolist()
                    dfc[self.target_col] = seasonal_diff(dfc[self.target_col], self.season_diff)
                    
                # Process categorical variables if provided
                if self.cat_var is not None:
                    for col in self.cat_var:
                        dfc[col] = dfc[col].astype('category')
                        
                # Create lag features using n_lag parameter
                if self.n_lag is not None:
                    if isinstance(self.n_lag, int):
                        lags = range(1, self.n_lag+1)
                    else:
                        lags = self.n_lag
                    for lag in lags:
                        dfc[f"{self.target_col}_lag_{lag}"] = dfc[self.target_col].shift(lag)
                        
                # Apply additional lag transformations if specified
                if self.lag_transform is not None:
                    for n, funcs in self.lag_transform.items():
                        df_array = np.array(dfc[self.target_col].shift(n))
                        for func in funcs:
                            # Check if function uses a tuple structure for additional parameters
                            if isinstance(func, tuple):
                                if func[0].__name__ == "rolling_quantile":
                                    dfc[f"q_{func[2]}_{n}_{func[1]}"] = func[0](df_array, func[1], func[2])
                                else:
                                    dfc[f"{func[0].__name__}_{n}_{func[1]}"] = func[0](df_array, func[1])
                            else:
                                dfc[f"{func.__name__}_{n}"] = func(np.array(dfc[self.target_col]), n)
                                
                # Add trend features if using "feature_lr" or "feature_ses"
                if self.trend:
                    # if (self.target_col in dfc.columns):
                    if self.trend_type == "feature_lr":
                        dfc["trend"] = self.lr_model.predict(np.array(range(self.len)).reshape(-1, 1))
                    elif self.trend_type == "feature_ses":
                        dfc["trend"] = self.ses_model.fittedvalues.values

                # Drop rows with missing values due to differencing/lagging
                return dfc.dropna()

            def fit(self, df, param=None):
                """
                Fit the ml model.

                Args:
                    df (pd.DataFrame): Input dataframe.
                    param (dict, optional): Parameters to pass into the model.
                """
                # Choose model parameters if provided
                if self.model.__name__ == "LGBMRegressor":
                    model_ = self.model(**param, verbose=-1) if param is not None else self.model(verbose=-1)
                else:
                    model_ = self.model(**param) if param is not None else self.model()
                model_df = self.data_prep(df)
                self.X = model_df.drop(columns=[self.target_col])
                self.y = model_df[self.target_col]
                # Fit the model (passing the categorical features if provided)
                if self.model.__name__ == "LGBMRegressor":
                    self.model_fit = model_.fit(self.X, self.y, categorical_feature=self.cat_var)
                else:
                    self.model_fit = model_.fit(self.X, self.y, cat_features=self.cat_var, verbose=True)

            def forecast(self, n_ahead, x_test=None):
                """
                Forecast n_ahead time steps.

                Args:
                    n_ahead (int): Number of forecast steps.
                    x_test (pd.DataFrame, optional): Exogenous variables for forecasting.

                Returns:
                    np.array: Forecasted values.
                """
                lags = self.y.tolist() # to keep the latest values for lag features
                predictions = []
                
                # Compute trend forecasts if needed
                if self.trend:
                    if self.trend_type in ["linear", "feature_lr"]:
                        trend_pred = self.lr_model.predict(np.array(range(self.len, self.len+n_ahead)).reshape(-1, 1))
                    elif self.trend_type in ["ses", "feature_ses"]:
                        trend_pred = self.ses_model.forecast(n_ahead).values

                for i in range(n_ahead):
                    # If external regressors are provided, extract the i-th row
                    if x_test is not None:
                        x_var = x_test.iloc[i, :].tolist()
                    else:
                        x_var = []
                        
                    # Build lag-based features from the latest forecast–history
                    if self.n_lag is not None:
                        if isinstance(self.n_lag, int):
                            lags_used = range(1, self.n_lag+1)
                        else:
                            lags_used = self.n_lag
                        inp_lag = [lags[-lag] for lag in lags_used]
                    else:
                        inp_lag = []
                    
                    # Similarly compute additional lag transforms if available
                    if self.lag_transform is not None:
                        transform_lag = []
                        for n, funcs in self.lag_transform.items():
                            series_array = np.array(pd.Series(lags).shift(n-1))
                            for func in funcs:
                                if isinstance(func, tuple):
                                    if func[0].__name__ == "rolling_quantile":
                                        t1 = func[0](series_array, func[1], func[2])[-1]
                                    else:
                                        t1 = func[0](series_array, func[1])[-1]
                                else:
                                    t1 = func(series_array, n)[-1]
                                transform_lag.append(t1)
                    else:
                        transform_lag = []
                        
                    # If using trend as a feature, add the forecasted trend component
                    if (self.trend) and (self.trend_type in ["feature_lr", "feature_ses"]):
                        trend_var = [trend_pred[i]]
                    else:
                        trend_var = []
                    
                    # Concatenate all features for the forecast step
                    inp = x_var + inp_lag + transform_lag + trend_var
                    # Ensure that the input is a DataFrame with the same columns as the training data
                    df_inp = pd.DataFrame(np.array(inp).reshape(1, -1), columns=self.X.columns)
                    # Get the forecast via the model
                    pred = self.model_fit.predict(df_inp)
                    predictions.append(pred)
                    lags.append(pred)  # update lag history

                forecasts = np.array(predictions)
                # Revert seasonal differencing if applied
                if self.season_diff is not None:
                    forecasts = invert_seasonal_diff(self.orig_d, forecasts, self.season_diff)
                # Revert ordinary differencing if applied
                if self.difference is not None:
                    forecasts = undiff_ts(self.orig, forecasts, self.difference)
                # Add static trend back if required
                if (self.trend) and (self.trend_type in ["linear","ses"]):
                    forecasts = trend_pred + forecasts
                # Ensure forecasts are nonnegative
                forecasts = np.array([max(0, x) for x in forecasts])
                # Finally, invert Box-Cox transform if it was applied
                if self.box_cox:
                    forecasts = back_box_cox_transform(y_pred=forecasts, lmda=self.lamda,
                                                        shift=self.is_zero,
                                                        box_cox_biasadj=self.biasadj)
                return forecasts

            def cv(self, df, cv_split, test_size, metrics, params=None):
                """
                Run cross-validation using time series splits.

                Args:
                    df (pd.DataFrame): Input data.
                    cv_split (int): Number of splits in TimeSeriesSplit.
                    test_size (int): Size of test window.
                    metrics (list): List of metric functions.
                    params (dict, optional): Model parameters.
                
                Returns:
                    pd.DataFrame: Performance metrics for CV.
                """
                tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
                self.metrics_dict = {m.__name__: [] for m in metrics}
                for train_index, test_index in tscv.split(df):
                    train, test = df.iloc[train_index], df.iloc[test_index]
                    x_test = test.drop(columns=[self.target_col])
                    y_test = np.array(test[self.target_col])
                    if params is not None:
                        self.fit(train, param=params)
                    else:
                        self.fit(train)
                    bb_forecast = self.forecast(test_size, x_test=x_test)
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

        else:
            def data_prep(self, df):
                """
                Prepare the data and handle categorical encoding, lag generation, trend removal, and differencing.

                Args:
                    df (pd.DataFrame): Raw input dataframe.

                Returns:
                    pd.DataFrame: Processed dataframe.
                """
                dfc = df.copy()
                # Handle categorical variables
                if self.cat_variables is not None:
                    if self.target_encode ==True:
                        for col in self.cat_variables:
                            encode_col = col+"_target_encoded"
                            dfc[encode_col] = kfold_target_encoder(dfc, col, self.target_col, 36)
                        self.df_encode = dfc.copy()
                        dfc = dfc.drop(columns = self.cat_variables)
                    # If target encoding is not used, convert categories to dummies    
                    else:
                        for col, cat in self.cat_var.items():
                            dfc[col] = dfc[col].astype('category')
                            # Set categories for categorical columns
                            dfc[col] = dfc[col].cat.set_categories(cat)
                        dfc = pd.get_dummies(dfc)

                        for i in self.drop_categ:
                            dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)

                if self.target_col in dfc.columns:
                    # Apply Box–Cox transformation if specified
                    if self.box_cox:
                        self.is_zero = np.any(np.array(dfc[self.target_col]) < 1) # check for zero or negative values
                        trans_data, self.lmda = box_cox_transform(x=dfc[self.target_col],
                                                                shift=self.is_zero,
                                                                box_cox_lmda=self.lmda)
                        dfc[self.target_col] = trans_data
                    # Detrend the series if specified
                    if self.trend:
                        self.len = len(dfc)
                        if self.trend_type in ["linear", "feature_lr"]:
                            self.lr_model = LinearRegression().fit(np.arange(self.len).reshape(-1, 1), dfc[self.target_col])
                            if self.trend_type == "linear":
                                dfc[self.target_col] = dfc[self.target_col] - self.lr_model.predict(np.arange(self.len).reshape(-1, 1))
                        if self.trend_type in ["ses", "feature_ses"]:
                            self.ses_model = ExponentialSmoothing(dfc[self.target_col], **self.ets_model).fit(**self.ets_fit)
                            if self.trend_type == "ses":
                                dfc[self.target_col] = dfc[self.target_col] - self.ses_model.fittedvalues.values

                    # Apply differencing if specified
                    if self.difference is not None or self.season_diff is not None:
                        self.orig = dfc[self.target_col].tolist()
                        if self.difference is not None:
                            dfc[self.target_col] = np.diff(dfc[self.target_col], n=self.difference,
                                                        prepend=np.repeat(np.nan, self.difference))
                        if self.season_diff is not None:
                            self.orig_d = dfc[self.target_col].tolist()
                            dfc[self.target_col] = seasonal_diff(dfc[self.target_col], self.season_diff)

                    # Create lag features based on n_lag parameter
                    if self.n_lag is not None:
                        if isinstance(self.n_lag, int):
                            lags = range(1, self.n_lag + 1)
                        else:
                            lags = self.n_lag
                        for lag in lags:
                            dfc[f"{self.target_col}_lag_{lag}"] = dfc[self.target_col].shift(lag)
                    # Create additional lag transformations if specified
                    if self.lag_transform is not None:
                        for n, funcs in self.lag_transform.items():
                            df_array = np.array(dfc[self.target_col].shift(n))
                            for func in funcs:
                                if isinstance(func, tuple):
                                    if func[0].__name__ == "rolling_quantile":
                                        dfc[f"q_{func[2]}_{n}_{func[1]}"] = func[0](df_array, func[1], func[2])
                                    else:
                                        dfc[f"{func[0].__name__}_{n}_{func[1]}"] = func[0](df_array, func[1])
                                else:
                                    dfc[f"{func.__name__}_{n}"] = func(np.array(dfc[self.target_col]), n)
                if self.trend:
                    if self.target_col in dfc.columns:
                        if self.trend_type == "feature_lr":
                            dfc["trend"] = self.lr_model.predict(np.arange(self.len).reshape(-1, 1))
                        if self.trend_type == "feature_ses":
                            dfc["trend"] = self.ses_model.fittedvalues.values
                return dfc.dropna()

            def fit(self, df, param=None):
                """
                Fit the ml model.

                Args:
                    df (pd.DataFrame): Input dataframe.
                    param (dict, optional): Parameters for XGBRegressor.
                """

                model_ = self.model(**param) if param is not None else self.model()
                if self.cat_variables is not None and not self.target_encode:
                    # If categorical variables are provided, create a dictionary of categories
                    self.cat_var = {c: sorted(df[c].drop_duplicates().tolist()) for c in self.cat_variables}
                    # Create a list of the first category for each categorical variable
                    self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
                model_train = self.data_prep(df)
                self.X = model_train.drop(columns=self.target_col)
                self.y = model_train[self.target_col]
                self.model_fit = model_.fit(self.X, self.y, verbose=True)

            def forecast(self, n_ahead, x_test=None):
                """
                Forecast future values for n_ahead periods.

                Args:
                    n_ahead (int): Number of periods to forecast.
                    x_test (pd.DataFrame, optional): Exogenous variables.

                Returns:
                    np.array: Forecasted values.
                """
                if x_test is not None:  # if external regressors are provided
                    if self.cat_variables and self.target_encode:
                        for col in self.cat_variables:
                            encode_col = col + "_target_encoded"
                            x_test[encode_col] = target_encoder_for_test(self.df_encode, x_test, col)
                        x_dummy = x_test.drop(columns=self.cat_variables)
                    else:
                        x_dummy = self.data_prep(x_test)
                lags = self.y.tolist()
                predictions = []
                if self.trend:
                    if self.trend_type in ["linear", "feature_lr"]:
                        trend_pred = self.lr_model.predict(np.arange(self.len, self.len + n_ahead).reshape(-1, 1))
                    elif self.trend_type in ["ses", "feature_ses"]:
                        trend_pred = self.ses_model.forecast(n_ahead).values
                # Forecast recursively one step at a time
                for i in range(n_ahead):
                    if x_test is not None:
                        x_var = x_dummy.iloc[i, :].tolist()
                    else:
                        x_var = []
                    if self.n_lag is not None:
                        if isinstance(self.n_lag, int):
                            lags_used = range(1, self.n_lag + 1)
                        else:
                            lags_used = self.n_lag
                        inp_lag = [lags[-lag] for lag in lags_used]
                    else:
                        inp_lag = []
                    if self.lag_transform is not None:
                        transform_lag = []
                        for n, funcs in self.lag_transform.items():
                            series_array = np.array(pd.Series(lags).shift(n - 1))
                            for func in funcs:
                                if not isinstance(func, tuple):
                                    t1 = func(np.array(lags), n - 1)[-1]
                                else:
                                    if func[0].__name__ == "rolling_quantile":
                                        t1 = func[0](series_array, func[1], func[2])[-1]
                                    else:
                                        t1 = func[0](series_array, func[1])[-1]
                                transform_lag.append(t1)
                    else:
                        transform_lag = []
                    if self.trend:
                        if self.trend_type in ["feature_lr", "feature_ses"]:
                            trend_var = [trend_pred[i]]
                        else:
                            trend_var = []
                    else:
                        trend_var = []
                    inp = x_var + inp_lag + transform_lag + trend_var
                    df_inp = pd.DataFrame(inp).T
                    df_inp.columns = self.X.columns
                    pred = self.model_fit.predict(df_inp)[0]
                    predictions.append(pred)
                    lags.append(pred)
                forecasts = np.array(predictions)
                # Revert seasonal differencing if applied
                if self.season_diff is not None:
                    forecasts = invert_seasonal_diff(self.orig_d, forecasts, self.season_diff)
                if self.difference is not None:
                    forecasts = undiff_ts(self.orig, forecasts, self.difference)
                if self.trend and self.trend_type in ["linear", "ses"]:
                    forecasts = trend_pred + forecasts
                forecasts = np.array([max(0, x) for x in forecasts])
                if self.box_cox:
                    forecasts = back_box_cox_transform(y_pred=forecasts,
                                                    lmda=self.lmda,
                                                    shift=self.is_zero,
                                                    box_cox_biasadj=self.biasadj)
                return forecasts

            def cv(self, df, cv_split, test_size, metrics, params=None):
                """
                Cross-validate the XGBoost model with time series split.

                Args:
                    df (pd.DataFrame): Input dataframe.
                    cv_split (int): Number of folds.
                    test_size (int): Forecast window for each split.
                    metrics (list): List of evaluation metric functions.
                    params (dict, optional): Hyperparameters for model training.

                Returns:
                    pd.DataFrame: CV performance metrics.
                """
                tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
                self.metrics_dict = {m.__name__: [] for m in metrics}
                self.cv_df = pd.DataFrame()
                self.cv_forecats_df = pd.DataFrame()
                for i, (train_index, test_index) in enumerate(tscv.split(df)):
                    train, test = df.iloc[train_index], df.iloc[test_index]
                    x_test = test.drop(columns=self.target_col)
                    y_test = np.array(test[self.target_col])
                    if params is not None:
                        self.fit(train, param=params)
                    else:
                        self.fit(train)
                    forecast_vals = self.forecast(test_size, x_test=x_test)
                    forecat_df = test[self.target_col].to_frame()
                    forecat_df["forecasts"] = forecast_vals
                    self.cv_forecats_df = pd.concat([self.cv_forecats_df, forecat_df], axis=0)
                    for m in metrics:
                        if m.__name__ == 'mean_squared_error':
                            val = m(y_test, forecast_vals, squared=False)
                        elif m.__name__ in ['MeanAbsoluteScaledError', 'MedianAbsoluteScaledError']:
                            val = m(y_test, forecast_vals, np.array(train[self.target_col]))
                        else:
                            val = m(y_test, forecast_vals)
                        self.metrics_dict[m.__name__].append(val)
                    cv_tr_df = pd.DataFrame({"feat_name": self.model_fit.feature_names_in_, 
                                            "importance": self.model_fit.feature_importances_}).sort_values(by="importance", ascending=False)
                    cv_tr_df["fold"] = i
                    self.cv_df = pd.concat([self.cv_df, cv_tr_df], axis=0)
                overall = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]
                return pd.DataFrame(overall).rename(columns={0: "eval_metric", 1: "score"})
        
class ml_bidirect_forecaster:
    """
    Bidirectional ml Forecaster for time-series forecasting.

    Args:
         target_cols (list): Names of the target variables.
         cat_variables (list, optional): List of categorical variable names.
         n_lag (dict, optional): Dictionary specifying the number of lags or list of lags for each target variable. Default is None. Example: {'target1': 3, 'target2': [1, 2, 3]}.
         difference (dict, optional): Dictionary specifying the order of ordinary differencing for each target variable. Default is None. Example: {'target1': 1, 'target2': 2}.
         seasonal_length (dict, optional): Seasonal differencing period. Example: {'target1': 7, 'target2': 7}.
         trend (dict, optional): Flag indicating if trend handling is applied. Default is False. Example: {'Target1': True, 'Target2': False}.
         trend_type (dict, optional): Trend handling strategy; one of 'linear', 'feature_lr', 'ses', or 'feature_ses'. Example: {'Target1': 'linear', 'Target2': 'feature_lr'}.
         ets_params (dict, optional): Dictionary of ETS model parameters (values are tuples of dictionaries of params) and fit settings for each target variable. Example: {'Target1': ({'trend': 'add', 'seasonal': 'add'}, {'damped_trend': True}), 'Target2': ({'trend': 'mul', 'seasonal': 'mul'}, {'damped_trend': False})}.
         target_encode (dict, optional): Flag determining if target encoding is used for categorical features for each target variable. Default is False. Example: {'Target1': True, 'Target2': False}.
         box_cox (dict, optional): Whether to apply a Box–Cox transformation for each target variable. Default is False. Example: {'Target1': True, 'Target2': False}.
         box_cox_lmda (dict, optional): Lambda parameter for the Box–Cox transformation for each target variable. Example: {'Target1': 0.5, 'Target2': 0.5}.
         box_cox_biasadj (dict, optional): Whether to adjust bias when inverting the Box–Cox transform for each target variable. Default is False. Example: {'Target1': True, 'Target2': False}.
         lag_transform (dict, optional): Dictionary specifying additional lag transformation functions for each target variable. Example: {'Target1': [func1, func2], 'Target2': [func3]}.
    """
    def __init__(self, model, target_cols, cat_variables=None, n_lag=None, difference=None, seasonal_length=None,
                 trend=False, trend_type="linear", ets_params=None, target_encode=False,
                 box_cox=False, box_cox_lmda=None, box_cox_biasadj=False, lag_transform=None):
        if n_lag is None and lag_transform is None:
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = model
        self.target_cols = target_cols
        self.cat_variables = cat_variables
        self.n_lag = n_lag
        self.difference = difference
        self.season_diff = seasonal_length
        self.trend = trend
        self.trend_type = trend_type
        if ets_params is not None:
            self.ets_model1 = ets_params[target_cols[0]][0]
            self.ets_fit1 = ets_params[target_cols[0]][1]
            self.ets_model2 = ets_params[target_cols[1]][0]
            self.ets_fit2 = ets_params[target_cols[1]][1]
        self.target_encode = target_encode
        self.box_cox = box_cox
        self.lmda = box_cox_lmda
        self.biasadj = box_cox_biasadj
        self.lag_transform = lag_transform
        self.tuned_params = None
        self.actuals = None
        self.prob_forecasts = None

        if self.model.__name__ in ["CatBoostRegressor", "LGBMRegressor"]:

            def data_prep(self, df):
                """
                Prepare the data and handle categorical encoding, lag generation, trend removal, and differencing.
                """
                dfc = df.copy()

                # Box-Cox transformation if flag is set
                if self.box_cox is not None:
                    if self.box_cox[self.target_cols[0]]:
                        self.is_zero1 = np.any(np.array(dfc[self.target_cols[0]]) < 1)
                        trans_data1, self.lamda1 = box_cox_transform(x=dfc[self.target_cols[0]],
                                                                    shift=self.is_zero1,
                                                                    box_cox_lmda=self.lamda[self.target_cols[0]])
                        dfc[self.target_cols[0]] = trans_data1
                    if self.box_cox[self.target_cols[1]]:
                        self.is_zero2 = np.any(np.array(dfc[self.target_cols[1]]) < 1)
                        trans_data2, self.lamda2 = box_cox_transform(x=dfc[self.target_cols[1]],
                                                                    shift=self.is_zero2,
                                                                    box_cox_lmda=self.lamda[self.target_cols[1]])
                        dfc[self.target_cols[1]] = trans_data2

                # Handle trend removal if specified
                if self.trend is not None:
                    self.len = len(dfc)
                    if self.trend[0]:
                        self.lr_model1 = LinearRegression().fit(np.arange(self.len).reshape(-1, 1), dfc[self.target_cols[0]])

                        if self.trend_type[0] in ["linear", "feature_lr"]:
                            dfc[self.target_cols[0]] = dfc[self.target_cols[0]] - self.lr_model1.predict(np.arange(self.len).reshape(-1, 1))
                        if self.trend_type[0] in ["ses", "feature_ses"]:
                            self.ses_model1 = ExponentialSmoothing(dfc[self.target_cols[0]], **self.ets_model1).fit(**self.ets_fit1)
                            dfc[self.target_cols[0]] = dfc[self.target_cols[0]] - self.ses_model1.fittedvalues.values

                    if self.trend[1]:
                        self.lr_model2 = LinearRegression().fit(np.arange(self.len).reshape(-1, 1), dfc[self.target_cols[1]])

                        if self.trend_type[1] in ["linear", "feature_lr"]:
                            dfc[self.target_cols[1]] = dfc[self.target_cols[1]] - self.lr_model2.predict(np.arange(self.len).reshape(-1, 1))

                        if self.trend_type[1] in ["ses", "feature_ses"]:
                            self.ses_model2 = ExponentialSmoothing(dfc[self.target_cols[1]], **self.ets_model2).fit(**self.ets_fit2)
                            dfc[self.target_cols[1]] = dfc[self.target_cols[1]] - self.ses_model2.fittedvalues.values
                # Handle differencing if specified
                if self.difference is not None:
                    if isinstance(self.difference, dict):
                        if self.difference[self.target_cols[0]] is not None:
                            self.orig1 = dfc[self.target_cols[0]].tolist()
                            dfc[self.target_cols[0]] = np.diff(dfc[self.target_cols[0]], n=self.difference[self.target_cols[0]],
                                                            prepend=np.repeat(np.nan, self.difference[self.target_cols[0]]))
                        if self.difference[self.target_cols[1]] is not None:
                            self.orig2 = dfc[self.target_cols[1]].tolist()
                            dfc[self.target_cols[1]] = np.diff(dfc[self.target_cols[1]], n=self.difference[self.target_cols[1]],
                                                            prepend=np.repeat(np.nan, self.difference[self.target_cols[1]]))
                if self.season_diff is not None:
                    if isinstance(self.season_diff, dict):
                        if self.season_diff[self.target_cols[0]] is not None:
                            self.orig_d1 = dfc[self.target_cols[0]].tolist()
                            dfc[self.target_cols[0]] = seasonal_diff(dfc[self.target_cols[0]], self.season_diff[self.target_cols[0]])
                        if self.season_diff[self.target_cols[1]] is not None:
                            self.orig_d2 = dfc[self.target_cols[1]].tolist()
                            dfc[self.target_cols[1]] = seasonal_diff(dfc[self.target_cols[1]], self.season_diff[self.target_cols[1]])

                # Process categorical variables if provided
                if self.cat_var is not None:
                    for col in self.cat_var:
                        dfc[col] = dfc[col].astype('category')

                # Create lag features based on n_lag parameter
                if self.n_lag is not None:
                    if isinstance(self.n_lag, dict):
                        for target in self.target_cols:
                            if isinstance(self.n_lag[target], int):
                                lags = range(1, self.n_lag[target] + 1)
                            else:
                                lags = self.n_lag[target]
                            for lag in lags:
                                dfc[f"{target}_lag_{lag}"] = dfc[target].shift(lag)
                    else:
                        raise ValueError("n_lag should be a dictionary with target column names as keys.")
                
                # Create additional lag transformations if specified (check this later)
                if self.lag_transform is not None:
                    for target, funcs in self.lag_transform.items():
                        if target not in self.target_cols:
                            raise ValueError(f"Target column {target} not found in the dataframe.")
                        for func in funcs:
                            df_array = np.array(dfc[target].shift(funcs[0])) # func[0] is the shift value
                            if isinstance(func, tuple):
                                if func[1].__name__ == "rolling_quantile":
                                    dfc[f"q_{func[3]}_{func[2]}_{func[0]}"] = func[1](df_array, func[2], func[3]) # The first element of tuple is shift value, the second element of the tuple is a function, the third is the window, the fourth is the quantile. EX: (1, rolling_quantile, 30, 0.5) will create a feature with rolling quantile of the target variable shifted by 1.
                                # write a condition for zero and 1 values (Croston, 2023) here later
                                # elif func[1].__name__ == "zeroCumulative" or func[1].__name__ == "nzInterval":
                                #     dfc[f"{func[1].__name__}_{func[2]}"] = func[1](df_array, func[2])
                                else:
                                    dfc[f"{func[1].__name__}_{func[2]}_{func[0]}"] = func[1](df_array, func[2]) # The first elemet of the tuple is shift value, the second element of the tuple is a function, the second is the window. EX: (1, rolling_mean, 30) will create a feature with rolling mean of the target variable shifted by 1.
                
                        
                # Add trend features if specified
                if self.trend is not None:
                    # if self.target_cols[0] in dfc.columns:
                    if self.trend_type[0] == "feature_lr":
                        dfc["trend1"] = self.lr_model1.predict(np.arange(self.len).reshape(-1, 1))
                    if self.trend_type[0] == "feature_ses":
                        dfc["trend1"] = self.ses_model1.fittedvalues.values
                    # if self.target_cols[1] in dfc.columns:
                    if self.trend_type[1] == "feature_lr":
                        dfc["trend2"] = self.lr_model2.predict(np.arange(self.len).reshape(-1, 1))
                    if self.trend_type[1] == "feature_ses":
                        dfc["trend2"] = self.ses_model2.fittedvalues.values

                return dfc.dropna()
            
            def fit(self, df, param=None):
                # Fit the model to the dataframe
                if param is not None:
                    model1_ = self.model(**param)
                    model2_ = self.model(**param)
                else:
                    model1_ = self.model()
                    model2_ = self.model()
                model_train = self.data_prep(df)
                self.X = model_train.drop(columns=self.target_cols)
                self.y1 = model_train[self.target_cols[0]]
                self.y2 = model_train[self.target_cols[1]]
                self.model1_fit = model1_.fit(self.X, self.y1, cat_features=self.cat_variables)
                self.model2_fit = model2_.fit(self.X, self.y2, cat_features=self.cat_variables)
            
            def forecast(self, n_ahead, x_test=None):
                """
                Forecast future values for n_ahead periods.

                Args:
                    n_ahead (int): Number of periods to forecast.
                    x_test (pd.DataFrame, optional): Exogenous variables.

                Returns:
                    np.array: Forecasted values.
                """
                target1_lags = self.y1.tolist()
                target2_lags = self.y2.tolist()
                tar1_forecasts = []
                tar2_forecasts = []

                if self.trend is not None:
                    if self.trend_type[0] in ["linear", "feature_lr"]:
                        trend_pred1 = self.lr_model1.predict(np.arange(self.len, self.len + n_ahead).reshape(-1, 1))
                    elif self.trend_type[0] in ["ses", "feature_ses"]:
                        trend_pred1 = self.ses_model1.forecast(n_ahead).values

                    if self.trend_type[1] in ["linear", "feature_lr"]:
                        trend_pred2 = self.lr_model2.predict(np.arange(self.len, self.len + n_ahead).reshape(-1, 1))
                    elif self.trend_type[1] in ["ses", "feature_ses"]:
                        trend_pred2 = self.ses_model2.forecast(n_ahead).values

                # Forecast recursively one step at a time
                for i in range(n_ahead):
                    if x_test is not None:
                        x_var = x_test.iloc[i, :].tolist()
                    else:
                        x_var = []

                    if self.n_lag is not None:
                        # For the first target variable
                        if isinstance(self.n_lag[self.target_cols[0]], int):
                            lags1 = range(1, self.n_lag[self.target_cols[0]] + 1)
                        else:
                            lags1 = self.n_lag[self.target_cols[0]]
                        inp_lag1 = [target1_lags[-lag] for lag in lags1]

                        # For the second target variable
                        if isinstance(self.n_lag[self.target_cols[1]], int):
                            lags2 = range(1, self.n_lag[self.target_cols[1]] + 1)
                        else:
                            lags2 = self.n_lag[self.target_cols[1]]
                        inp_lag2 = [target2_lags[-lag] for lag in lags2]
                    else:
                        inp_lag1 = []
                        inp_lag2 = []

                    if self.lag_transform is not None:
                        transform_lag = []
                        for target, funcs in self.lag_transform.items():
                            if target not in self.target_cols:
                                raise ValueError(f"Target column {target} not found in the dataframe.")
                            for func in funcs:
                                series_array = np.array(pd.Series(target1_lags if target == self.target_cols[0] else target2_lags).shift(funcs[0]))
                                if not isinstance(func, tuple):
                                    if func[1].__name__ == "rolling_quantile":
                                        t1 = func[1](series_array, funcs[2], funcs[3])[-1]
                                    else:
                                        t1 = func[1](series_array, funcs[2])[-1]
                                transform_lag.append(t1)
                    else:
                        transform_lag = []

                    inp = x_var + inp_lag1 + inp_lag2 + transform_lag
                    df_inp = pd.DataFrame(inp).T
                    df_inp.columns = self.X.columns
                    pred1 = self.model1_fit.predict(df_inp)[0]
                    tar1_forecasts.append(pred1)
                    target1_lags.append(pred1)
                    pred2 = self.model2_fit.predict(df_inp)[0]
                    tar2_forecasts.append(pred2)
                    target2_lags.append(pred2)
                forecasts1 = np.array(tar1_forecasts)
                forecasts2 = np.array(tar2_forecasts)
                # Revert seasonal differencing if applied
                if self.season_diff is not None:
                    forecasts1 = invert_seasonal_diff(self.orig_d1, forecasts1, self.season_diff[self.target_cols[0]])
                    forecasts2 = invert_seasonal_diff(self.orig_d2, forecasts2, self.season_diff[self.target_cols[1]])
                if self.difference is not None:
                    forecasts1 = undiff_ts(self.orig1, forecasts1, self.difference[self.target_cols[0]])
                    forecasts2 = undiff_ts(self.orig2, forecasts2, self.difference[self.target_cols[1]])
                if self.trend is not None:
                    if self.trend_type[0] in ["linear", "ses"]:
                        forecasts1 = trend_pred1 + forecasts1
                    if self.trend_type[1] in ["linear", "ses"]:
                        forecasts2 = trend_pred2 + forecasts2
                forecasts1 = np.array([max(0, x) for x in forecasts1])
                forecasts2 = np.array([max(0, x) for x in forecasts2])
                if self.box_cox is not None:
                    if self.box_cox[self.target_cols[0]]:
                        forecasts1 = back_box_cox_transform(y_pred=forecasts1,
                                                            lmda=self.lamda1,
                                                            shift=self.is_zero1,
                                                            box_cox_biasadj=self.biasadj[self.target_cols[0]])
                    if self.box_cox[self.target_cols[1]]:
                        forecasts2 = back_box_cox_transform(y_pred=forecasts2,
                                                            lmda=self.lamda2,
                                                            shift=self.is_zero2,
                                                            box_cox_biasadj=self.biasadj[self.target_cols[1]])
                return forecasts1, forecasts2
            
            def cv(self, df, cv_split, test_size, metrics, params=None):
                """"
                cross-validate the bidirectional CatBoost model with time series split.
                Args:
                    df (pd.DataFrame): Input dataframe.
                    cv_split (int): Number of folds.
                    test_size (int): Forecast window for each split.
                    metrics (list): List of evaluation metric functions.
                    params (dict, optional): Hyperparameters for model training.
                Returns:
                    pd.DataFrame: CV performance metrics for each target variable.
                """
                tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
                self.metrics_dict = {m.__name__: [] for m in metrics}
                self.cv_fi = pd.DataFrame()
                self.cv_forecasts_df = pd.DataFrame()
                for i, (train_index, test_index) in enumerate(tscv.split(df)):
                    train, test = df.iloc[train_index], df.iloc[test_index]
                    x_test = test.drop(columns=self.target_cols)
                    y_test1 = np.array(test[self.target_cols[0]])
                    y_test2 = np.array(test[self.target_cols[1]])
                    if params is not None:
                        self.fit(train, param=params)
                    else:
                        self.fit(train)
                    forecast_vals1, forecast_vals2 = self.forecast(test_size, x_test=x_test)
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
        else:
            # For models like XGBRegressor, LGBMRegressor, etc.
            def data_prep(self, df):
                # Prepare the data for modeling
                dfc = df.copy()
                if self.cat_variables is not None:
                    for col, cat in self.cat_var.items():
                        dfc[col] = dfc[col].astype('category')
                        # Set categories for categorical columns
                        dfc[col] = dfc[col].cat.set_categories(cat)
                    dfc = pd.get_dummies(dfc)

                    for i in self.drop_categ:
                        dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)

                if (self.target_cols[0] in dfc.columns) or (self.target_cols[1] in dfc.columns):
                    # Apply Box–Cox transformation if specified
                    if self.box_cox[self.target_cols[0]]:
                        self.is_zero1 = np.any(np.array(dfc[self.target_cols[0]]) < 1)
                        trans_data1, self.lmda1 = box_cox_transform(x=dfc[self.target_cols[0]],
                                                                    shift=self.is_zero1,
                                                                    box_cox_lmda=self.lmda[self.target_cols[0]])
                        dfc[self.target_cols[0]] = trans_data1
                    if self.box_cox[self.target_cols[1]]:
                        self.is_zero2 = np.any(np.array(dfc[self.target_cols[1]]) < 1)
                        trans_data2, self.lmda2 = box_cox_transform(x=dfc[self.target_cols[1]],
                                                                    shift=self.is_zero2,
                                                                    box_cox_lmda=self.lmda[self.target_cols[1]])
                        dfc[self.target_cols[1]] = trans_data2
                # Handle trend removal if specified
                if self.trend is not None:
                    self.len = len(dfc)
                    if self.trend[0]:
                        self.lr_model1 = LinearRegression().fit(np.arange(self.len).reshape(-1, 1), dfc[self.target_cols[0]])

                        if self.trend_type[0] in ["linear", "feature_lr"]:
                            dfc[self.target_cols[0]] = dfc[self.target_cols[0]] - self.lr_model1.predict(np.arange(self.len).reshape(-1, 1))
                        if self.trend_type[0] in ["ses", "feature_ses"]:
                            self.ses_model1 = ExponentialSmoothing(dfc[self.target_cols[0]], **self.ets_model1).fit(**self.ets_fit1)
                            dfc[self.target_cols[0]] = dfc[self.target_cols[0]] - self.ses_model1.fittedvalues.values

                    if self.trend[1]:
                        self.lr_model2 = LinearRegression().fit(np.arange(self.len).reshape(-1, 1), dfc[self.target_cols[1]])

                        if self.trend_type[1] in ["linear", "feature_lr"]:
                            dfc[self.target_cols[1]] = dfc[self.target_cols[1]] - self.lr_model2.predict(np.arange(self.len).reshape(-1, 1))

                        if self.trend_type[1] in ["ses", "feature_ses"]:
                            self.ses_model2 = ExponentialSmoothing(dfc[self.target_cols[1]], **self.ets_model2).fit(**self.ets_fit2)
                            dfc[self.target_cols[1]] = dfc[self.target_cols[1]] - self.ses_model2.fittedvalues.values
                # Handle differencing if specified
                if self.difference is not None:
                    if isinstance(self.difference, dict):
                        if self.difference[self.target_cols[0]] is not None:
                            self.orig1 = dfc[self.target_cols[0]].tolist()
                            dfc[self.target_cols[0]] = np.diff(dfc[self.target_cols[0]], n=self.difference[self.target_cols[0]],
                                                            prepend=np.repeat(np.nan, self.difference[self.target_cols[0]]))
                        if self.difference[self.target_cols[1]] is not None:
                            self.orig2 = dfc[self.target_cols[1]].tolist()
                            dfc[self.target_cols[1]] = np.diff(dfc[self.target_cols[1]], n=self.difference[self.target_cols[1]],
                                                            prepend=np.repeat(np.nan, self.difference[self.target_cols[1]]))
                if self.season_diff is not None:
                    if isinstance(self.season_diff, dict):
                        if self.season_diff[self.target_cols[0]] is not None:
                            self.orig_d1 = dfc[self.target_cols[0]].tolist()
                            dfc[self.target_cols[0]] = seasonal_diff(dfc[self.target_cols[0]], self.season_diff[self.target_cols[0]])
                        if self.season_diff[self.target_cols[1]] is not None:
                            self.orig_d2 = dfc[self.target_cols[1]].tolist()
                            dfc[self.target_cols[1]] = seasonal_diff(dfc[self.target_cols[1]], self.season_diff[self.target_cols[1]])

                # Create lag features based on n_lag parameter
                if self.n_lag is not None:
                    if isinstance(self.n_lag, dict):
                        for target in self.target_cols:
                            if isinstance(self.n_lag[target], int):
                                lags = range(1, self.n_lag[target] + 1)
                            else:
                                lags = self.n_lag[target]
                            for lag in lags:
                                dfc[f"{target}_lag_{lag}"] = dfc[target].shift(lag)
                    else:
                        raise ValueError("n_lag should be a dictionary with target column names as keys.")
                
                # Create additional lag transformations if specified (check this later)
                if self.lag_transform is not None:
                    for target, funcs in self.lag_transform.items():
                        if target not in self.target_cols:
                            raise ValueError(f"Target column {target} not found in the dataframe.")
                        for func in funcs:
                            df_array = np.array(dfc[target].shift(funcs[0])) # func[0] is the shift value
                            if isinstance(func, tuple):
                                if func[1].__name__ == "rolling_quantile":
                                    dfc[f"q_{func[3]}_{func[2]}_{func[0]}"] = func[1](df_array, func[2], func[3]) # The first element of tuple is shift value, the second element of the tuple is a function, the third is the window, the fourth is the quantile. EX: (1, rolling_quantile, 30, 0.5) will create a feature with rolling quantile of the target variable shifted by 1.
                                # write a condition for zero and 1 values (Croston, 2023) here later
                                # elif func[1].__name__ == "zeroCumulative" or func[1].__name__ == "nzInterval":
                                #     dfc[f"{func[1].__name__}_{func[2]}"] = func[1](df_array, func[2])
                                else:
                                    dfc[f"{func[1].__name__}_{func[2]}_{func[0]}"] = func[1](df_array, func[2]) # The first elemet of the tuple is shift value, the second element of the tuple is a function, the second is the window. EX: (1, rolling_mean, 30) will create a feature with rolling mean of the target variable shifted by 1.
                
                        
                # Add trend features if specified
                if self.trend is not None:
                    # if self.target_cols[0] in dfc.columns:
                    if self.trend_type[0] == "feature_lr":
                        dfc["trend1"] = self.lr_model1.predict(np.arange(self.len).reshape(-1, 1))
                    if self.trend_type[0] == "feature_ses":
                        dfc["trend1"] = self.ses_model1.fittedvalues.values
                    # if self.target_cols[1] in dfc.columns:
                    if self.trend_type[1] == "feature_lr":
                        dfc["trend2"] = self.lr_model2.predict(np.arange(self.len).reshape(-1, 1))
                    if self.trend_type[1] == "feature_ses":
                        dfc["trend2"] = self.ses_model2.fittedvalues.values

                return dfc.dropna()
            
            def fit(self, df, param = None):
                """
                Fit the model to the dataframe.

                Args:
                    df (pd.DataFrame): Input dataframe.
                    param (dict, optional): Hyperparameters for model training.

                Returns:
                    None
                """
                if param is not None:
                    model1_ = self.model(**param)
                    model2_ = self.model(**param)
                else:
                    model1_ = self.model()
                    model2_ = self.model()
                if self.cat_variables is not None:
                    self.cat_var = {c: sorted(df[c].drop_duplicates().tolist(), key=lambda x: x[0]) for c in self.cat_variables}
                    self.drop_categ= [sorted(df[i].drop_duplicates().tolist(), key=lambda x: x[0])[0] for i in self.cat_variables]
                model_train = self.data_prep(df)
                self.X = model_train.drop(columns=self.target_cols)
                self.y1 = model_train[self.target_cols[0]]
                self.y2 = model_train[self.target_cols[1]]
                self.model1_fit = model1_.fit(self.X, self.y1, verbose=True)
                self.model2_fit = model2_.fit(self.X, self.y2, verbose=True)
            
            def forecast(self, n_ahead, x_test=None):
                """
                Forecast future values for n_ahead periods.

                Args:
                    n_ahead (int): Number of periods to forecast.
                    x_test (pd.DataFrame, optional): Exogenous variables.

                Returns:
                    np.array: Forecasted values.
                """
                if x_test is not None:
                    x_dummy = self.data_prep(x_test)

                target1_lags = self.y1.tolist()
                target2_lags = self.y2.tolist()
                tar1_forecasts = []
                tar2_forecasts = []

                if self.trend is not None:
                    if self.trend_type[0] in ["linear", "feature_lr"]:
                        trend_pred1 = self.lr_model1.predict(np.arange(self.len, self.len + n_ahead).reshape(-1, 1))
                    elif self.trend_type[0] in ["ses", "feature_ses"]:
                        trend_pred1 = self.ses_model1.forecast(n_ahead).values

                    if self.trend_type[1] in ["linear", "feature_lr"]:
                        trend_pred2 = self.lr_model2.predict(np.arange(self.len, self.len + n_ahead).reshape(-1, 1))
                    elif self.trend_type[1] in ["ses", "feature_ses"]:
                        trend_pred2 = self.ses_model2.forecast(n_ahead).values

                # Forecast recursively one step at a time
                for i in range(n_ahead):
                    if x_test is not None:
                        x_var = x_test.iloc[i, :].tolist()
                    else:
                        x_var = []

                    if self.n_lag is not None:
                        # For the first target variable
                        if isinstance(self.n_lag[self.target_cols[0]], int):
                            lags1 = range(1, self.n_lag[self.target_cols[0]] + 1)
                        else:
                            lags1 = self.n_lag[self.target_cols[0]]
                        inp_lag1 = [target1_lags[-lag] for lag in lags1]

                        # For the second target variable
                        if isinstance(self.n_lag[self.target_cols[1]], int):
                            lags2 = range(1, self.n_lag[self.target_cols[1]] + 1)
                        else:
                            lags2 = self.n_lag[self.target_cols[1]]
                        inp_lag2 = [target2_lags[-lag] for lag in lags2]
                    else:
                        inp_lag1 = []
                        inp_lag2 = []

                    if self.lag_transform is not None:
                        transform_lag = []
                        for target, funcs in self.lag_transform.items():
                            if target not in self.target_cols:
                                raise ValueError(f"Target column {target} not found in the dataframe.")
                            for func in funcs:
                                series_array = np.array(pd.Series(target1_lags if target == self.target_cols[0] else target2_lags).shift(funcs[0]))
                                if not isinstance(func, tuple):
                                    if func[1].__name__ == "rolling_quantile":
                                        t1 = func[1](series_array, funcs[2], funcs[3])[-1]
                                    else:
                                        t1 = func[1](series_array, funcs[2])[-1]
                                transform_lag.append(t1)
                    else:
                        transform_lag = []

                    inp = x_var + inp_lag1 + inp_lag2 + transform_lag
                    df_inp = pd.DataFrame(inp).T
                    df_inp.columns = self.X.columns
                    pred1 = self.model1_fit.predict(df_inp)[0]
                    tar1_forecasts.append(pred1)
                    target1_lags.append(pred1)
                    pred2 = self.model2_fit.predict(df_inp)[0]
                    tar2_forecasts.append(pred2)
                    target2_lags.append(pred2)
                forecasts1 = np.array(tar1_forecasts)
                forecasts2 = np.array(tar2_forecasts)
                # Revert seasonal differencing if applied
                if self.season_diff is not None:
                    forecasts1 = invert_seasonal_diff(self.orig_d1, forecasts1, self.season_diff[self.target_cols[0]])
                    forecasts2 = invert_seasonal_diff(self.orig_d2, forecasts2, self.season_diff[self.target_cols[1]])
                if self.difference is not None:
                    forecasts1 = undiff_ts(self.orig1, forecasts1, self.difference[self.target_cols[0]])
                    forecasts2 = undiff_ts(self.orig2, forecasts2, self.difference[self.target_cols[1]])
                if self.trend is not None:
                    if self.trend_type[0] in ["linear", "ses"]:
                        forecasts1 = trend_pred1 + forecasts1
                    if self.trend_type[1] in ["linear", "ses"]:
                        forecasts2 = trend_pred2 + forecasts2
                forecasts1 = np.array([max(0, x) for x in forecasts1])
                forecasts2 = np.array([max(0, x) for x in forecasts2])
                if self.box_cox is not None:
                    if self.box_cox[self.target_cols[0]]:
                        forecasts1 = back_box_cox_transform(y_pred=forecasts1,
                                                            lmda=self.lamda1,
                                                            shift=self.is_zero1,
                                                            box_cox_biasadj=self.biasadj[self.target_cols[0]])
                    if self.box_cox[self.target_cols[1]]:
                        forecasts2 = back_box_cox_transform(y_pred=forecasts2,
                                                            lmda=self.lamda2,
                                                            shift=self.is_zero2,
                                                            box_cox_biasadj=self.biasadj[self.target_cols[1]])
                return forecasts1, forecasts2
            
            def cv(self, df, cv_split, test_size, metrics, params=None):
                """"
                cross-validate the bidirectional XGBoost model with time series split.
                Args:
                    df (pd.DataFrame): Input dataframe.
                    cv_split (int): Number of folds.
                    test_size (int): Forecast window for each split.
                    metrics (list): List of evaluation metric functions.
                    params (dict, optional): Hyperparameters for model training.
                Returns:
                    pd.DataFrame: CV performance metrics for each target variable.
                """
                tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
                self.metrics_dict = {m.__name__: [] for m in metrics}
                self.cv_fi = pd.DataFrame()
                self.cv_forecasts_df = pd.DataFrame()
                for i, (train_index, test_index) in enumerate(tscv.split(df)):
                    train, test = df.iloc[train_index], df.iloc[test_index]
                    x_test = test.drop(columns=self.target_cols)
                    y_test1 = np.array(test[self.target_cols[0]])
                    y_test2 = np.array(test[self.target_cols[1]])
                    if params is not None:
                        self.fit(train, param=params)
                    else:
                        self.fit(train)
                    forecast_vals1, forecast_vals2 = self.forecast(test_size, x_test=x_test)
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

class VARModel:
    """
    Vector Autoregressive Model class supporting data preprocessing, fitting, forecasting, and cross-validation.

    Parameters
    ----------
    target_cols : List[str]
        Target columns for the VAR model (dependent variables).
    lags : Dict[str, List[int]]
        Dictionary specifying lags for each target variable. For example, {'target1': [1, 2], 'target2': [1, 2, 3]} or {'target1': 3, 'target2': 5}.
    lag_transform : Optional[Dict[str, List[tuple]]] (default=None)
        Dictionary specifying lag transformations per target (e.g. rolling, quantile), each as a list of tuples:
        (lag, func, window, [quantile]). For example, { 'target1': [(1, rolling_mean, 30), (2, rolling_quantile, 30, 0.5)] }.
    difference : Optional[Dict[str, int]] (default=None)
        Dictionary specifying order of differencing for each variable. For example, {'target1': 1, 'target2': 2}.
    seasonal_diff : Optional[Dict[str, int]] (default=None)
        Dictionary specifying seasonal differencing for each variable. For example, {'target1': 12, 'target2': 7}.
    trend : Optional[Dict[str, bool]] (default=None)
        Dictionary specifying which variables require detrending.
    trend_types : Optional[Dict[str, str]] (default=None)
        Dictionary specifying trend type for each variable: "linear", "ses", "feature_lr", or "feature_ses".
    ets_params : Optional[Dict[str, tuple]] (default=None)
        Dictionary specifying params for ExponentialSmoothing per variable.
        For example, {'target1': ({'trend': 'add', 'seasonal': 'add'}, {'damped_trend': True}), 'target2': ({'trend': 'add', 'seasonal': 'add'}, {'damped_trend': True})}.
    box_cox : Optional[Dict[str, bool]] (default=None)
        Dictionary specifying which variables require Box-Cox transform.
    box_cox_lmda : Optional[Dict[str, float]] (default=None)
        Dictionary of Box-Cox lambdas for each variable.
    box_cox_biasadj : bool or Dict[str, bool] (default=False)
        Whether to use bias adjustment for Box-Cox for each variable.
    add_constant : bool (default=True)
        If True, add a constant column to exogenous variables.
    cat_variables : Optional[List[str]] (default=None)
        List of categorical columns to one-hot encode.
    verbose : bool (default=False)
        If True, print verbose messages.

    Methods
    -------
    data_prep(df)
        Prepare the data for VAR model.
    fit(df_train)
        Fit the VAR model to training data.
    forecast(H, exog=None)
        Forecast H steps ahead.
    predict(X)
        Predict with model coefficients.
    cv_var(df, target_col, cv_split, test_size, metrics)
        Cross-validate VAR model.

    Notes
    -----
    - Assumes external utility functions exist for Box-Cox, seasonal differencing, etc.
    """

    def __init__(
        self,
        target_cols: List[str],
        lags: Dict[str, List[int]],
        lag_transform: Optional[Dict[str, List[tuple]]] = None,
        difference: Optional[Dict[str, int]] = None,
        seasonal_diff: Optional[Dict[str, int]] = None,
        trend: Optional[Dict[str, bool]] = None,
        trend_types: Optional[Dict[str, str]] = None,
        ets_params: Optional[Dict[str, tuple]] = None,
        box_cox: Optional[Dict[str, bool]] = None,
        box_cox_lmda: Optional[Dict[str, float]] = None,
        box_cox_biasadj: Any = False,
        add_constant: bool = True,
        cat_variables: Optional[List[str]] = None,
        verbose: bool = False
    ):
        self.target_cols = target_cols
        self.lags = lags
        self.lag_transform = lag_transform
        self.diffs = difference
        self.season_diffs = seasonal_diff
        self.trend = trend
        self.trend_types = trend_types
        self.ets_params = ets_params
        self.box_cox = box_cox
        self.lamdas = box_cox_lmda
        self.biasadj = box_cox_biasadj
        self.cons = add_constant
        self.cat_variables = cat_variables
        self.verbose = verbose

        # Handle box_cox bias adjustment dict default
        if self.box_cox is not None and not isinstance(self.box_cox, dict):
            raise TypeError("box_cox must be a dictionary of target values")
        if isinstance(self.box_cox, dict) and not isinstance(self.biasadj, dict):
            self.biasadj = {k: False for k in self.box_cox}
        
        # Handle trend default types
        if self.trend is not None and not isinstance(self.trend, dict):
            raise TypeError("trend must be a dictionary of target values")
        if self.trend is not None and self.trend_types is None:
            self.trend_types = {k: "linear" for k in self.trend}

    def data_prep(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the data according to the model configuration.
        Applies categorical encoding, Box-Cox, detrending, differencing, seasonal differencing, lags, and lag transforms.
        Drops rows with any NaN after transformation.
        """
        dfc = df.copy()
        # Handle categorical variables
        if self.cat_variables is not None:
            # self.cat_var = {c: sorted(dfc[c].drop_duplicates().tolist()) for c in self.cat_variables}
            # self.drop_categ = [self.cat_var[c][0] for c in self.cat_variables]
            for col, cats in self.cat_var.items():
                dfc[col] = pd.Categorical(dfc[col], categories=cats)
            dfc = pd.get_dummies(dfc)
            for i in self.drop_categ:
                dfc.drop(list(dfc.filter(regex=i)), axis=1, inplace=True)
        
        # Check all target columns exist
        if not all(col in dfc.columns for col in self.target_cols):

            # Box-Cox transformation
            if self.box_cox is not None:
                if self.lamdas is None:
                    self.lamdas = {i: None for i in self.box_cox}
                self.is_zeros = {i: None for i in self.lamdas}
                for k, lm in self.lamdas.items():
                    self.is_zeros[k] = (dfc[k] < 1).any()
                    trans_data, self.lamdas[k] = box_cox_transform(
                        x=dfc[k], shift=self.is_zeros[k], box_cox_lmda=lm
                    )
                    if self.box_cox.get(k, False):
                        dfc[k] = trans_data

            # Detrending
            if self.trend is not None:
                self.tr_models = {i: None for i in self.trend_types}
                self.len = len(dfc)
                for k, tr in self.trend_types.items():
                    if tr in ["linear", "feature_lr"]:
                        lr = LinearRegression()
                        lr.fit(np.arange(self.len).reshape(-1, 1), dfc[k])
                        self.tr_models[k] = lr
                        if tr == "linear":
                            dfc[k] = dfc[k] - lr.predict(np.arange(self.len).reshape(-1, 1))
                    elif tr in ["ses", "feature_ses"]:
                        ets = ExponentialSmoothing(dfc[k], **self.ets_params[k][0])
                        fit = ets.fit(**self.ets_params[k][1])
                        self.tr_models[k] = fit
                        if tr == "ses":
                            dfc[k] = dfc[k] - fit.fittedvalues.values

            # Differencing
            if self.diffs is not None:
                self.origs = {i: dfc[i].tolist() for i in self.diffs}
                for x, d in self.diffs.items():
                    dfc[x] = np.diff(dfc[x], n=d, prepend=np.repeat(np.nan, d))

            # Seasonal differencing
            if self.season_diffs is not None:
                self.orig_ds = {i: dfc[i].tolist() for i in self.season_diffs}
                for w, s in self.season_diffs.items():
                    dfc[w] = seasonal_diff(dfc[w], s)

            # Lag features
            if self.lags is not None:
                for a, lags in self.lags.items():
                    lag_used = lags if isinstance(lags, list) else range(1, lags + 1) # Ensure lags is a list, even if a single int
                    for lg in lag_used:
                        dfc[f"{a}_lag_{lg}"] = dfc[a].shift(lg)

            # Lag transforms
            if self.lag_transform is not None:
                for n, transforms in self.lag_transform.items():
                    for f in transforms:
                        df_array = np.array(dfc[n].shift(f[0]))
                        if f[1].__name__ == "rolling_quantile":
                            dfc[f"q_{f[3]}_{n}_{f[0]}_w{f[2]}"] = f[1](df_array, f[2], f[3])
                        else:
                            dfc[f"{f[1].__name__}_{n}_{f[0]}_{f[2]}"] = f[1](df_array, f[2])

        dfc = dfc.dropna()
        return dfc

    def fit(self, df_train: pd.DataFrame) -> None:
        """
        Fit the VAR model to the training data.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data.
        """
        if self.cat_variables is not None:
            self.cat_var = {c: sorted(df_train[c].drop_duplicates().tolist()) for c in self.cat_variables}
            self.drop_categ = [self.cat_var[c][0] for c in self.cat_variables]

        df = self.data_prep(df_train)
        X = df.drop(columns=self.target_cols)
        if self.cons:
            X = sm.add_constant(X)
        self.X = np.array(X)
        self.y = np.array(df[self.target_cols])
        self.coeffs = np.linalg.lstsq(self.X, self.y, rcond=None)[0]

    def predict(self, X: List[float]) -> np.ndarray:
        """
        Predict the model output for input X.

        Parameters
        ----------
        X : pd.DataFrame or List[float] or np.ndarray
            Feature DataFrame for prediction.

        Returns
        -------
        np.ndarray
            Model predictions for each target.
        """
        arr = np.array(X)
        return np.dot(self.coeffs.T, arr.T)

    def forecast(self, H: int, exog: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Forecast H steps ahead.

        Parameters
        ----------
        H : int
            Number of steps to forecast.
        exog : Optional[pd.DataFrame]
            Exogenous variables for forecasting.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of forecasts for each target variable.
        """
        y_lists = {j: self.y[:, i].tolist() for i, j in enumerate(self.target_cols)} # Initialize lists for each target variable
        if exog is not None:
            if self.cons:
                if exog.shape[0] == 1:
                    exog.insert(0, 'const', 1)
                else:
                    exog = sm.add_constant(exog)
            exog = np.array(self.data_prep(exog))

        forecasts = {i: [] for i in self.target_cols}

        # Trend forecasting
        if self.trend is not None:
            trend_preds = {i: [] for i in self.trend_types}
            for k, tr in self.trend_types.items():
                if tr in ["linear", "feature_lr"]:
                    trend_preds[k] = self.tr_models[k].predict(np.arange(self.len, self.len+H).reshape(-1, 1))
                elif tr in ["ses", "feature_ses"]:
                    trend_preds[k] = self.tr_models[k].forecast(H).values

        for t in range(H):
            # Exogenous input for step t
            if exog is not None:
                exo_inp = exog[t].tolist()
            else:
                exo_inp = [1] if self.cons else []

            # Lagged features
            lags = []
            if self.lag_dict is not None:
                for tr, v in y_lists.items():
                    if self.lag_dict[tr]:
                        ys = [v[-x] for x in self.lag_dict[tr]]
                        lags += ys

            # Lag transforms
            transform_lag = []
            if self.lag_transform is not None:
                for n, transforms in self.lag_transform.items():
                    for f in transforms:
                        df_array = np.array(pd.Series(lags).shift(f[0]-1))
                        if f[1].__name__ == "rolling_quantile":
                            t1 = f[1](df_array, f[2], f[3])[-1]
                        else:
                            t1 = f[1](df_array, f[2])[-1]
                        transform_lag.append(t1)

            # Trend feature
            trend_var = []
            if self.trend is not None:
                for k, tr in self.trend_types.items():
                    if tr in ["feature_lr", "feature_ses"]:
                        trend_var.append(trend_preds[k][t])

            inp = exo_inp + lags + transform_lag + trend_var
            pred = self.predict(inp)
            for id_, ff in enumerate(forecasts):
                forecasts[ff].append(pred[id_])
                y_lists[ff].append(pred[id_])

        # Invert seasonal difference
        if self.season_diffs is not None:
            for s in self.orig_ds:
                forecasts[s] = invert_seasonal_diff(self.orig_ds[s], np.array(forecasts[s]), self.season_diffs[s])

        # Invert difference
        if self.diffs is not None:
            for d in self.diffs:
                forecasts[d] = undiff_ts(self.origs[d], np.array(forecasts[d]), self.diffs[d])

        # Add back trend
        if self.trend is not None:
            for k, tr in self.trend_types.items():
                if tr in ["linear", "ses"]:
                    forecasts[k] = trend_preds[k] + forecasts[k]

        # Non-negativity
        for f in forecasts:
            forecasts[f] = np.array([max(0, x) for x in forecasts[f]])

        # Invert Box-Cox
        if self.box_cox is not None:
            for k, lmd in self.lamdas.items():
                if self.box_cox.get(k, False):
                    forecasts[k] = back_box_cox_transform(
                        y_pred=forecasts[k], lmda=lmd, shift=self.is_zeros[k], box_cox_biasadj=self.biasadj[k]
                    )

        return forecasts

    def cv_var(
        self,
        df: pd.DataFrame,
        target_col: str,
        cv_split: int,
        test_size: int,
        metrics: List[Callable]
    ) -> pd.DataFrame:
        """
        Perform cross-validation for VAR model.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        target_col : str
            Target variable for evaluation.
        cv_split : int
            Number of cross-validation folds.
        test_size : int
            Test size per fold.
        metrics : List[Callable]
            List of metric functions.

        Returns
        -------
        pd.DataFrame
            DataFrame with averaged cross-validation metric scores.
        """
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        self.metrics_dict = {m.__name__: [] for m in metrics}
        self.cv_forecasts_df = pd.DataFrame()

        for i, (train_index, test_index) in enumerate(tscv.split(df)):
            train, test = df.iloc[train_index], df.iloc[test_index]
            x_test, y_test = test.drop(columns=self.target_cols), np.array(test[target_col])
            self.fit(train)
            bb_forecast = self.forecast(H=test_size, exog=x_test)[target_col]

            forecast_df = test[target_col].to_frame()
            forecast_df["forecasts"] = bb_forecast
            self.cv_forecasts_df = pd.concat([self.cv_forecasts_df, forecast_df], axis=0)

            for m in metrics:
                if m.__name__ == 'mean_squared_error':
                    eval_score = m(y_test, bb_forecast, squared=False)
                elif m.__name__ in ['MeanAbsoluteScaledError', 'MedianAbsoluteScaledError']:
                    eval_score = m(y_test, bb_forecast, np.array(train[self.target_cols]))
                else:
                    eval_score = m(y_test, bb_forecast)
                self.metrics_dict[m.__name__].append(eval_score)

        overall_perform = [[m.__name__, np.mean(self.metrics_dict[m.__name__])] for m in metrics]
        return pd.DataFrame(overall_perform, columns=["eval_metric", "score"])
    
    