import numpy as np
import pandas as pd
from statsforecast.models import ARIMA, AutoARIMA, TBATS, AutoTBATS
class bag_boost_ts_conformalizer():
    def __init__(self, delta, train_df, n_windows, model, H, calib_metric = "mae", model_param=None):
        self.delta = delta
        self.model = model
        self.train_df = train_df
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.calib_metric = calib_metric
        self.param = model_param
        self.calibrate()
    def backtest(self):
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            x_back = self.train_df[:-self.H-i]
            if i !=0:
                test_y = self.train_df[-self.H-i:-i].iloc[:, 0]
                if len(self.train_df.columns)>1:
                    test_x = self.train_df[-self.H-i:-i].iloc[:, 1:]
                else:
                    test_x = None
            else:
                test_y = self.train_df[-self.H:].iloc[:, 0]
                if len(self.train_df.columns)>1:
                    test_x = self.train_df[-self.H:].iloc[:, 1:]
                else:
                    test_x = None
                
#             mod_arima = ARIMA(y_back, exog=x_back, order = (0,1,2), seasonal_order=(0,1,1, 7)).fit()
#             y_pred = mod_arima.forecast(self.H, exog = test_x)
            
            if self.param is not None:
                self.model.fit(x_back, param=self.param)
            else:
                self.model.fit(x_back)
            if test_x is not None:
                forecast = self.model.forecast(self.H, test_x)
            else:
                forecast = self.model.forecast(self.H)
            
            test_y = np.array(test_y)
            predictions.append(forecast)
            actuals.append(test_y)
            print("model "+str(i+1)+" is completed")
        return np.row_stack(actuals), np.row_stack(predictions)
    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta and non-conformity scores
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            # calculating metrics horizon i
            mae =np.abs(acts[:,i] - preds[:,i]) 
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    q_hat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    q_hat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self, X=None):
        if self.param is not None:
            self.model.fit(self.train_df, param=self.param)
        else:
            self.model.fit(self.train_df)
            
        if X is not None:
            y_pred = self.model.forecast(n_ahead = self.H, x_test= X)
        else:
            y_pred = self.model.forecast(n_ahead = self.H)
            
        result = []
        result.append(y_pred)
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs
    
class s_arima_conformalizer():
    def __init__(self, model, delta, n_windows, H, calib_metric = "mae"):
        self.delta = delta
        self.model = model.__class__
        self.order = model.order
        self.S_order = model.seasonal_order
    
        self.y_train = model.endog.flatten()
        self.x_train = model.exog
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.calib_metric = calib_metric
        self.model_fit = self.model(self.y_train, order= self.order, exog = self.x_train, seasonal_order= self.S_order).fit()
        self.calibrate()
    def backtest(self):
        #making H-step-ahead forecast n_windows times for each 1-step backward sliding window.
        # We can the think of n_windows as the size of calibration set for each H horizon 
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            y_back = self.y_train[:-self.H-i]
            if self.x_train is not None:
                x_back = self.x_train[:-self.H-i]
            else:
                x_back = None
            if i !=0:
                test_y = self.y_train[-self.H-i:-i]
                if self.x_train is not None:
                    test_x = self.x_train[-self.H-i:-i]
                else:
                    test_x = None
            else:
                test_y = self.y_train[-self.H:]
                if self.x_train is not None:
                    test_x = self.x_train[-self.H:]
                else:
                    test_x = None
                
            mod_arima = self.model(y_back, exog=x_back, order = self.order, seasonal_order=self.S_order).fit()
            y_pred = mod_arima.forecast(self.H, exog = test_x)
            
            predictions.append(y_pred)
            actuals.append(test_y)
            print("model "+str(i+1)+" is completed")
        return np.row_stack(actuals), np.row_stack(predictions)
    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta value
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        #Calculate non-conformity scores (mae, smape and mape for now) for each forecasted horizon
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            mae =np.abs(acts[:,i] - preds[:,i])
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat for all given delta values
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    qhat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    qhat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self, X = None):
        y_pred = self.model_fit.forecast(self.H, exog = X)

        result = []
        result.append(y_pred)
        #Calculate the prediction intervals given the calibration metric used for non-conformity score
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs
    
from statsmodels.tsa.holtwinters import ExponentialSmoothing
class ets_conformalizer():
    def __init__(self, train_data, delta, model_params, fit_params, n_windows, H, calib_metric = "mae"):
        self.delta = delta
        self.train = train_data
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.model_param = model_params
        self.fit_param = fit_params
        self.calib_metric = calib_metric
        self.model_fit = ExponentialSmoothing(self.train, **self.model_param).fit(**self.fit_param)
        self.calibrate()
    def backtest(self):
        #making H-step-ahead forecast n_windows times for each 1-step backward sliding window.
        # We can the think of n_windows as the size of calibration set for each H horizon 
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            y_train = self.train[:-self.H-i]
            if i !=0:
                y_test = self.train[-self.H-i:-i]
            else:
                y_test = self.train[-self.H:]

            model_ets = ExponentialSmoothing(y_train, **self.model_param)
            fit_ets = model_ets.fit(**self.fit_param)
    
            y_pred = fit_ets.forecast(self.H)
            
            predictions.append(y_pred)
            actuals.append(y_test)
            print("model "+str(i+1)+" is completed")
        return np.row_stack(actuals), np.row_stack(predictions)
    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta value
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        #Calculate non-conformity scores (mae, smape and mape for now) for each forecasted horizon
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            mae =np.abs(acts[:,i] - preds[:,i])
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat for all given delta values
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    qhat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    qhat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self):
        y_pred = self.model_fit.forecast(self.H)

        result = []
        result.append(y_pred)
        #Calculate the prediction intervals given the calibration metric used for non-conformity score
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs
    
class bidirect_ts_conformalizer():
    def __init__(self, delta, train_df, col_index, n_windows, model, H, calib_metric = "mae", model_param=None):
        self.delta = delta
        self.model = model
        self.train_df = train_df
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.calib_metric = calib_metric
        self.param = model_param
        self.col=col_index
        self.calibrate()
    def backtest(self):
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            x_back = self.train_df[:-self.H-i]
            if i !=0:
                test_y = self.train_df[-self.H-i:-i].iloc[:, self.col]
                if len(self.train_df.columns)>1:
                    test_x = self.train_df[-self.H-i:-i].iloc[:, 2:]
                else:
                    test_x = None
            else:
                test_y = self.train_df[-self.H:].iloc[:, self.col]
                if len(self.train_df.columns)>1:
                    test_x = self.train_df[-self.H:].iloc[:, 2:]
                else:
                    test_x = None
                
#             mod_arima = ARIMA(y_back, exog=x_back, order = (0,1,2), seasonal_order=(0,1,1, 7)).fit()
#             y_pred = mod_arima.forecast(self.H, exog = test_x)
            
            if self.param is not None:
                self.model.fit(x_back, param=self.param)
            else:
                self.model.fit(x_back)
            if test_x is not None:
                forecast = self.model.forecast(self.H, test_x)[self.col]
            else:
                forecast = self.model.forecast(self.H)[self.col]
            
            test_y = np.array(test_y)
            predictions.append(forecast)
            actuals.append(test_y)
            print("model "+str(i+1)+" is completed")
        return np.row_stack(actuals), np.row_stack(predictions)
    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta and non-conformity scores
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            # calculating metrics horizon i
            mae =np.abs(acts[:,i] - preds[:,i]) 
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    q_hat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    q_hat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self, X=None):
        if self.param is not None:
            self.model.fit(self.train_df, param=self.param)
        else:
            self.model.fit(self.train_df)
            
        if X is not None:
            y_pred = self.model.forecast(n_ahead = self.H, x_test= X)[self.col]
        else:
            y_pred = self.model.forecast(n_ahead = self.H)[self.col]
            
        result = []
        result.append(y_pred)
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs
    
from statsmodels.tsa.api import VAR
class var_conformalizer():
    def __init__(self, model_fit, delta, n_windows, H, col_index, calib_metric = "mae", non_stationary_series = None):
        self.delta = delta
        self.lag_order = model_fit.k_ar
        self.y_train = model_fit.endog
        self.x_train = model_fit.exog
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.col = col_index
        self.origin = non_stationary_series
        self.calib_metric = calib_metric
        self.model_fit = VAR(self.y_train, exog=self.x_train).fit(self.lag_order)
        self.calibrate()
    def backtest(self):
        #making H-step-ahead forecast n_windows times for each 1-step backward sliding window.
        # We can the think of n_windows as the size of calibration set for each H horizon 
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            y_back = self.y_train[:-self.H-i]
            if self.x_train is not None:
                x_back = self.x_train[:-self.H-i]
            else:
                x_back = None
            if i !=0:
                if self.origin is not None:
                    test_y = np.array(self.origin)[-self.H-i:-i]
                    last_train = np.array(self.origin)[:-self.H-i][-1]
                else:
                    test_y = self.y_train[-self.H-i:-i][:, self.col]
                if self.x_train is not None:
                    test_x = self.x_train[-self.H-i:-i]
                else:
                    test_x = None
            else:
                if self.origin is not None:
                    test_y = np.array(self.origin)[-self.H:]
                    last_train = np.array(self.origin)[:-self.H-i][-1]
                else:
                    test_y = self.y_train[-self.H:][:, self.col]
                if self.x_train is not None:
                    test_x = self.x_train[-self.H:]
                else:
                    test_x = None
                
            var_result = VAR(y_back, exog=x_back).fit(self.lag_order)
            y_pred = var_result.forecast(y = y_back[-self.lag_order:], steps = self.H, exog_future = test_x)[:, self.col]
            
            if self.origin is not None:
                pred_dif = np.insert(y_pred, 0, last_train)
                pred_var = np.cumsum(pred_dif)[-self.H:]
            else:
                pred_var = y_pred
            predictions.append(pred_var)
            actuals.append(test_y)
            print("model "+str(i+1)+" is completed")
        return np.row_stack(actuals), np.row_stack(predictions)
    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta value
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        #Calculate non-conformity scores (mae, smape and mape for now) for each forecasted horizon
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            mae =np.abs(acts[:,i] - preds[:,i])
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat for all given delta values
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    qhat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    qhat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self, X = None):
        fore_var = self.model_fit.forecast(y = self.y_train[-self.lag_order:], steps = self.H, exog_future = X)[:, self.col]
        if self.origin is not None:
            last_origin = np.array(self.origin)[-1]
            add_orig = np.insert(fore_var, 0, last_origin)
            y_pred = np.cumsum(add_orig)[-self.H:]
        else:
            y_pred = fore_var

        result = []
        result.append(y_pred)
        #Calculate the prediction intervals given the calibration metric used for non-conformity score
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs
    

class bag_boost_aggr_conformalizer():
    def __init__(self, delta, train_df, n_windows, models, cat_cols, H, calib_metric = "mae", model_param=None):
        self.delta = delta
        self.models = models
        self.train_df = train_df
        self.cats = cat_cols
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.calib_metric = calib_metric
        self.param = model_param
        self.cols = [m.target_col for m in self.models]
        
        self.models_f = self.models.copy()
        for i, j in zip(self.cols, range(len(self.cols))):
            train = pd.concat([self.train_df[i], self.train_df[self.cats]], axis=1)
            # self.models[i] = ExponentialSmoothing(self.train[i], **self.params[i][0]).fit(**self.params[i][1])
            self.models_f[j].fit(train, self.param[i])

        
        self.calibrate()
    def backtest(self):
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            sum_forecasts = np.zeros((self.H,))
            sum_actuals = np.zeros((self.H,))
            for m in self.models:
                bactest_df = pd.concat([self.train_df[m.target_col], self.train_df[self.cats]], axis=1)
                x_back = bactest_df[:-self.H-i]
                if i !=0:
                    test_y = bactest_df[-self.H-i:-i][m.target_col]
                    if len(bactest_df.columns)>1:
                        test_x = bactest_df[-self.H-i:-i].iloc[:, 1:]
                    else:
                        test_x = None
                else:
                    test_y = bactest_df[-self.H:][m.target_col]
                    if len(bactest_df.columns)>1:
                        test_x = bactest_df[-self.H:].iloc[:, 1:]
                    else:
                        test_x = None
                    

                
                if self.param is not None:
                    m.fit(x_back, param=self.param[m.target_col])
                else:
                    m.fit(x_back)
                if test_x is not None:
                    forecast = m.forecast(self.H, test_x)
                else:
                    forecast = m.forecast(self.H)
                sum_forecasts +=forecast
                sum_actuals +=np.array(test_y)
            
            predictions.append(sum_forecasts)
            actuals.append(sum_actuals)
            print("model "+str(i+1)+" is completed")
        self.predictions = np.row_stack(predictions)
        self.actuals = np.row_stack(actuals)
        return np.row_stack(actuals), np.row_stack(predictions)
    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta and non-conformity scores
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            # calculating metrics horizon i
            mae =np.abs(acts[:,i] - preds[:,i]) 
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    q_hat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    q_hat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self, X=None):

        y_pred = np.zeros((self.H,))
        for f in self.models_f:
            # y_pred += self.models_f[i].forecast(self.H,  x_test= X)
            if X is not None:
                y_pred += f.forecast(self.H,  x_test= X)
            else:
                y_pred += f.forecast(n_ahead = self.H)
            
        result = []
        result.append(y_pred)
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs
    
class bidirect_aggr_conformalizer():
    def __init__(self, delta, train_df, col_index, n_windows, models, cat_cols, H, calib_metric = "mae", model_param=None):
        self.delta = delta
        self.models = models
        self.cats = cat_cols
        self.idx = col_index
        self.train_df = train_df
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.calib_metric = calib_metric
        self.param = model_param
        # self.col=col_index
        
        self.models_f = self.models.copy()
        for f, j in zip(self.models_f, range(len(self.models_f))):
            train = pd.concat([self.train_df[f.target_col], self.train_df[self.cats]], axis=1)
            # self.models[i] = ExponentialSmoothing(self.train[i], **self.params[i][0]).fit(**self.params[i][1])
            self.models_f[j].fit(train, self.param[f.target_col[self.idx]])
        self.calibrate()
    def backtest(self):
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            sum_forecasts = np.zeros((self.H,))
            sum_actuals = np.zeros((self.H,))
            for m in self.models:
                bactest_df = pd.concat([self.train_df[m.target_col], self.train_df[self.cats]], axis=1)
                x_back = bactest_df[:-self.H-i]
                if i !=0:
                    test_y = bactest_df[-self.H-i:-i][m.target_col[self.idx]]
                    if len(bactest_df.columns)>1:
                        test_x = bactest_df[-self.H-i:-i].iloc[:, 2:]
                    else:
                        test_x = None
                else:
                    test_y = bactest_df[-self.H:][m.target_col[self.idx]]
                    if len(bactest_df.columns)>1:
                        test_x = bactest_df[-self.H:].iloc[:, 2:]
                    else:
                        test_x = None
                    
    #             mod_arima = ARIMA(y_back, exog=x_back, order = (0,1,2), seasonal_order=(0,1,1, 7)).fit()
    #             y_pred = mod_arima.forecast(self.H, exog = test_x)
                
                if self.param is not None:
                    m.fit(x_back, param=self.param[m.target_col[self.idx]])
                else:
                    m.fit(x_back)
                if test_x is not None:
                    forecast = m.forecast(self.H, test_x)[self.idx]
                else:
                    forecast = m.forecast(self.H)[self.idx]
                sum_forecasts +=forecast
                sum_actuals +=np.array(test_y)
            
            predictions.append(sum_forecasts)
            actuals.append(sum_actuals)
            print("model "+str(i+1)+" is completed")
        self.predictions = np.row_stack(predictions)
        self.actuals = np.row_stack(actuals)
        return np.row_stack(actuals), np.row_stack(predictions)
    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta and non-conformity scores
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            # calculating metrics horizon i
            mae =np.abs(acts[:,i] - preds[:,i]) 
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    q_hat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    q_hat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self, X=None):

        y_pred = np.zeros((self.H,))
        for f in self.models_f:
            # y_pred += self.models_f[i].forecast(self.H,  x_test= X)
            if X is not None:
                y_pred += f.forecast(self.H,  x_test= X)[self.idx]
            else:
                y_pred += f.forecast(n_ahead = self.H)[self.idx]
            
        result = []
        result.append(y_pred)
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs

class ets_aggr_conformalizer():
    def __init__(self, train_data, tar_cols, delta, params, n_windows, H, calib_metric = "mae"):
        self.delta = delta
        self.train = train_data
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.cols = tar_cols
        self.params = params
        self.calib_metric = calib_metric
        self.models = {}
        for i in self.cols:
            self.models[i] = ExponentialSmoothing(self.train[i], **self.params[i][0]).fit(**self.params[i][1])
        self.calibrate()
    def backtest(self):
        #making H-step-ahead forecast n_windows times for each 1-step backward sliding window.
        # We can the think of n_windows as the size of calibration set for each H horizon 
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            sum_forecasts = np.zeros((self.H,))
            sum_actuals = np.zeros((self.H,))
            for j in self.cols:
                y_train = self.train[:-self.H-i][j]
                if i !=0:
                    y_test = self.train[-self.H-i:-i][j]
                else:
                    y_test = self.train[-self.H:][j]
    
                model_ets = ExponentialSmoothing(y_train, **self.params[j][0])
                fit_ets = model_ets.fit(**self.params[j][1])
        
                y_pred = np.array(fit_ets.forecast(self.H))
                sum_forecasts +=y_pred
                sum_actuals +=np.array(y_test)
            
            predictions.append(sum_forecasts)
            actuals.append(sum_actuals)
            print("model "+str(i+1)+" is completed")
        self.predictions = np.row_stack(predictions)
        self.actuals = np.row_stack(actuals)
        return np.row_stack(actuals), np.row_stack(predictions)

    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta value
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        #Calculate non-conformity scores (mae, smape and mape for now) for each forecasted horizon
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            mae =np.abs(acts[:,i] - preds[:,i])
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat for all given delta values
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    qhat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    qhat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self):
        y_pred = np.zeros((self.H,))
        for i in self.cols:
            y_pred += np.array(self.models[i].forecast(self.H))

        result = []
        result.append(y_pred)
        #Calculate the prediction intervals given the calibration metric used for non-conformity score
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs
    
class s_arima_aggr_conformalizer():
    def __init__(self, train_data, tar_cols, orders, delta, n_windows, H, exog = None, calib_metric = "mae"):
        self.delta = delta
        self.orders = orders
        self.cols = tar_cols
        self.train = train_data
        self.exog = exog
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.calib_metric = calib_metric

        self.models = {}
        for i in self.cols:
            # self.models[i] = ExponentialSmoothing(self.train[i], **self.params[i][0]).fit(**self.params[i][1])
            self.models[i] = ARIMA(order=(self.orders[i][0],self.orders[i][5],self.orders[i][1]),
                                   seasonal_order=(self.orders[i][2],self.orders[i][6],self.orders[i][3]),
                                   season_length=self.orders[i][4]).fit(y=np.array(self.train[i]), X=np.array(self.exog, dtype = np.float64))

            # y_pred = arima_m.forecast(y=y_train, h=test_size, X=x_train, X_future=x_test)["mean"]
            # self.models[i] = self.models[i].fit(y=y_train, X=x_train)
            
        self.calibrate()
    def backtest(self):
        #making H-step-ahead forecast n_windows times for each 1-step backward sliding window.
        # We can the think of n_windows as the size of calibration set for each H horizon 
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            sum_forecasts = np.zeros((self.H,))
            sum_actuals = np.zeros((self.H,))
            for j in self.cols:
                y_back = np.array(self.train[:-self.H-i][j])
                if self.exog is not None:
                    x_back = np.array(self.exog[:-self.H-i], dtype = np.float64)
                else:
                    x_back = None
                if i !=0:
                    test_y = np.array(self.train[-self.H-i:-i][j])
                    if self.exog is not None:
                        test_x = np.array(self.exog[-self.H-i:-i], np.float64)
                    else:
                        test_x = None
                else:
                    test_y = np.array(self.train[-self.H:][j])
                    if self.exog is not None:
                        test_x = np.array(self.exog[-self.H:], np.float64)
                    else:
                        test_x = None
                    
                # mod_arima = self.model(y_back, exog=x_back, order = self.order, seasonal_order=self.S_order).fit()
                mod_arima = ARIMA(order=(self.orders[j][0],self.orders[j][5],self.orders[j][1]),
                                   seasonal_order=(self.orders[j][2],self.orders[j][6],self.orders[j][3]),
                                   season_length=self.orders[j][4]).fit(y=y_back, X=x_back)
                # y_pred = mod_arima.forecast(self.H, exog = test_x)
                y_pred = mod_arima.predict(self.H,  X=test_x)["mean"]
                sum_forecasts +=y_pred
                sum_actuals +=test_y
            
            predictions.append(sum_forecasts)
            actuals.append(sum_actuals)
            print("model "+str(i+1)+" is completed")
        self.predictions = np.row_stack(predictions)
        self.actuals = np.row_stack(actuals)
        return np.row_stack(actuals), np.row_stack(predictions)
    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta value
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        #Calculate non-conformity scores (mae, smape and mape for now) for each forecasted horizon
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            mae =np.abs(acts[:,i] - preds[:,i])
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat for all given delta values
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    qhat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    qhat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self, exog = None):
        y_pred = np.zeros((self.H,))
        for i in self.cols:
            y_pred += self.models[i].predict(self.H,  X=np.array(exog, np.float64))["mean"]
            # y_pred += np.array(self.models[i].forecast(self.H))

        result = []
        result.append(y_pred)
        #Calculate the prediction intervals given the calibration metric used for non-conformity score
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs
    
class var_aggr_conformalizer():
    def __init__(self, train_data, exogs, tar_columns, delta, n_windows, H, col_index, lags, diffs, calib_metric = "mae"):
        self.delta = delta
        self.lag_order = lags
        self.train = train_data
        self.exogs = exogs
        self.cols = tar_columns
        self.n_windows = n_windows
        self.n_calib = n_windows
        self.H = H
        self.idx = col_index
        self.diffs = diffs
        self.calib_metric = calib_metric
        self.models = {}
        self.last_vals = {i: [] for i in tar_columns}
        for i in self.cols:
            # self.models[i] = ExponentialSmoothing(self.train[i], **self.params[i][0]).fit(**self.params[i][1])
            tar_col = self.train.columns[self.train.columns.str.contains(i)].tolist()
            dfi = self.train[tar_col]
            if self.diffs[i][0] ==1:
                dfi[tar_col[0]] = dfi[tar_col[0]].diff()
                self.last_vals[i].append(self.train[tar_col[0]][-1])
            else:
                self.last_vals[i].append(None)
            if self.diffs[i][1] ==1:
                dfi[tar_col[1]] = dfi[tar_col[1]].diff()
                self.last_vals[i].append(self.train[tar_col[1]][-1])
            else:
                self.last_vals[i].append(None)
                
            if (self.diffs[i][0] ==1) | (self.diffs[i][1] ==1):
                exog_f = exogs[1:]
            else:
                exog_f = exogs
            dfi = dfi.dropna()

            self.models[i] = VAR(dfi, exog=exog_f).fit(self.lag_order)
        self.calibrate()
    def backtest(self):
        #making H-step-ahead forecast n_windows times for each 1-step backward sliding window.
        # We can the think of n_windows as the size of calibration set for each H horizon 
        actuals = []
        predictions = []
        for i in range(self.n_windows):
            sum_forecasts = np.zeros((self.H,))
            sum_actuals = np.zeros((self.H,))
            for j in self.cols:
                tar_col = self.train.columns[self.train.columns.str.contains(j)].tolist()
                dfj = self.train[tar_col]
                if self.diffs[j][0] ==1:
                    dfj[tar_col[0]] = dfj[tar_col[0]].diff()
                if self.diffs[j][1] ==1:
                    dfj[tar_col[1]] = dfj[tar_col[1]].diff()
                    
                y_back = dfj[:-self.H-i]
                x_back = self.exogs[:-self.H-i]
                
                if i !=0:
                    if self.diffs[j][self.idx] ==1:
                        test_y = np.array(self.train[tar_col[self.idx]])[-self.H-i:-i]
                        last_train = np.array(self.train[tar_col[self.idx]])[:-self.H-i][-1]
                    else:
                        test_y = np.array(self.train[tar_col[self.idx]])[-self.H-i:-i]
                   
                    test_x = self.exogs[-self.H-i:-i]

                else:
                    if self.diffs[j][self.idx] ==1:
                        test_y = np.array(self.train[tar_col[self.idx]])[-self.H:]
                        last_train = np.array(self.train[tar_col[self.idx]])[:-self.H-i][-1]
                    else:
                        test_y = np.array(self.train[tar_col[self.idx]])[-self.H:]
                    test_x = self.exogs[-self.H:]
                
                y_back = y_back.dropna()
                # y_back[self.idx].fillna(y_back[self.idx][17:33].mean())
                # y_back = y_back.fillna(y_back.rolling(window=16, min_periods=1).mean().shift(-16 + 1)

                if (self.diffs[j][0] ==1) | (self.diffs[j][1] ==1):
                    x_back = x_back[1:]
                    
                var_result = VAR(y_back, exog=x_back).fit(self.lag_order)
             
                y_pred = var_result.forecast(y = y_back.values[-self.lag_order:], steps = self.H, exog_future = np.array(test_x))[:, self.idx]
                
                if self.diffs[j][self.idx] ==1:
                    pred_dif = np.insert(y_pred, 0, last_train)
                    sum_forecasts += np.cumsum(pred_dif)[-self.H:]
                else:
                    sum_forecasts += y_pred
                    
                sum_forecasts +=y_pred
                sum_actuals +=test_y
            
            predictions.append(sum_forecasts)
            actuals.append(sum_actuals)
            print("model "+str(i+1)+" is completed")
        self.predictions = np.row_stack(predictions)
        self.actuals = np.row_stack(actuals)
        return np.row_stack(actuals), np.row_stack(predictions)
    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile values for each delta value
        delta_q = []
        for i in self.delta:
            which_quantile = np.ceil((i)*(self.n_calib+1))/self.n_calib
            q_data = np.quantile(scores_calib, which_quantile, method = "lower")
            delta_q.append(q_data)
        self.delta_q = delta_q
        return delta_q
    
    def non_conformity_func(self):
        #Calculate non-conformity scores (mae, smape and mape for now) for each forecasted horizon
        acts, preds = self.backtest()
        horizon_scores = []
        dists = []
        for i in range(self.H):
            mae =np.abs(acts[:,i] - preds[:,i])
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
            dist = 2*acts[:,i] - preds[:,i]
            dists.append(dist)
        self.cp_dist = np.stack(dists).T
        return horizon_scores
    
    
    def calibrate(self):
         # Calibrate the conformalizer to calculate q_hat for all given delta values
        scores_calib = self.non_conformity_func()
        self.q_hat_D = []
        for d in range(len(self.delta)):
            q_hat_H = []
            for i in range(self.H):
                scores_i = scores_calib[i]
                if self.calib_metric == "smape":
                    qhat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    qhat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self, X = None):
        y_pred = np.zeros((self.H,))
        for i in self.cols:
            y_train = self.models[i].endog
            # x_train = self.models[i].exog
            fore_var = self.models[i].forecast(y = y_train[-self.lag_order:], steps = self.H, exog_future = np.array(X))[:, self.idx]
            if self.diffs[i][self.idx] ==1:
                last_origin = self.last_vals[i][self.idx]
                add_orig = np.insert(fore_var, 0, last_origin)
                y_pred += np.cumsum(add_orig)[-self.H:]
            else:
                y_pred += fore_var

        result = []
        result.append(y_pred)
        #Calculate the prediction intervals given the calibration metric used for non-conformity score
        for i in range(len(self.delta)):
            if self.calib_metric == "mae":
                y_lower, y_upper = y_pred - np.array(self.q_hat_D[i]).flatten(), y_pred + np.array(self.q_hat_D[i]).flatten()
            elif self.calib_metric == "mape":
                y_lower, y_upper = y_pred/(1+np.array(self.q_hat_D[i]).flatten()), y_pred/(1-np.array(self.q_hat_D[i]).flatten())
            elif self.calib_metric == "smape":
                y_lower = y_pred*(2-np.array(self.q_hat_D[i]).flatten())/(2+np.array(self.q_hat_D[i]).flatten())
                y_upper = y_pred*(2+np.array(self.q_hat_D[i]).flatten())/(2-np.array(self.q_hat_D[i]).flatten())
            else:
                raise ValueError("not a valid metric")
            result.append(y_lower)
            result.append(y_upper)
        CPs = pd.DataFrame(result).T
        CPs.rename(columns = {0:"point_forecast"}, inplace = True)
        for i in range(0, 2*len(self.delta), 2):
            d_index = round(i/2)
            CPs.rename(columns = {i+1:"lower_"+str(round(self.delta[d_index]*100)), i+2:"upper_"+str(round(self.delta[d_index]*100))}, inplace = True)
        return CPs