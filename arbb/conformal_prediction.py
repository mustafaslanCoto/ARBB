import numpy as np
import pandas as pd
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
        for i in range(self.H):
            # calculating metrics horizon i
            mae =np.abs(acts[:,i] - preds[:,i]) 
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
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
        for i in range(self.H):
            mae =np.abs(acts[:,i] - preds[:,i])
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
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
        for i in range(self.H):
            mae =np.abs(acts[:,i] - preds[:,i])
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
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
        for i in range(self.H):
            # calculating metrics horizon i
            mae =np.abs(acts[:,i] - preds[:,i]) 
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
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
        from statsmodels.tsa.api import VAR
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
        for i in range(self.H):
            mae =np.abs(acts[:,i] - preds[:,i])
            smape = 2*mae/(np.abs(acts[:,i])+np.abs(preds[:,i]))
            mape = mae/acts[:,i]
            metrics = np.stack((smape,  mape, mae), axis=1)
            horizon_scores.append(metrics)
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