import numpy as np
import pandas as pd
class ts_conformalizer():
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
                mod_fit = self.model.fit(x_back, param=self.param)
            else:
                mod_fit = self.model.fit(x_back)
            if test_x is not None:
                forecast = self.model.forecast(mod_fit, self.H, test_x)
            else:
                forecast = self.model.forecast(mod_fit, self.H)
            
            test_y = np.array(test_y)
            predictions.append(forecast)
            actuals.append(test_y)
            print("model "+str(i+1)+" is completed")
        return np.row_stack(actuals), np.row_stack(predictions)
    
    def calculate_qunatile(self, scores_calib):
        # Calculate the quantile value based on delta and non-conformity scores
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
                    middle_qhat = self.calculate_qunatile(scores_i[:, 0])[d]
                elif self.calib_metric == "mape":
                    middle_qhat = self.calculate_qunatile(scores_i[:, 1])[d]
                elif self.calib_metric == "mae":
                    q_hat = self.calculate_qunatile(scores_i[:, 2])[d]
                else:
                    raise ValueError("not a valid metric")
                q_hat_H.append(q_hat)
            self.q_hat_D.append(q_hat_H)
            
    def forecast(self, X=None):
        if self.param is not None:
            model = self.model.fit(self.train_df, param=self.param)
        else:
            model = self.model.fit(self.train_df)
            
        if X is not None:
            y_pred = self.model.forecast(model, n_ahead = self.H, x_test= X)
        else:
            y_pred = self.model.forecast(model, n_ahead = self.H)
            
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