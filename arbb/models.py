import pandas as pd
import numpy as np


## Catboost
from sklearn.model_selection import TimeSeriesSplit
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope

class cat_forecaster:
    def __init__(self, model, target_col, n_lag = None, differencing_number = None, lag_transform = None, cat_variables = None):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = model
        self.target_col = target_col
        self.cat_var = cat_variables
        self.n_lag = n_lag
        self.difference = differencing_number
        self.lag_transform = lag_transform
        
    def data_prep(self, df):
        dfc = df.copy()
        if self.difference is not None:
            self.last_train = df[self.target_col].tolist()[-1]
            for i in range(1, self.difference+1):
                dfc[self.target_col] = dfc[self.target_col].diff(1)
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
            model_cat = self.model(**param)
        else:
            model_cat = self.model()

        model_df = self.data_prep(df)
        X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        self.model_cat = model_cat.fit(X, self.y, cat_features=self.cat_var, verbose = True)
    
    def forecast(self, n_ahead, x_test = None):
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
            pred = self.model_cat.predict(inp)
            predictions.append(pred)
            lags.append(pred)

        if self.difference is not None:
            predictions.insert(0, self.last_train)
            forecasts = np.cumsum(predictions)[-n_ahead:]
        else:
            forecasts = np.array(predictions)
        return forecasts
    

    def tune_model(self, df, cv_split, test_size, param_space, eval_metric, eval_num = 100):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)

        def objective(params):
            model =self.model(**params)

            
            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])
                model_train = self.data_prep(train)
                X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                self.model_cat = model.fit(X, self.y, cat_features=self.cat_var,
                            verbose = False)
                yhat = self.forecast(n_ahead =len(y_test), x_test=x_test)
                if eval_metric.__name__== 'mean_squared_error':
                    accuracy = eval_metric(y_test, yhat, squared=False)
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
    def __init__(self, model, target_col, n_lag = None, differencing_number = None, lag_transform = None, cat_variables = None):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = model
        self.target_col = target_col
        self.cat_var = cat_variables
        self.n_lag = n_lag
        self.difference = differencing_number
        self.lag_transform = lag_transform
        
    def data_prep(self, df):
        dfc = df.copy()
        if self.difference is not None:
            self.last_train = df[self.target_col].tolist()[-1]
            for i in range(1, self.difference+1):
                dfc[self.target_col] = dfc[self.target_col].diff(1)
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
            model_lgb = self.model(**param, verbose=-1)
        else:
            model_lgb = self.model(verbose=-1)

        model_df = self.data_prep(df)
        self.X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        self.model_lgb = model_lgb.fit(self.X, self.y, categorical_feature=self.cat_var)
    
    def forecast(self, n_ahead, x_test = None):
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
            pred = self.model_lgb.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)
        if self.difference is not None:
            predictions.insert(0, self.last_train)
            forecasts = np.cumsum(predictions)[-n_ahead:]
        else:
            forecasts = np.array(predictions)
        return forecasts
    
    def tune_model(self, df, cv_split, test_size, param_space,eval_metric, eval_num = 100):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            model =self.model(**params, verbose=-1)

            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])
                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                self.model_lgb = model.fit(self.X, self.y, categorical_feature=self.cat_var)
                yhat = self.forecast(n_ahead =len(y_test), x_test=x_test)
                if eval_metric.__name__== 'mean_squared_error':
                    accuracy = eval_metric(y_test, yhat, squared=False)
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
        # best_params = {i: int(best_hyperparams[i]) if i in ["num_iterations", "num_leaves", "max_depth","min_data_in_leaf", "top_k"] 
        #                    else best_hyperparams[i] for i in best_hyperparams}
        return space_eval(param_space, best_hyperparams)
            
    
class xgboost_forecaster:
    def __init__(self, model, target_col, n_lag = None, differencing_number = None, lag_transform = None, cat_dict = None, drop_categ = None):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = model
        self.target_col = target_col
        self.cat_var = cat_dict
        self.n_lag = n_lag
        self.difference = differencing_number
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
            if self.difference is not None:
                self.last_train = df[self.target_col].tolist()[-1]
                for i in range(1, self.difference+1):
                    dfc[self.target_col] = dfc[self.target_col].diff(1)
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
            model_xgb =self.model(**param)
        else:
            model_xgb =self.model()
        model_df = self.data_prep(df)
        self.X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        self.model_xgb = model_xgb.fit(self.X, self.y, verbose = True)

    def forecast(self, n_ahead, x_test = None):
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

            pred = self.model_xgb.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)

        if self.difference is not None:
            predictions.insert(0, self.last_train)
            forecasts = np.cumsum(predictions)[-n_ahead:]
        else:
            forecasts = np.array(predictions)
        return forecasts

    
    def tune_model(self, df, cv_split, test_size, param_space, eval_metric, eval_num= 100):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            model =self.model(**params)   
                
            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])
                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                self.model_xgb = model.fit(self.X, self.y, verbose = True)
                yhat = self.forecast(n_ahead =len(y_test), x_test=x_test)
                if eval_metric.__name__== 'mean_squared_error':
                    accuracy = eval_metric(y_test, yhat, squared=False)
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
        # best_params = {i: int(best_hyperparams[i]) if i in ["n_estimators", "max_depth"] 
        #                    else best_hyperparams[i] for i in best_hyperparams}
        return space_eval(param_space, best_hyperparams)
    
class RandomForest_forecaster:
    def __init__(self, model, target_col, n_lag, lag_transform = None, differencing_number = None, cat_dict = None, drop_categ = None):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = model
        self.target_col = target_col
        self.cat_var = cat_dict
        self.n_lag = n_lag
        self.difference = differencing_number
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

            if self.difference is not None:
                self.last_train = df[self.target_col].tolist()[-1]
                for i in range(1, self.difference+1):
                    dfc[self.target_col] = dfc[self.target_col].diff(1)

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
            model_rf =self.model(**param)
        else:
            model_rf =self.model()

        model_df = self.data_prep(df)
        self.X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        self.model_rf = model_rf.fit(self.X, self.y)

    def forecast(self, n_ahead, x_test = None):
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

            pred = self.model_rf.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)

        if self.difference is not None:
            predictions.insert(0, self.last_train)
            forecasts = np.cumsum(predictions)[-n_ahead:]
        else:
            forecasts = np.array(predictions)
        return forecasts

    
    def tune_model(self, df, cv_split, test_size, param_space, eval_metric, eval_num= 100):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            model =self.model(**params)   
                
            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])
                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                self.model_rf = model.fit(self.X, self.y)
                yhat = self.forecast(n_ahead =len(y_test), x_test=x_test)
                if eval_metric.__name__== 'mean_squared_error':
                    accuracy = eval_metric(y_test, yhat, squared=False)
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
        # best_params = {i: int(best_hyperparams[i]) if i in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"] 
        #                    else best_hyperparams[i] for i in best_hyperparams}
        return space_eval(param_space, best_hyperparams)
    
class AdaBoost_forecaster:
    def __init__(self, model, target_col, n_lag, lag_transform = None, differencing_number = None, cat_dict = None, drop_categ = None):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = model
        self.target_col = target_col
        self.cat_var = cat_dict
        self.n_lag = n_lag
        self.difference = differencing_number
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

            if self.difference is not None:
                self.last_train = df[self.target_col].tolist()[-1]
                for i in range(1, self.difference+1):
                    dfc[self.target_col] = dfc[self.target_col].diff(1)

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
            model_ada = self.model(**param)
        else:
            model_ada = self.model()
        model_df = self.data_prep(df)
        self.X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        self.model_ada = model_ada.fit(self.X, self.y)

    def forecast(self, n_ahead, x_test = None):
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

            pred = self.model_ada.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)
#             lags = lags[-max_lag:]
        if self.difference is not None:
            predictions.insert(0, self.last_train)
            forecasts = np.cumsum(predictions)[-n_ahead:]
        else:
            forecasts = np.array(predictions)
        return forecasts

    
    def tune_model(self, df, cv_split, test_size, param_space, eval_metric, eval_num= 100):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            model =self.model(**params)   
                
            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])
                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                self.model_ada = model.fit(self.X, self.y)
                yhat = self.forecast(n_ahead =len(y_test), x_test=x_test)
                if eval_metric.__name__== 'mean_squared_error':
                    accuracy = eval_metric(y_test, yhat, squared=False)
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
#         best_params = {i: int(best_hyperparams[i]) if i in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"] 
#                            else best_hyperparams[i] for i in best_hyperparams}
        return space_eval(param_space, best_hyperparams)
    
class Cubist_forecaster:
    def __init__(self, model, target_col, n_lag = None, differencing_number = None, lag_transform = None, cat_dict = None, drop_categ = None):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = model
        self.target_col = target_col
        self.cat_var = cat_dict
        self.n_lag = n_lag
        self.difference = differencing_number
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

            if self.difference is not None:
                self.last_train = df[self.target_col].tolist()[-1]
                for i in range(1, self.difference+1):
                    dfc[self.target_col] = dfc[self.target_col].diff(1)
                    
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
            model_cub =self.model(**param)
        else:
            model_cub =self.model()
        model_df = self.data_prep(df)
        self.X, self.y = model_df.drop(columns =self.target_col), model_df[self.target_col]
        self.model_cub = model_cub.fit(self.X, self.y)

    def forecast(self, n_ahead, x_test = None):
        x_dummy = self.data_prep(x_test)

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

            pred = self.model_cub.predict(df_inp)[0]
            predictions.append(pred)
            lags.append(pred)

        if self.difference is not None:
            predictions.insert(0, self.last_train)
            forecasts = np.cumsum(predictions)[-n_ahead:]
        else:
            forecasts = np.array(predictions)
        return forecasts

    
    def tune_model(self, df, cv_split, test_size, param_space, eval_metric, eval_num= 100):
        tscv = TimeSeriesSplit(n_splits=cv_split, test_size=test_size)
        
        def objective(params):
            model =self.model(**params)   
                
            metric = []
            for train_index, test_index in tscv.split(df):
                train, test = df.iloc[train_index], df.iloc[test_index]
                x_test, y_test = test.iloc[:, 1:], np.array(test[self.target_col])
                model_train = self.data_prep(train)
                self.X, self.y = model_train.drop(columns =self.target_col), model_train[self.target_col]
                self.model_cub = model.fit(self.X, self.y)
                yhat = self.forecast(n_ahead =len(y_test), x_test=x_test)
                if eval_metric.__name__== 'mean_squared_error':
                    accuracy = eval_metric(y_test, yhat, squared=False)
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
        return space_eval(param_space, best_hyperparams)
    
class lightGBM_bidirect_forecaster:
    def __init__(self, model, target_col, n_lag = None, difference_1 = None, difference_2 = None, lag_transform = None, cat_variables = None):
        if (n_lag == None) and (lag_transform == None):
            raise ValueError('Expected either n_lag or lag_transform args')
        self.model = model
        self.target_col = target_col
        self.cat_var = cat_variables
        self.n_lag = n_lag
        self.difference1 = difference_1
        self.difference2 = difference_2
        self.lag_transform = lag_transform
        
    def data_prep(self, df):
        dfc = df.copy()
        # self.raw_df = df.copy()
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
        
        self.X1, self.y1 = model_df.drop(columns =self.target_col), model_df[self.target_col[0]]
        self.model_lgb1 = model_lgb1.fit(self.X1, self.y1, categorical_feature=self.cat_var)

        self.X2, self.y2 = model_df.drop(columns =self.target_col), model_df[self.target_col[1]]
        self.model_lgb2 = model_lgb2.fit(self.X2, self.y2, categorical_feature=self.cat_var)
    
    def forecast(self, n_ahead, x_test = None):
        tar1_lags = self.y1.tolist()
        tar2_lags = self.y2.tolist()
        tar1_predictions = []
        tar2_predictions = []
        
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

            inp1 = x_var + inp_lag1+inp_lag2+transform_lag
            df_inp1 = pd.DataFrame(inp1).T
            df_inp1.columns = self.X1.columns

            inp2 = x_var + inp_lag1+inp_lag2+transform_lag
            df_inp2 = pd.DataFrame(inp2).T
            df_inp2.columns = self.X2.columns
            
            for i,j in zip(df_inp1.columns, df_inp2.columns):
                if self.cat_var is not None:
                    if i in self.cat_var:
                        df_inp1[i] = df_inp1[i].astype('category')
                        df_inp2[i] = df_inp2[i].astype('category')
                    else:
                        df_inp1[i] = df_inp1[i].astype('float64')
                        df_inp2[j] = df_inp2[j].astype('float64')
                else:
                    df_inp1[i] = df_inp1[i].astype('float64')
                    df_inp2[j] = df_inp2[j].astype('float64')
            pred1 = self.model_lgb1.predict(df_inp1)[0]
            tar1_predictions.append(pred1)
            tar1_lags.append(pred1)
            
            pred2 = self.model_lgb2.predict(df_inp2)[0]
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
            
        return forecast1, forecast2
    
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
                
                self.X1, self.y1 = model_train.drop(columns =self.target_col), model_train[self.target_col[0]]
                self.X2, self.y2 = model_train.drop(columns =self.target_col), model_train[self.target_col[1]]
                self.model_lgb1 = model1.fit(self.X1, self.y1, categorical_feature=self.cat_var)
                self.model_lgb2 = model2.fit(self.X2, self.y2, categorical_feature=self.cat_var)
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