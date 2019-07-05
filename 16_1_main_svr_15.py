#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:54:33 2019

@author: zhangqiang
"""
import numpy as np
import pandas as pd
import datetime
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import random
from sklearn.externals import joblib

def RMSE(array_a,array_b):
    '''
    input: two np.array
    output: RMSE
    '''
    rmse = np.sqrt(np.mean((array_a-array_b)**2))
    return rmse

#X_train_16 = pd.read_csv('./data/16_1_X_train_normalized.csv')
#X_test_16 = pd.read_csv('./data/16_1_X_test_normalized.csv')

X_train_15 = pd.read_csv('./data/16_1_X_train_normalized_pca_15.csv')
X_test_15 = pd.read_csv('./data/16_1_X_test_normalized_pca_15.csv')

#X_train_12 = pd.read_csv('./data/16_1_X_train_normalized_pca.csv')
#X_test_12 = pd.read_csv('./data/16_1_X_test_normalized_pca.csv')

y_train = pd.read_csv('./data/16_1_y_train.csv')
y_test = pd.read_csv('./data/16_1_y_test.csv')

print("Last time for svr_15(epsilon = 0.8)")
#index_list = list(range(1,3000,1))+list(range(100000,103000,1))+list(range(50000,54000,1))
#num_row = X_train_15.shape[0]
#index_list = random.sample(range(0,num_row,1),120000)
#SVR
#16
#svr = GridSearchCV(SVR(kernel = 'rbf'), cv=3 , scoring = 'neg_mean_squared_error', n_jobs = 50 , param_grid={"C": [1e0, 1e1, 1e2, 1e3],
#                               "gamma": np.logspace(-2, 2, 5)})

#svr = GridSearchCV(SVR(kernel = 'rbf',gamma = 'auto'), cv=3 , scoring = 'neg_mean_squared_error', n_jobs = 50 , param_grid={"C": [1e0, 1e1, 1e2, 1e3]})
#best_C = 100,RMSE 5.720324920585942
#svr = GridSearchCV(SVR(kernel = 'rbf',C=100), cv=3 , scoring = 'neg_mean_squared_error', n_jobs = 50 , param_grid={"gamma": np.logspace(-2, 2, 5)})
#{'gamma': 1.0}
#RMSE:5.437845691094133


#x = X_train_16.iloc[index_list].values
y = y_train.values.T[0]
"""
print('Model selection begins for 16D input')
time_start = datetime.datetime.now()
print(time_start.strftime('%Y-%m-%d %H:%M:%S'))
svr.fit(x,y)
time_end = datetime.datetime.now()
print('*'*10+'Finished'+'*'*10)
print(time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('*'*50)
print('Best paras for 16D input are：')
print(svr.best_params_)
y_pred_svr_16 = svr.predict(X_test_16.values)
rmse_svr_16 = RMSE(y_pred_svr_16,y_test.values.T)
print('RMSE:')
print(rmse_svr_16)
print('*'*50)
"""
"""
svr = SVR(kernel = 'rbf',C=100,gamma = 1.0)
time_start = datetime.datetime.now()
print(time_start.strftime('%Y-%m-%d %H:%M:%S'))
svr.fit(x,y)
time_end = datetime.datetime.now()
print('*'*10+'Finished'+'*'*10)
print(time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('*'*50)
y_pred_svr_16 = svr.predict(X_test_16.values)
rmse_svr_16 = RMSE(y_pred_svr_16,y_test.values.T)
print('RMSE:')
print(rmse_svr_16)
print('*'*50)
"""

#15
#svr_15 = GridSearchCV(SVR(kernel = 'rbf'), cv=3 , scoring = 'neg_mean_squared_error', n_jobs = 50 , param_grid={"C": [1e0, 1e1, 1e2, 1e3],
#                               "gamma": np.logspace(-2, 2, 5)})
#svr_15 = GridSearchCV(SVR(kernel = 'rbf',gamma = 'auto'), cv=3 , scoring = 'neg_mean_squared_error', n_jobs = 50 , param_grid={"C": [1e0, 1e1, 1e2, 1e3]})
#best_C = 100 RMSE 5.713726436981453
#svr_15 = GridSearchCV(SVR(kernel = 'rbf',C=100), cv=3 , scoring = 'neg_mean_squared_error', n_jobs = 50 , param_grid={"gamma": np.logspace(-2, 2, 5)})
#{'gamma': 1.0}
#RMSE:5.437845251642316
x_15 = X_train_15.values
"""
print('Model selection begins for 15D input')
time_start = datetime.datetime.now()
print(time_start.strftime('%Y-%m-%d %H:%M:%S'))
svr_15.fit(x_15,y)
time_end = datetime.datetime.now()
print('*'*10+'Finished'+'*'*10)
print(time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('*'*50)
print('Best paras for 15D input are：')
print(svr_15.best_params_)
y_pred_svr_15 = svr_15.predict(X_test_15.values)
rmse_svr_15 = RMSE(y_pred_svr_15,y_test.values.T)
print('RMSE:')
print(rmse_svr_15)
print('*'*50)
"""
svr_15 = SVR(kernel = 'rbf',C=100,gamma = 1.0,epsilon=0.8)
time_start = datetime.datetime.now()
print(time_start.strftime('%Y-%m-%d %H:%M:%S'))
svr_15.fit(x_15,y)
joblib.dump(svr_15,'./model/svr_final_15.m')
time_end = datetime.datetime.now()
print('*'*10+'Finished'+'*'*10)
print(time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('*'*50)
y_pred_svr_15 = svr_15.predict(X_test_15.values)
rmse_svr_15 = RMSE(y_pred_svr_15,y_test.values.T)
print('RMSE:')
print(rmse_svr_15)
print('*'*50)
#12
#svr_12 = GridSearchCV(SVR(kernel = 'rbf'), cv=3 , scoring = 'neg_mean_squared_error', n_jobs = 50 , param_grid={"C": [1e0, 1e1, 1e2, 1e3],
#                               "gamma": np.logspace(-2, 2, 5)})
#svr_12 = GridSearchCV(SVR(kernel = 'rbf',gamma = 'auto'), cv=3 , scoring = 'neg_mean_squared_error', n_jobs = 50 , param_grid={"C": [1e0, 1e1, 1e2, 1e3]})
#best_C = 10 RMSE 6.020602044298314
#svr_12 = GridSearchCV(SVR(kernel = 'rbf',C=10), cv=3 , scoring = 'neg_mean_squared_error', n_jobs = 50 , param_grid={"gamma": np.logspace(-2, 2, 5)})
#{'gamma': 1.0}
#RMSE:5.636200321050568
#x_12 = X_train_12.iloc[index_list].values
"""
print('Model selection begins for 12D input')
time_start = datetime.datetime.now()
print(time_start.strftime('%Y-%m-%d %H:%M:%S'))
svr_12.fit(x_12,y)
time_end = datetime.datetime.now()
print('*'*10+'Finished'+'*'*10)
print(time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('*'*50)
print('Best paras for 12D input are：')
print(svr_12.best_params_)
y_pred_svr_12 = svr_12.predict(X_test_12.values)
rmse_svr_12 = RMSE(y_pred_svr_12,y_test.values.T)
print('RMSE:')
print(rmse_svr_12)
print('*'*50)
"""
"""
svr_12 = SVR(kernel = 'rbf',C=100,gamma = 1.0)
time_start = datetime.datetime.now()
print(time_start.strftime('%Y-%m-%d %H:%M:%S'))
svr_12.fit(x_12,y)
time_end = datetime.datetime.now()
print('*'*10+'Finished'+'*'*10)
print(time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('*'*50)
y_pred_svr_12 = svr_12.predict(X_test_12.values)
rmse_svr_12 = RMSE(y_pred_svr_12,y_test.values.T)
print('RMSE:')
print(rmse_svr_12)
print('*'*50)
"""

    
    
    
    
    

