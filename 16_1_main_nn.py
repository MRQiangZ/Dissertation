#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 18:55:15 2019

@author: zhangqiang
"""

import numpy as np
import pandas as pd
import datetime
#from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

def RMSE(array_a,array_b):
    '''
    input: two np.array
    output: RMSE
    '''
    rmse = np.sqrt(np.mean((array_a-array_b)**2))
    return rmse

X_train_16 = pd.read_csv('./data/16_1_X_train_normalized.csv')
X_test_16 = pd.read_csv('./data/16_1_X_test_normalized.csv')

#X_train_15 = pd.read_csv('./data/16_1_X_train_normalized_pca_15.csv')
#X_test_15 = pd.read_csv('./data/16_1_X_test_normalized_pca_15.csv')

#X_train_12 = pd.read_csv('./data/16_1_X_train_normalized_pca.csv')
#X_test_12 = pd.read_csv('./data/16_1_X_test_normalized_pca.csv')

y_train = pd.read_csv('./data/16_1_y_train.csv')
y_test = pd.read_csv('./data/16_1_y_test.csv')
"""
mlp = MLPRegressor(hidden_layer_sizes=(100,100,100),
                   activation='tanh',
                   solver='adam',
                   alpha=0.1,
                   batch_size='auto',
                   learning_rate='constant',
                   learning_rate_init=0.001,
                   power_t=0.5,
                   max_iter=200,
                   shuffle=True,
                   random_state=None,
                   tol=0.0001,
                   verbose=False,
                   warm_start=False,
                   momentum=0.9,
                   nesterovs_momentum=True,
                   early_stopping=True,
                   validation_fraction=0.3,
                   beta_1=0.9,
                   beta_2=0.999,
                   epsilon=1e-8)
"""
mlp = MLPRegressor(hidden_layer_sizes=(100,100,100),
                   activation='relu',
                   solver='adam',
                   alpha=0.1,
                   batch_size='auto',
                   learning_rate='constant',
                   learning_rate_init=0.001,
                   power_t=0.5,
                   max_iter=200,
                   shuffle=True,
                   random_state=None,
                   tol=0.0001,
                   verbose=False,
                   warm_start=False,
                   momentum=0.9,
                   nesterovs_momentum=True,
                   early_stopping=True,
                   validation_fraction=0.3,
                   beta_1=0.9,
                   beta_2=0.999,
                   epsilon=1e-8)
x = X_train_16.values
y = y_train.values.T[0]
print('MLP')
print('Model selection begins for 16D input')
time_start = datetime.datetime.now()
print(time_start.strftime('%Y-%m-%d %H:%M:%S'))
mlp.fit(x,y)
time_end = datetime.datetime.now()
print('*'*10+'Finished'+'*'*10)
print(time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('*'*50)
#print('Best paras for 16D input are：')
#print(svr.best_params_)
y_pred_mlp_16 = mlp.predict(X_test_16.values)
rmse_mlp_16 = RMSE(y_pred_mlp_16,y_test.values.T)
print('RMSE:')
print(rmse_mlp_16)
#3.26577542345609
print('*'*50)