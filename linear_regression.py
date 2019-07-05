#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 22:58:38 2019

@author: zhangqiang
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def RMSE(array_a,array_b):
    '''
    input: two np.array
    output: RMSE
    '''
    rmse = np.sqrt(np.mean((array_a-array_b)**2))
    return rmse

X_train_16 = pd.read_csv('./data/16_1_X_train_normalized.csv').values
X_test_16 = pd.read_csv('./data/16_1_X_test_normalized.csv')

X_train_15 = pd.read_csv('./data/16_1_X_train_normalized_pca_15.csv').values
X_test_15 = pd.read_csv('./data/16_1_X_test_normalized_pca_15.csv')

X_train_12 = pd.read_csv('./data/16_1_X_train_normalized_pca.csv').values
X_test_12 = pd.read_csv('./data/16_1_X_test_normalized_pca.csv')

y_train = pd.read_csv('./data/16_1_y_train.csv').values.T[0]
y_test = pd.read_csv('./data/16_1_y_test.csv')

print('Data Loaded')
clf_16 = LinearRegression()

clf_16.fit(X_train_16,y_train)
print('*'*10+'Finished'+'*'*10)
print('*'*50)
y_pred = clf_16.predict(X_test_16.values)
rmse = RMSE(y_pred,y_test.values.T)
print('RMSE for 16D:')
print(rmse)
print('*'*50)

clf_15 = LinearRegression()

clf_15.fit(X_train_15,y_train)
print('*'*10+'Finished'+'*'*10)
print('*'*50)
y_pred = clf_15.predict(X_test_15.values)
rmse = RMSE(y_pred,y_test.values.T)
print('RMSE for 15D:')
print(rmse)
print('*'*50)

clf_12 = LinearRegression()

clf_12.fit(X_train_12,y_train)
print('*'*10+'Finished'+'*'*10)
print('*'*50)
y_pred = clf_12.predict(X_test_12.values)
rmse = RMSE(y_pred,y_test.values.T)
print('RMSE for 12D:')
print(rmse)
print('*'*50)

