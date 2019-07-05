#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 23:52:55 2019

@author: zhangqiang
"""

import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor
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

X_train_16 = pd.read_csv('./data/16_1_X_train_normalized.csv')
X_test_16 = pd.read_csv('./data/16_1_X_test_normalized.csv')

#X_train_15 = pd.read_csv('./data/16_1_X_train_normalized_pca_15.csv')
#X_test_15 = pd.read_csv('./data/16_1_X_test_normalized_pca_15.csv')

#X_train_12 = pd.read_csv('./data/16_1_X_train_normalized_pca.csv')
#X_test_12 = pd.read_csv('./data/16_1_X_test_normalized_pca.csv')

y_train = pd.read_csv('./data/16_1_y_train.csv')
y_test = pd.read_csv('./data/16_1_y_test.csv')

print('Data Loaded')

num_row = X_train_16.shape[0]
random.seed(0)
index_list = random.sample(range(0,num_row,1),num_row)

x = X_train_16.iloc[index_list].values
y = y_train.iloc[index_list].values.T[0]

#print(y)

clf = RandomForestRegressor(
            n_estimators=2,            
            criterion='mse',            
            max_depth=None,            
            min_samples_split=2,        
            min_samples_leaf=1,      
            max_features='auto',        
            max_leaf_nodes=None,     
            bootstrap=True,             
            min_weight_fraction_leaf=0,
            n_jobs=50)

"""
# 1 首先确定迭代次数
param_test1 = {
        'n_estimators': [i for i in range(100, 201, 20)]
    }
"""

clf.set_params(n_estimators=160)

# 2.1 对max_depth 和 min_samples_split 和 min_samples_leaf 进行粗调

"""
param_test2_1 = {
        'max_depth': [45,50,55,60],
        'min_samples_split' : np.arange(2,5,2),
        'min_samples_leaf' : np.arange(1,5,2)
}
"""


clf.set_params(min_samples_split=2,min_samples_leaf= 1)

"""
max_d = 45
param_test2_2 = {
        'max_depth': [max_d-2, max_d, max_d+2]
    }
"""
clf.set_params(max_depth = 47)
"""
param_test3_1 = {
        'max_features': [0.1,0.3,0.5, 0.7,0.9]
    }
"""

max_f = 0.5

param_test3_2 = {
        'max_features': [max_f-0.1, max_f, max_f+0.1]
    }

#clf.set_params(max_features = 0.6)


gsearch = GridSearchCV(clf, param_grid=param_test3_2, scoring='neg_mean_squared_error',n_jobs=1, cv=3)
time_start = datetime.datetime.now()
print(time_start.strftime('%Y-%m-%d %H:%M:%S'))
gsearch.fit(x,y)
time_end = datetime.datetime.now()
print('*'*10+'Finished'+'*'*10)
print(time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('*'*50)
print(gsearch.best_params_)
print('*'*50)
print(gsearch.cv_results_['params'])
print(gsearch.cv_results_['mean_test_score'])
joblib.dump(gsearch,'./model/param_test_rf_allData_3_2.m')
y_pred = gsearch.predict(X_test_16.values)
rmse = RMSE(y_pred,y_test.values.T)
print('RMSE:')
print(rmse)
print('*'*50)
"""
time_start = datetime.datetime.now()
print(time_start.strftime('%Y-%m-%d %H:%M:%S'))
clf.fit(x,y)
time_end = datetime.datetime.now()
print('*'*10+'Finished'+'*'*10)
print(time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('*'*50)
joblib.dump(clf,'./model/rf.m')
y_pred = clf.predict(X_test_16.values)
print(y_pred)
print(y_test.values.T)
rmse = RMSE(y_pred,y_test.values.T)
print('RMSE:')
print(rmse)
print('*'*50)
"""









