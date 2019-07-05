#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:54:33 2019

@author: zhangqiang
"""
import numpy as np
import pandas as pd
import datetime
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
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

X_train_16 = pd.read_csv('./data/space_x_train_normalised.csv')
X_test_16 = pd.read_csv('./data/space_x_test_normalised.csv')

#X_train_15 = pd.read_csv('./data/16_1_X_train_normalized_pca_15.csv')
#X_test_15 = pd.read_csv('./data/16_1_X_test_normalized_pca_15.csv')

#X_train_12 = pd.read_csv('./data/16_1_X_train_normalized_pca.csv')
#X_test_12 = pd.read_csv('./data/16_1_X_test_normalized_pca.csv')

y_train = pd.read_csv('./data/space_y_train.csv')
y_test = pd.read_csv('./data/space_y_test.csv')

print('Data Loaded')
"""
num_row = X_train_16.shape[0]
random.seed(0)
index_list = random.sample(range(0,num_row,1),num_row)

x = X_train_16.iloc[index_list].values
y = y_train.iloc[index_list].values
"""
x = X_train_16.values
y = y_train.values
#xgboost
#print(x.std())

clf = XGBRegressor(
        objective='reg:linear',
        learning_rate=0.1, 
        gamma=0,  
        subsample=0.8, 
        colsample_bytree=0.8,  
        reg_alpha=1, 
        reg_lambda=1,  
        max_depth=10,  
        min_child_weight=1,  
        n_jobs=20
)

"""
dtrain = xgb.DMatrix(x, y)
xgb_params = clf.get_xgb_params()
cvresult = xgb.cv(xgb_params, dtrain, nfold=5, num_boost_round=2000,
                      early_stopping_rounds=50)
#clf_xgb = xgb.train(xgb_params, dtrain, num_boost_round=cvresult.shape[0])
#fscore = clf_xgb.get_fscore()
#print(cvresult.shape[0], fscore)
print(cvresult.shape[0])
"""

clf.set_params(n_estimators=2000)
"""
param_test1 = {
        'max_depth': [i for i in range(3, 12, 2)],
        'min_child_weight': [i for i in range(1, 10, 2)]
    }

max_d = 11
min_cw = 3
param_test2 = {
        'max_depth': [max_d-1, max_d, max_d+1],
        'min_child_weight': [min_cw-1, min_cw, min_cw+1]
    }
"""
clf.set_params(max_depth=11,min_child_weight=3)
"""
param_test3 = {
        'gamma': [i / 10.0 for i in range(0, 10, 2)]
    }

b_gamma = 0
param_test4 = {
        'gamma': [b_gamma, b_gamma+0.1]
    }
"""
clf.set_params(gamma = 0)
"""
param_test5 = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }

b_subsample = 0.9
b_colsample_bytree = 0.9

param_test6 = {
        'subsample': [b_subsample-0.05, b_subsample, b_subsample+0.05],
        'colsample_bytree': [b_colsample_bytree-0.05, b_colsample_bytree, b_colsample_bytree+0.05]
    }
"""

clf.set_params(b_subsample = 0.9,b_colsample_bytree = 0.95)
"""
param_test7 = {
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 2],
        'reg_lambda': [1e-5, 1e-2, 0.1, 1, 2]
    }

b_alp = 2
b_lam = 2

param_test8 = {
        'reg_alpha': [b_alp, 2*b_alp, 3*b_alp,4*b_alp,5*b_alp,6*b_alp],
        'reg_lambda': [b_lam, 2*b_lam, 3*b_lam,4*b_lam,5*b_lam,6*b_lam]
    }
"""
"""
param_test9 = {
        'reg_alpha': [6*b_alp,7*b_alp,8*b_alp,9*b_alp,10*b_alp],
        'reg_lambda': [6*b_lam,7*b_lam,8*b_lam,9*b_lam,10*b_lam]
    }
"""
clf.set_params(reg_alpha = 12,reg_lambda = 16)
clf.set_params(learning_rate=0.1)
"""
gsearch = GridSearchCV(clf, param_grid=param_test8, scoring='neg_mean_squared_error',n_jobs=1, cv=3)
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
joblib.dump(gsearch,'./model/param_test8.m')
y_pred = gsearch.predict(X_test_16.values)
rmse = RMSE(y_pred,y_test.values.T)
print('RMSE:')
print(rmse)
print('*'*50)
"""


"""
dtrain = xgb.DMatrix(x, y)
xgb_params = clf.get_xgb_params()
cvresult = xgb.cv(xgb_params, dtrain, nfold=5, num_boost_round=2000,
                      early_stopping_rounds=50)
clf_xgb = xgb.train(xgb_params, dtrain, num_boost_round=cvresult.shape[0])
fscore = clf_xgb.get_fscore()
print(cvresult.shape[0], fscore)
"""


"""
gsearch = GridSearchCV(clf, param_grid=param_test8, scoring='neg_mean_squared_error',n_jobs=1, cv=3)
time_start = datetime.datetime.now()
print(time_start.strftime('%Y-%m-%d %H:%M:%S'))
gsearch.fit(x,y)
time_end = datetime.datetime.now()
print('*'*10+'Finished'+'*'*10)
print(time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('*'*50)
print(gsearch.best_params_)
y_pred = gsearch.predict(X_test_16.values)
rmse = RMSE(y_pred,y_test.values.T)
print('RMSE:')
print(rmse)
print('*'*50)
"""

#here
#clf.set_params(n_estimators=2000)
time_start = datetime.datetime.now()
print(time_start.strftime('%Y-%m-%d %H:%M:%S'))
clf.fit(x,y)
time_end = datetime.datetime.now()
print('*'*10+'Finished'+'*'*10)
print(time_end.strftime('%Y-%m-%d %H:%M:%S'))
print('*'*50)
#joblib.dump(clf,'./model/xgboost_time.m')

y_pred = clf.predict(X_test_16.values)
rmse = RMSE(y_pred,y_test.values.T)
print('RMSE:')
print(rmse)
print('*'*50)


  
    
    
    
    
    
    
    
    
    
    
    
    

