#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 00:16:52 2019

@author: zhangqiang
"""


import h5py
import pandas as pd


#concat all the datasets
#！concat合并的结果是否是我想要的需要验证
keys = ['jakobshavn11', 'jakobshavn12', 'jakobshavn13', 'jakobshavn14', 'jakobshavn15','jakobshavn16', 
         'southeast11', 'southeast12', 'southeast13', 'southeast14', 
        'southeast15', 'southeast16']
i = 0
data = []
for each_key in keys:
    if i==0:
        data = pd.read_hdf('./data/Nearest.h5',key = each_key)
        i = i+1
    else:
        new_data = pd.read_hdf('./data/Nearest.h5',key = each_key)
        data = pd.concat([data,new_data],axis = 0)
        
keys_test = ['storstrommen11', 'storstrommen12', 'storstrommen13', 
        'storstrommen14', 'storstrommen15','storstrommen16']
j = 0
test_data = []
for each_key in keys_test:
    if j==0:
        test_data = pd.read_hdf('./data/Nearest.h5',key = each_key)
        j = j+1
    else:
        new_test_data = pd.read_hdf('./data/Nearest.h5',key = each_key)
        test_data = pd.concat([test_data,new_test_data],axis = 0)


#create pene feature and delete unneeded features
data['pene'] = data['Elev_Oib'] - data['Elev_Swath']
data.drop(['Elev_Swath','Elev_Oib'],axis = 1 , inplace = True)
data.drop(['DemDiff_Swath','DemDiffMad_Swath','DemDiff_SwathOverPoca','MeanDiffSpread_Swath',
           'Wf_Number_Swath','StartTime_Swath','Y_Swath','X_Swath','Lon_Swath','Lat_Swath',
           ],axis = 1 , inplace = True)
    
test_data['pene'] = test_data['Elev_Oib'] - test_data['Elev_Swath']
test_data.drop(['Elev_Swath','Elev_Oib'],axis = 1 , inplace = True)
test_data.drop(['DemDiff_Swath','DemDiffMad_Swath','DemDiff_SwathOverPoca','MeanDiffSpread_Swath',
           'Wf_Number_Swath','StartTime_Swath','Y_Swath','X_Swath','Lon_Swath','Lat_Swath',
           ],axis = 1 , inplace = True)

x_train = data.drop('pene',axis = 1)
y_train = pd.DataFrame(data['pene'],columns=['pene']).values.T[0]
x_test = test_data.drop('pene',axis = 1)
y_test = pd.DataFrame(test_data['pene'],columns=['pene']).values.T[0]

mean_col_list = list(x_train.mean())
std_col_list = list(x_train.std())

x_train_normlized = (x_train - mean_col_list)/std_col_list
x_test_normalized = (x_test - mean_col_list)/std_col_list

x_train_normlized.to_csv ("./data/space_x_train_normalised.csv" , encoding = "utf-8" , index = False)
x_test_normalized.to_csv ("./data/space_x_test_normalised.csv" , encoding = "utf-8" , index = False)
pd.DataFrame(y_train).to_csv ("./data/space_y_train.csv" , encoding = "utf-8" , index = False)
pd.DataFrame(y_test).to_csv ("./data/space_y_test.csv" , encoding = "utf-8" , index = False)

