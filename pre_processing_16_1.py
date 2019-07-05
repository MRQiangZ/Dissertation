#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:36:30 2019

@author: zhangqiang
"""

import h5py
import pandas as pd


'''
#Find all the names of all datasets
data_h5py = h5py.File('/Users/zhangqiang/Desktop/project1/NearestPoints/PandasFile.h5','r')
keys = list(data_h5py.keys())
print(keys)
data_h5py.close()
'''

#concat all the datasets
keys = ['jakobshavn11', 'jakobshavn12', 'jakobshavn13', 'jakobshavn14', 'jakobshavn15', 
        'jakobshavn16', 'southeast11', 'southeast12', 'southeast13', 'southeast14', 
        'southeast15', 'southeast16', 'storstrommen11', 'storstrommen12', 'storstrommen13', 
        'storstrommen14', 'storstrommen15', 'storstrommen16']
i = 0
data = []
for each_key in keys:
    if i==0:
        data = pd.read_hdf('./data/allPoints.h5',key = each_key)
        i = i+1
    else:
        new_data = pd.read_hdf('./data/allPoints.h5',key = each_key)
        data = pd.concat([data,new_data],axis = 0)
        
#print(data)


#create pene feature and delete unneeded features
data['pene'] = data['Elev_Oib'] - data['Elev_Swath']
data.drop(['Elev_Swath','Elev_Oib'],axis = 1 , inplace = True)
data.drop(['DemDiff_Swath','DemDiffMad_Swath','DemDiff_SwathOverPoca','MeanDiffSpread_Swath',
           'Wf_Number_Swath','StartTime_Swath','Y_Swath','X_Swath','Lon_Swath','Lat_Swath',
           ],axis = 1 , inplace = True)
    
data.to_csv ("./data/all.csv" , encoding = "utf-8" , index = False)

