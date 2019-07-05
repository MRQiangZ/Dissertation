#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 18:58:28 2019

@author: zhangqiang
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


data = pd.read_csv('./data/all.csv')

#split input and output
X = data.drop('pene',axis = 1)

#Code below is not right，because y will not be DataFrame and thus there is no
#column name in y
#y = data['pene']
y = pd.DataFrame(data['pene'],columns=['pene']).values.T[0]
#print(y)
"""
plt.hist(y,bins = np.arange(-40,40,0.1))
plt.title('The distribution of penetration depth')
plt.ylabel('Frequency')
plt.xlabel('Penetration depth')
plt.xlim(-40,40)


pdf = PdfPages('distribution_pene.pdf')
pdf.savefig()
plt.close()
pdf.close()
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



pd.DataFrame(X_train).to_csv ("./data/all_X_train.csv" , encoding = "utf-8" , index = False)
pd.DataFrame(X_test).to_csv ("./data/all_X_test.csv" , encoding = "utf-8" , index = False)
pd.DataFrame(y_train).to_csv ("./data/all_y_train.csv" , encoding = "utf-8" , index = False)
pd.DataFrame(y_test).to_csv ("./data/all_y_test.csv" , encoding = "utf-8" , index = False)
print('done')
