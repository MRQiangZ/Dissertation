#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 21:07:51 2019

@author: zhangqiang
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages

X_train = pd.read_csv('./data/all_X_train.csv')
y_train = pd.read_csv('./data/all_y_train.csv')
X_test = pd.read_csv('./data/all_X_test.csv')
y_test = pd.read_csv('./data/all_y_test.csv')


#plt.hist(y_train.values)
mean_col_list = list(X_train.mean())
std_col_list = list(X_train.std())

#nomalization
#DataFrame-list and DataFrame/list are column-wise operations
X_train_normlized = (X_train - mean_col_list)/std_col_list
#print(X_train_normlized)
X_test_normalized = (X_test - mean_col_list)/std_col_list


#output normalized data
X_train_normlized.to_csv("./data/all_X_train_normalized.csv" , encoding = "utf-8" , index = False)
X_test_normalized.to_csv("./data/all_X_test_normalized.csv" , encoding = "utf-8" , index = False)

"""
#determine how many PC we want to keep
pca_all = PCA()
pca_all.fit(X_train_normlized)
print(pca_all.explained_variance_ratio_)
explained_variance_ratio_n = []
for i in range(len(pca_all.explained_variance_ratio_),0,-1):
    j = 0
    explained_variance_ratio_n.append(0)
    while j < i :
        explained_variance_ratio_n[-1] += pca_all.explained_variance_ratio_[j]
        j += 1
    #round ratio to 2 decimals
    explained_variance_ratio_n[-1] = round(explained_variance_ratio_n[-1],2)
print(explained_variance_ratio_n)
print(pca_all.explained_variance_)
#plot explained ratio against dimensionality
name_list = list(range(16,0,-1))
rects = plt.bar(range(len(explained_variance_ratio_n)), explained_variance_ratio_n, color='rgby')
index = list(range(0,16,1))
index = [float(c)+0.06 for c in index]
plt.ylim(ymax = 1.1, ymin = 0)
plt.xticks(index, name_list)
plt.ylabel("Ratio of Explained Variance")
plt.xlabel("Dimensionality after PCA")
for rect in rects:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')
plt.show()

pdf = PdfPages('PCA_variance_ratio.pdf')
#pdf.savefig()
plt.close()
pdf.close()
"""


#pca to 12D
pca_12 = PCA(n_components = 12)
pca_12.fit(X_train_normlized)
X_train_normlized_pca = pd.DataFrame(pca_12.transform(X_train_normlized))
X_test_normalized_pca = pd.DataFrame(pca_12.transform(X_test_normalized))

#output normalized data after pca
X_train_normlized_pca.to_csv('./data/16_1_X_train_normalized_pca.csv',encoding='utf-8',index=False)
X_test_normalized_pca.to_csv('./data/16_1_X_test_normalized_pca.csv',encoding='utf-8',index=False)



#pca to 15D
pca_15 = PCA(n_components = 15)
pca_15.fit(X_train_normlized)
X_train_normlized_pca_15 = pd.DataFrame(pca_15.transform(X_train_normlized))
X_test_normalized_pca_15 = pd.DataFrame(pca_15.transform(X_test_normalized))

#output normalized data after pca
X_train_normlized_pca_15.to_csv('./data/16_1_X_train_normalized_pca_15.csv',encoding='utf-8',index=False)
X_test_normalized_pca_15.to_csv('./data/16_1_X_test_normalized_pca_15.csv',encoding='utf-8',index=False)



 
