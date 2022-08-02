# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 23:21:45 2018

@author: nikunj
"""
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import csv
import random
from math import sqrt
from sklearn.decomposition import PCA

#reading Data
clus_train = pd.read_csv("C01.csv")
ch = pd.read_csv("input.csv")
model = KMeans(n_clusters = 5)
clus_train = clus_train.iloc[: , :-1]
model.fit(clus_train)
clussAssignment = model.predict(clus_train)
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x= plot_columns[:,0], y = plot_columns[:,1], c=model.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 5 Clusters')
plt.show()

#these list will have index of each element of respective cluster
c0=list()
c1 = list()
c2=list()
c3=list()
c4=list()
for i in range(len(clussAssignment)):
    if(clussAssignment[i] == 0 ):
        c0.append(i)
    elif(clussAssignment[i] == 1 ):
        c1.append(i)
    elif(clussAssignment[i] == 2 ):
        c2.append(i)
    elif(clussAssignment[i] == 3 ):
        c3.append(i)
    elif(clussAssignment[i] == 4 ):
        c4.append(i)
   
result =model.predict(ch)
pc = list() #this will have predicted cluster
if(result == 0):
    pc = c0
elif(result == 1):
    pc = c1
elif(result == 2):
    pc = c2
elif(result == 3):
    pc = c3
elif(result == 4):
    pc = c4
clus_train.iloc[pc].to_csv('final_cluster.csv')


"""
    clustring is done , now we are doinf regression
"""

from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
#reg=linear_model.LinearRegression()
from math import sqrt

clus = pd.read_csv("final_cluster.csv")
x=clus.iloc[:,:-1].values
y = clus.iloc[: , -1].values
x_train , x_test ,y_train , y_test = train_test_split(x , y , test_size=0.1 , random_state=0)

regressor = LinearRegression()

regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)


err= sqrt(mean_squared_error(y_test, y_pred))

ans = regressor.predict(ch)
print(ans)