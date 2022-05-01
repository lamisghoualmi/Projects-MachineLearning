# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:51:11 2020

@author: Benk
"""
import numpy as np
import pandas  as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, precision_score
from sklearn import svm
from sklearn.feature_selection import mutual_info_classif



#from sklearn.datasets import load_iris
#iris = load_iris()
#df= pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
#target=iris.target

from sklearn.datasets import load_digits
digits = load_digits()
df=digits.data
df = pd.DataFrame(df)
target=digits.target

#from sklearn.datasets import fetch_olivetti_faces
#f = fetch_olivetti_faces()
#df=f.data
#df=pd.DataFrame(df)
#target=f.target



bigy_test=[]
bigy_pred=[]
randEst=[42,10,20,32, 0, 62, 82, 100, 150, 200]
bigy_test=[]
bigy_pred=[]
for i in range (0,1):
    X_train,X_test,y_train,y_test = train_test_split(df,target,test_size=0.5,random_state=randEst[i])
    clf = svm.SVC(kernel='linear') 
    clf.fit(X_train, y_train)
    y_pred= clf.predict(X_test)
    bigy_test.append(y_test)
    bigy_pred.append(y_pred)
    
bigy_test = [j for sub in bigy_test for j in sub]
bigy_pred = [jj for subj in bigy_pred for jj in subj]
Accuracy=accuracy_score(bigy_test, bigy_pred)

print("----------Accuracy Full feature:",accuracy_score(bigy_test, bigy_pred) *100)
print('----------feature size:', df.shape)







    