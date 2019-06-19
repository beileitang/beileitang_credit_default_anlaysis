#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:14:38 2019

@author: beileitang
"""

import pandas as pd
import numpy as np           # for math calculation
import matplotlib            # for plotting graphs
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns           # for data visualization
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")

file="~/Desktop/loan data/new_loan_data.csv"
data=pd.read_csv(file)
data.describe().transpose()
#%%

data.isnull().sum()
data['dti'].fillna(data['dti'].median(), inplace=True)
data.isnull().sum()
data.drop(["loan_status"],axis=1, inplace= True)
#%%

obj_df = data.select_dtypes(include=['object']).copy()
obj_df.tail()
obj_df.isnull().sum()
#%%

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

label= LabelEncoder()

obj_df['term']=label.fit_transform(obj_df['term'])
print(obj_df['term'])
term=obj_df['term']

obj_df['purpose']=label.fit_transform(obj_df['purpose'])
print(obj_df['purpose'])
purpose=obj_df['purpose']

obj_df['grade']=label.fit_transform(obj_df['grade'])
print(obj_df['grade'])
grade=obj_df['grade']

#%%
# onehotencoding
onehotencoder = OneHotEncoder()

## convge to muerical 
term = onehotencoder.fit_transform(obj_df.term.values.reshape(-1,1)).toarray()
dfOneHot_term = pd.DataFrame(term, columns = ["loan_term"+str(int(i)) for i in range(term.shape[1])]) #term 0 is 36 and term 1 is 60

purpose = onehotencoder.fit_transform(obj_df.purpose.values.reshape(-1,1)).toarray()
dfOneHot_purpose = pd.DataFrame(purpose, columns = ["loam_purpose"+str(int(i)) for i in range(purpose.shape[1])]) #term 0 is 36 and term 1 is 60

#int_rate = onehotencoder.fit_transform(obj_df.int_rate.values.reshape(-1,1)).toarray()
#dfOneHot_int = pd.DataFrame(int_rate, columns = ["loan_int_rate"+str(int(i)) for i in range(int_rate.shape[1])])

grade = onehotencoder.fit_transform(obj_df.grade.values.reshape(-1,1)).toarray()
dfOneHot_grade = pd.DataFrame(grade, columns = ["loan_grade"+str(int(i)) for i in range(grade.shape[1])])


data=pd.concat([data, dfOneHot_grade], axis=1)
#data=pd.concat([data, dfOneHot_int], axis=1)
data=pd.concat([data, dfOneHot_purpose], axis=1)
data=pd.concat([data, dfOneHot_term], axis=1)
#%%
data.isnull().sum()
#%%
data.drop(['grade'],axis=1,inplace=True)
data.drop(['purpose'],axis=1,inplace=True)
#data.drop(['int_rate'],axis=1,inplace=True)
data.drop(["term"],axis=1, inplace= True)

data.drop(["loam_purpose11"],axis=1, inplace= True)
data.drop(["loan_term1"],axis=1, inplace= True)
data.drop(["loan_grade6"],axis=1, inplace= True)

#%%

from sklearn.model_selection import train_test_split
y = data.loan_default # define the target variable (dependent variable) as y
# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

#%%

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
#%%


# Standarize features
scaler = StandardScaler()
Xtrain_std = scaler.fit_transform(X_train)
Xtest_std = scaler.fit_transform(X_test)
# normalization 
X = preprocessing.normalize(X_train, norm='l2')
#%%
svc_rbf = SVC(kernel='rbf', class_weight='balanced', C=10.0, random_state=0)
model = svc_rbf.fit(Xtrain_std, y_train)
# Create support vector classifier
svc = SVC(kernel='linear', class_weight='balanced', C=10.0, random_state=0)
model = svc.fit(Xtrain_std, y_train)

logmodel = LogisticRegression()
logmodel.fit(Xtrain_std,y_train) # traning the model by logistic 
#%%

predictions = logmodel.predict(X_test) # input new data to predict the result 
print(classification_report(y_test,predictions)) # compare the predicted and y_test
print("accuracy_score")
print( accuracy_score(predictions, y_test) )
#%%
# test normalizated 

svm_prediction=svc.predict(X_test)
print(classification_report(y_test,svm_prediction))
print("accuracy_score")
print( accuracy_score(svm_prediction, y_test) )

#%%

svm_prediction_rbf=svc_rbf.predict(X_test)
print(classification_report(y_test,svm_prediction_rbf))
print("accuracy_score")
print( accuracy_score(svm_prediction_rbf, y_test) )

#%%
from sklearn.utils import resample
df_majority = data[data.loan_default==0]
df_minority = data[data.loan_default==1]
 
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=1056,     # to match minority class
                                 random_state=123) # reproducible results

# Combine minority class with downsampled majority class
df_downsampled  = pd.concat([df_majority_downsampled, df_minority])
 
# Display new class counts
df_downsampled.loan_default.value_counts()
# 1    49
# 0    49
# Name: balance, dtype: int64

#%%


y = df_downsampled.loan_default # define the target variable (dependent variable) as y
# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(df_downsampled, y, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
#%%

# Standarize features
scaler = StandardScaler()
Xtrain_std = scaler.fit_transform(X_train)
Xtest_std = scaler.fit_transform(X_test)
# normalization 
X = preprocessing.normalize(X_train, norm='l2')

svc_rbf = SVC(kernel='rbf', class_weight='balanced', C=10.0, random_state=0)
model = svc_rbf.fit(Xtrain_std, y_train)
# Create support vector classifier
svc = SVC(kernel='linear', class_weight='balanced', C=10.0, random_state=0)
model = svc.fit(Xtrain_std, y_train)

logmodel = LogisticRegression()
logmodel.fit(Xtrain_std,y_train) # traning the model by logistic 
#%%
predictions = logmodel.predict(X_test) # input new data to predict the result 
print(classification_report(y_test,predictions)) # compare the predicted and y_test
print("accuracy_score")
print( accuracy_score(predictions, y_test) )

#%%

svm_prediction=svc.predict(X_test)
print(classification_report(y_test,svm_prediction))
print("accuracy_score")
print( accuracy_score(svm_prediction, y_test) )

#%%

svm_prediction_rbf=svc_rbf.predict(X_test)
print(classification_report(y_test,svm_prediction_rbf))
print("accuracy_score")
print( accuracy_score(svm_prediction_rbf, y_test) )





