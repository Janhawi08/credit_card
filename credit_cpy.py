# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 00:37:09 2020

@author: Janhawi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('creditcard.csv')

#counting the no. of 1's and 0's
count = dataset["Class"].value_counts()

#creating independent and dependent variables
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1:]

#splitting the data into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state =100)

from sklearn.metrics import roc_auc_score
roc_values = []
for feature in X_train.columns:
    clf = LogisticRegression(random_state = 0)
    clf.fit(X_train[feature].fillna(0).to_frame(), y_train)
    y_scored = clf.predict_proba(X_test[feature].fillna(0).to_frame())
    roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))

roc_values = pd.Series(roc_values)
roc_values.index = X_train.columns
roc_values.sort_values(ascending=False)
roc_values.sort_values(ascending=False).plot.bar(figsize=(20, 8))
plt.title('Univariate ROC-AUC')

X_train = X_train.drop(['V22','Amount','V15','V26','V25'],axis='columns')
X_test = X_test.drop(['V22','Amount','V15','V26','V25'],axis='columns')

#feature scaling
from sklearn.preprocessing import RobustScaler
scaler= RobustScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train = pd.DataFrame(X_train_scaled,columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled,columns=X_test.columns)

from sklearn.preprocessing import PolynomialFeatures
poly= PolynomialFeatures(degree=2)
X_train =poly.fit_transform(X_train)
X_test = poly.transform(X_test)

#training model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0,C=0.01,penalty='l2',n_jobs = 8,solver='saga')
classifier.fit(X_train,y_train)

#training model
y_pred = classifier.predict(X_test).reshape(-1,1)
y_pred1 = classifier.predict(X_train)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred, labels=None, sample_weight=None, normalize=None))

#predicting values
from sklearn.metrics import jaccard_similarity_score
print("Accuracy of train set : " + str(jaccard_similarity_score(y_train,y_pred1)))
print('Accuracy of test set : '+str(jaccard_similarity_score(y_test,y_pred)))
