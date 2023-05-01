# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 05:25:28 2023

@author: Ronak
"""

import pandas as pd 
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#from category_encoders.binary import BinaryEncoder
path= r"C:\Users\Ronak\Desktop"
df_fake = pd.read_csv(os.path.join(path,"cleaned_fake_transactional_data.csv"))
df_firm = pd.read_csv(os.path.join(path,"firms.csv"),encoding="latin")
print(df_firm.columns)

df_firm.rename(columns={"To account":"to_account"},inplace=True)
print(df_firm.columns)

df_master=df_fake.merge(df_firm,on="to_account",how="left")
df_master.dropna(inplace=True)
df_master.drop(["Unnamed: 0"],inplace=True,axis=1)

ct = ColumnTransformer([('encode',OneHotEncoder(dtype='int',drop='first'),[2,4])])


X = df_master.iloc[:,0:5]
y= df_master.iloc[:,5:7]
X=ct.fit_transform(X)



xtrain,xtest,ytrain,ytest = train_test_split(X,y,train_size=0.8)


rtree_classifier = RandomForestClassifier()
rtree_classifier.fit(xtrain,ytrain)
ypred=rtree_classifier.predict(xtest)
print(accuracy_score(ytest,ypred))
