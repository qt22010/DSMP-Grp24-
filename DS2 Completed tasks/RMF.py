# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:30:02 2023

@author: white
"""

import pandas as pd
df=pd.read_csv("C://users/white/Desktop/DSMP/features_second.csv")
print(df.head())
print(df.columns)

df_between=df[(df["Third Party Name"].notnull())]
print(df_between)
df_between["Date"]= pd.to_datetime(df_between["Date"])
#filter monetary by -ve then make it +Ve

df_between=df_between[(df_between["Amount"]<0)]
print(df_between["Amount"])

df_between["Amount"]=df_between["Amount"].apply(lambda x: x*-1)
print(df_between["Amount"])

df_customers = df_between.groupby('Account Number').agg({'Amount':['min','max','mean','sum','count'], 'Date': pd.Series.max})
print(df_customers)
df_customers.iloc[:, 5] = pd.to_datetime(df_customers.iloc[:, 5])
recent_date = df_between.Date.max()
df_customers['Recency'] = df_customers.iloc[:, 5].apply(lambda x: (recent_date - x).days)
print(df_customers)

df_customers = df_customers.drop(df_customers.columns[0:3], axis =1)
df_customers = df_customers.drop(df_customers.columns[2], axis =1)
print(df_customers)

df_customers.to_csv(r"C:\\users\white\Downloads\features_seconds.csv")








