# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 15:42:19 2023

@author: white
"""

import pandas as pd
df=pd.read_csv("C:\\users\white\Downloads\cleaned_second.csv", index_col=False)
print(df.head())

list1=df["Account Number"].unique()
list1.sort()
print(len(list1))

df_midnight=df[(df.Time=="00:00:00") & (df.Amount>=0) & (df["Third Party Name"].notnull())]
print(df_midnight)

for i in list1:
    dfloop=df_midnight[(df["Account Number"]==i)]
    print(dfloop)
    mean=dfloop["Amount"].mean()
    mean="%.2f" % mean
    df.loc[df.index[(df["Account Number"]==i)], "Income"] = mean
print(df)

df.loc[df.index[(df["Income"]!="nan")], "Status"] = "Employed"
df.loc[df.index[(df["Income"]=="nan")], "Status"] = "Unemployed"

print(df)

from datetime import datetime


list2=df["Date"].unique()
for i in list2:
    Date=datetime.strptime(i,"%m/%d/%Y")
    Date=Date.weekday()
    if Date <5:
        df.loc[df.index[(df["Date"]==i)], "Type_Day"] = "weekday"
    else:
        df.loc[df.index[(df["Date"]==i)], "Type_Day"] = "weekend"
    if Date==0:
        df.loc[df.index[(df["Date"]==i)], "Day"] = "Monday"
    elif Date==1:
        df.loc[df.index[(df["Date"]==i)], "Day"] = "Tuesday"
    elif Date==2:
        df.loc[df.index[(df["Date"]==i)], "Day"] = "Wednesday"
    elif Date==3:
        df.loc[df.index[(df["Date"]==i)], "Day"] = "Thursday"
    elif Date==4:
        df.loc[df.index[(df["Date"]==i)], "Day"] = "Friday"
    elif Date==5:
        df.loc[df.index[(df["Date"]==i)], "Day"] = "Saturday"
    else:
        df.loc[df.index[(df["Date"]==i)], "Day"] = "Sunday"
        
print(df)
df.iloc[22714]


df.to_csv(r"C:\\users\white\Downloads\features_second.csv", index=False)