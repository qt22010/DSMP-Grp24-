# -*- coding: utf-8 -*-
"""

@author: white
"""
#import packages
import pandas as pd
from datetime import datetime, date, time
#load dataset
df=pd.read_csv(r"C:\\users\white\Desktop\DSMP\features_second.csv")
#removing between transactions will not be required for this section
df=df[(df["Third Party Name"].notnull())&(df.Amount<0) ]
#convert date time
df['Time'] = pd.to_datetime((df['Time']))
#filter between 9am to 5pm
df.set_index('Time').between_time('09:00', '17:00').reset_index()
#weekday transactions only
df=df[(df.Type_Day=="weekday")]
#check
print(df)
#list of columns
print(df.columns)
#dropping columns
df= df.drop(["Unnamed: 0", 'Day'], axis =1)
#data check
print(df.head())
#getting the count of each transaction by company
timetransactioncount = df.groupby(['Account Number', 'Third Party Name']).size().reset_index(name='frequency')
time_totals = df.groupby(['Account Number', 'Third Party Name'])['Amount'].sum().reset_index()
timetransactioncount = timetransactioncount.merge(time_totals, on=["Account Number", "Third Party Name"])
timetransactioncount["Amount"]=timetransactioncount["Amount"].abs()
#sanity check
print(timetransactioncount)

#list of unique values we will drop firms that the employed would visit
print(timetransactioncount["Third Party Name"].unique())

#sash double check this

timetransactioncount=timetransactioncount.loc[~timetransactioncount["Third Party Name"].isin(['Coffee #1','Costa Coffee','Halifax', 
                                                                      'PUREGYM','Starbucks','SUNNY CARE NURSERY'])]

print(timetransactioncount)