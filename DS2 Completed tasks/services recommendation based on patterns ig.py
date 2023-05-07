# -*- coding: utf-8 -*-
"""
Created on Sun May  7 10:42:25 2023

@author: white
"""
#import packages
import pandas as pd
#load dataset
df=pd.read_csv(r"C:\\users\white\Desktop\DSMP\features_second.csv")
#removing between transactions will not be required for this section
df=df[(df["Third Party Name"].notnull())&(df.Amount<0) ]
#list of columns
print(df.columns)
#dropping columns
df= df.drop(['Type_Day', "Unnamed: 0", 'Day'], axis =1)
#data check
print(df.head())
#listing all firm names
firm=df["Third Party Name"].unique()
print(firm)
#load dataset of industry
dfindust=pd.read_csv(r"C:\\users\white\Desktop\DSMP\Industry_firm_list.csv")
#merging dataframe with a list of industries
df = df.merge(dfindust, on="Third Party Name")
print(df.head())

"""creating df for frequency of transactions
 with that particular industry and amount spent"""
 
industry_counts = df.groupby(['Account Number', 'Industry']).size().reset_index(name='frequency')
industry_totals = df.groupby(['Account Number', 'Industry'])['Amount'].sum().reset_index()

#sanity check
print(industry_counts)
print(industry_totals)

#merging dataframes based on account number and industry
industry = industry_counts.merge(industry_totals, on=["Account Number", "Industry"])

#taking absolute values for later will make some analysis easier to understand 
industry["Amount"]=industry["Amount"].abs()
#sanity check
print(industry.head())
