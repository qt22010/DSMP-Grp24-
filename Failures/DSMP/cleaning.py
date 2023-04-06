# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:13:45 2023

@author: Igris
"""
#importing packages and shit
import pandas as pd
import numpy as np
#loading dataset 
df=pd.read_csv("fake_transactional_data.csv")
#renaming the rows 
df=df.rename(columns={'from_totally_fake_account':'from_account', 'monopoly_money_amount':'money_amount', 'to_randomly_generated_account':'to_account', 'not_happened_yet_date':'date'})
#sanity check to check dimensions of the data i.e the rows
print(df.shape)
print(df.head())
#checking for missing values in the money amount, important for my task ig idfk
dfmoney=df["money_amount"]
#checking columns for empty values 
dfmoneycheck=dfmoney.isnull()
#dropping empty money values
df=df.dropna(axis=0, subset="money_amount")
print(df.shape)
print(df.describe())
print(df["money_amount"].isnull().sum())
#
#more data cleaning FML
#
#working out total expenditure and adding it to a new column
totalexpend=df.groupby("from_account")[["money_amount"]].sum()
print(totalexpend)
#merging both dataframes based on the from account column
testdf=df.merge(totalexpend, how="left",on="from_account",  )
#renaming column from default
testdf=testdf.rename(columns={"money_amount_y":"estimated_expenditure"})
df=testdf
print(df)
#filter out high sums wages?
#
#working out income based on expenditure and adding it to a new column
#need to calculate the multiplier
income_multiplier=1.5
incomedf=df["estimated_expenditure"]*income_multiplier
df["estimated_income"]=incomedf
print(df)
#graphs and shit 
#
#
#customer segmentation
#
#
#fraud detection code
#
#remove outliers based on fraud or the accounts as a whole
#investments/saving/bonds and shit
#
#
#mortage/loan approval/cars w.e 
#
#
