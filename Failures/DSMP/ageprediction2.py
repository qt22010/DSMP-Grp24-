# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 13:06:20 2023

@author: white
"""

#importing packages
import pandas as pd
import numpy as np
#reading in cleaned csv
df=pd.read_csv(r"C:/Users/white/Downloads/unique_accounts.csv")
#dropping seconmd column not needed
df.drop(df.iloc[:,:0] , axis=1, inplace=True)
#sanity check
print(df)
#reading csv
df3=pd.read_csv(r"C:/Users/white/Downloads/clean_dataset.csv")
#filtering out the between account transactions
df3=df3.loc[(df3["Categorised"] != "To account")]
#dropping NA values 
df3=df3.dropna(subset=["from_account"])
#sanity check
print(df3)
#sorting dataframe by account and category so its easier for me to read the output below
df3 = df3.sort_values(['from_account', 'Categorised'],
              ascending = [True, True])

#obtaining frequency for each category
ratio=df3[["from_account", "Categorised"]].value_counts(sort=False)
count=df3.groupby("from_account")[["Categorised"]].count()
#merging it back onto the main dataframe
testdf=df3.merge(count, how="left",on="from_account",  )
#sanity checks on both methods
print(ratio)
print(count)
#renaming columns for easier read
testdf = testdf.rename(columns={'Categorised_x': 'Categorised', 'Categorised_y': 'Frequency'})
print(testdf)
#merging on the main dataframe
testdf1=testdf.merge(ratio.rename("Categorised_frequency"), how="left", on=["from_account","Categorised"],  )
print(testdf1)
print(testdf1.columns)
#dropping columns for an easier read
finaldf=testdf1.drop(["Unnamed: 0", "money_amount", "date", "transaction_ID"],axis=1)
print(finaldf)

#normalising the frequency
finaldf["ratio"]=finaldf["Categorised_frequency"]/finaldf["Frequency"]
print(finaldf)
#dropping clutter
finaldf=finaldf.drop(["Categorised_frequency", "Frequency", "to_account"],axis=1)
print(finaldf)
#removing duplicate rows we only need frequency per category per account
finaldf=finaldf.drop_duplicates()
#resetting the index
finaldf=finaldf.reset_index()
#dropping previous index column
finaldf=finaldf.drop(["index"],axis=1)
#output check
print(finaldf)
#exporting csv
finaldf.to_csv("final_ratio.csv",index=False)