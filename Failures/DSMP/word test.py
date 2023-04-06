# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#importing packages that we might potentially use later
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from gensim.models import Word2Vec
SEED='0161'
#loading dataset 
df=pd.read_csv("/Users/qt22010/Downloads/wordembedding_df_customers_expenditure.csv") 
#sanity check
print(df)
#'picking' out or 'filtering' the 'shops' that we extracted beforehand 
text_df=df["to_account"]
#obtaining unique words from above this gives us a list
text_df=text_df.unique()
#to see how long my list is
print(len(text_df))
print(type(text_df[0]))
#word2vec preferred input is a list of list just a simple loop to create a list of list
#cleaning and splitting words 
text_df2=[]
for i in range(len(text_df)):    
    element=text_df[i].replace('_',' ')
    element=element.split()
    text_df2.append(element)
test=text_df2
#sanity check 
print(test)
#word2vec model
model = Word2Vec(test, min_count=1)
#testing
model.wv.most_similar("SHOP")
#potential dead end?
#kmeans clustering

"""length=6
df = df[df["from_totally_fake_account"].notna()]
print(df)
df["from_totally_fake_account"] = df["from_totally_fake_account"].astype('int')
df["from_totally_fake_account"] = df["from_totally_fake_account"].astype('str')
df=df[(df["from_totally_fake_account"].str.len()==length)]
print(df)b"""
