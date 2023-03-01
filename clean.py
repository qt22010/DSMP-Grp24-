# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 20:59:52 2023

@author: Ronak
"""

import pandas as pd
df = pd. read_csv(r"C:\Users\Ronak\Downloads\fake_transactional_data.csv")

df1 = df[df['to_randomly_generated_account'].str.isnumeric() == False]

df1=df.rename(columns={'from_totally_fake_account':'from_account', 'monopoly_money_amount':'money_amount', 'to_randomly_generated_account':'to_account', 'not_happened_yet_date':'date'})

df1=df1.dropna()

df1.to_csv('cleaned_fake_transactional_data.csv', index=False)
