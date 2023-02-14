"""
Created on Wed Feb  8 18:35:58 2023

@author: Ronak
"""

import pandas as pd

df = pd.read_csv('C:\\Users\\Ronak\\Downloads\\fake_transactional_data.csv')

df.drop(columns='from_totally_fake_account', inplace=True)

df = df[df['to_randomly_generated_account'].str.isnumeric() == False]

df.to_csv('cleaned_fake_transactional_data.csv', index=False)
