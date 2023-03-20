import pandas as pd
import numpy as np
#reading in cleaned csv
df=pd.read_csv(r"C:/Users/white/Desktop/DSMP/unique_accounts.csv",index_col=False)
#dropping seconmd column not needed
df.drop(df.iloc[:,:0] , axis=1, inplace=True)
#sanity check
print(df)
#reading csv
df3=pd.read_csv("cleaned_dataset.csv")
#filtering out the between account transactions
df3=df3.loc[(df3["Categorised"] != "To account")]
#dropping NA values 
df3=df3.dropna(subset=["from_account"])
#sanity check
print(df3)
#sorting dataframe by account and category so its easier for me to read the output below
df3 = df3.sort_values(['from_account', 'Categorised'],
              ascending = [True, True])
#sanity check
print(df3)
#frquency/total for each account and category
df2=df3[["from_account", "Categorised"]].value_counts(normalize=True, sort=False)
#changing series to data frame
df2.to_frame()
#output check
print(df2)

