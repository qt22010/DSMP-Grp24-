import pandas as pd
"""df_unique=pd.read_csv("unique_accounts.csv")
df_ratio=pd.read_csv("ratio.csv")"""

import pandas as pd
import numpy as np
#reading in cleaned csv
df=pd.read_csv(r"C:/Users/white/Desktop/ageprob.csv",)
#sanity check
print(df)
#printing columns 
print(df.columns)
#removing unecessary columns easier to read output 
df=df.drop(columns=["ratio", "18-34", "35-54", "55+", "Categorised"])
#sanity check
print(df)

#calculating the sum of probabilities per age 
a18_34=df.groupby("from_account")[["18-34x"]].sum()
#merging the sum back into the datagrame
df=df.merge(a18_34, how="left",on="from_account",  )
a35_54=df.groupby("from_account")[["35-54x"]].sum()
df=df.merge(a35_54, how="left",on="from_account",  )
a55_=df.groupby("from_account")[["55+x"]].sum()
df=df.merge(a55_, how="left",on="from_account",  )
print(df)

#dropping columns as we no longer need them + easier to read output
df=df.drop(columns=["18-34x_x", "35-54x_x", "55+x_x"])
#removiing duplicate rows we only need one row of summed probability per account
df=df.drop_duplicates()
#resetting index
df=df.reset_index()
#dropping the previous index column
df=df.drop(columns=["index"])
#renaming columns for easier reads or future merges
df=df.rename({"18-34x_y": "18-34", "35-54x_y": "35-54", "55+x_y":"55+"}, axis=1)
#sanity check
print(df)

#finding the most likely age per account
df["age"] = df[["18-34","35-54","55+"]].idxmax(1)
#dropping columns as they are no longer needed
df=df.drop(columns=["18-34", "35-54", "55+"])
#sanity check
print(df)

#viewing number of accounts categorised in each age group
print(df["age"].value_counts())
