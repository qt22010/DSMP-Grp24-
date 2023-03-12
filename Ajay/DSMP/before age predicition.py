import pandas as pd
import numpy as np
#reading in cleaned csv
df=pd.read_csv("cleaned_dataset.csv")
#filtering out the between account transactions
df=df.loc[(df["Categorised"] != "To account")]
#dropping na values from account
df=df.dropna(subset=["from_account"])
#sanity check
print(df)
#getting a unique list 
from_account=[df["from_account"].unique()]
#sort in order
from_account.sort()
#transposing the list and creating a dataframe
df1=pd.DataFrame((np.array(from_account).T))
#writing out to csv without the index so we dont duplicate index on read
df1.to_csv("unique_accounts.csv",index=False)
#renaming column
df3 = df1.rename(columns = {"0" : "from_account"})
#sanity check
print(df3)


