# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
df=pd.read_csv("C://users/white/Downloads/tester.csv")
print(df.head())
df=df.rename(columns={"date":"date_transaction"})
df["transaction_ID"]=df.index
print(len(df["Account Number"].unique()))
list1=df["Account Number"].unique()
print(list1)
i=0
listcopy=list1
strings = [str(x) for x in list1]
print(type(strings[40]))
#df["Account Number"].astype('string').dtypes
#df.to_csv("C://users/white/Downloads/tester2.csv",header=False,index= False)


import mysql.connector

# Connect to MySQL database
conn = mysql.connector.connect(
    host="database-1.ccxxkgtnwaut.us-east-1.rds.amazonaws.com",
    user="admin",
    password="password",
    database="dataset2"
)

# Check connection
if not conn.is_connected():
    print("Failed to connect to MySQL database")
    exit()

# Array of user names
user_names = strings

# Loop through user names
for user_name in user_names:
    table_name = "table_" + user_name # Generate table name based on user name

    # Check if table already exists
    cursor = conn.cursor()
    cursor.execute(f"SHOW TABLES LIKE '{table_name}';")
    result = cursor.fetchone()

    if result is None:
        # Table does not exist, create it
        cursor.execute(f"CREATE TABLE {table_name} AS SELECT * FROM dataset2 WHERE From_Account={user_name};")
        conn.commit()
        print(f"Table {table_name} created for user {user_name}")
    else:
        print(f"Table {table_name} already exists for user {user_name}")

# Close MySQL connection
cursor.close()
conn.close()