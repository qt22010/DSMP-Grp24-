import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
df=pd.read_csv("clusteringtest.csv")
print(df.head())
df=df.rename(columns={'money_amount':'estimated_expenditure'})
income_multiplier=1.5
incomedf=df["estimated_expenditure"]*income_multiplier
df["estimated_income"]=incomedf
print(df.head())
del df["estimated_expenditure"]
print(df.head())
#kmeans test
from sklearn.cluster import KMeans
inertias = []

for i in range(1,30):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,30), inertias, marker='o')
plt.plot(range(1,30), inertias, marker='o')
plt.xlim([1, 10])
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=3)
label=kmeans.fit(df)
print(kmeans.labels_)
print(Counter(kmeans.labels_))
sns.scatterplot(data=df, x="from_account" ,y="estimated_expenditure" hue=kmeans.labels_)
plt.show()
#hierachial test

#gmm test