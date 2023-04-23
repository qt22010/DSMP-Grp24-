# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 01:10:35 2023

@author: white
"""
import pandas as pd
df10=pd.read_csv(r"C:\\users\white\Downloads\features_seconds.csv")
df=pd.read_csv(r"C:\\users\white\Downloads\features_seconds.csv")
print(df)





df1=pd.read_csv("C://users/white/Desktop/DSMP/features_second.csv")
df = df1.merge(df, on="Account Number")
df=df.sort_values(by="Unnamed: 0")
df=df.reset_index()
df= df.drop(["index", "Unnamed: 0"], axis =1)
print(df.head())

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs[0].hist(df['Recency'])
axs[0].set_title('Recency')
axs[1].hist(df['Frequency'])
axs[1].set_title('Frequency')
axs[2].hist(df['Monetary'])
axs[2].set_title('Monetary')

# Calculate quartiles for each RFM metric
quantiles = df.quantile(q=[0.2,0.4,0.6,0.8])

# Assign scores based on quartiles
def assign_quartile_score(x, metric, quantiles):
    if x <= quantiles[metric][0.2]:
        return 1
    elif x <= quantiles[metric][0.4]:
        return 2
    elif x <= quantiles[metric][0.6]:
        return 3
    elif x <= quantiles[metric][0.8]:
        return 4
    else:
        return 5
    
df['R_Quartile'] = df['Recency'].apply(assign_quartile_score, args=('Recency', quantiles))
df['F_Quartile'] = df['Frequency'].apply(assign_quartile_score, args=('Frequency', quantiles))
df['M_Quartile'] = df['Monetary'].apply(assign_quartile_score, args=('Monetary', quantiles))

# Combine the scores for each metric to create an RFM score
df['RFM_Score'] = df['R_Quartile'].astype(str) + df['F_Quartile'].astype(str) + df['M_Quartile'].astype(str)

print(df)
print(df['RFM_Score'].unique())

# Define the segments based on RFM scores
segments = {
    'High Value': range(555, 444, -1),
    'Mid-High Value': range(444, 333, -1),
    'Mid Value': range(333, 222, -1),
    'Mid-Low Value': range(222, 111, -1),
    'Low Value': range(111, 0, -1)
}


df['Segment'] = df['RFM_Score'].apply(lambda x: [k for k, i in segments.items() if int(x) in i][0])
print(df)

print(df.columns)
df1=df.drop(columns=['Amount', 'Balance', 'Third Party Account Number',
       'Third Party Name', 'Date', 'Time', 'transaction_ID', 'Decimal Time',
       'Income', 'Status', 'Type_Day', 'Day'])
print(df1)
df1=df1.drop_duplicates()
print(df1)


# Plot the distribution of RFM scores for each segment
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
sns.histplot(df1['Recency'], bins=20, ax=axes[0])
sns.histplot(df1['Frequency'], bins=20, ax=axes[1])
sns.histplot(df1['Monetary'], bins=20, ax=axes[2])
plt.suptitle('Distribution of RFM Metrics by Segment')
plt.show()

# Plot the distribution of RFM scores by segment
sns.set_style('whitegrid')
plt.figure(figsize=(8, 6))
sns.histplot(df1, x='RFM_Score', hue='Segment', multiple='stack', bins=50)
plt.title('Distribution of RFM Scores by Segment')
plt.show()

import numpy as np
df_log=df1
df_log["Recency"] = np.log(df1["Recency"] + 1)
df_log["Frequency"] = np.log(df1["Frequency"] + 1)
df_log["Monetary"] = np.log(df1["Monetary"] + 1)
f,ax = plt.subplots(1,3,figsize=(9,9))
sns.distplot(df_log['Recency'],kde=True, ax=ax[0])
ax[0].set_title('')
sns.distplot(df_log['Frequency'],kde=True, ax=ax[1])
ax[1].set_title('')
sns.distplot(df_log['Monetary'],kde=True, ax=ax[2])
ax[2].set_title('')
plt.show()

print(df_log.columns)
df_log=df_log.drop(columns=['R_Quartile',
       'F_Quartile', 'M_Quartile', 'RFM_Score', 'Segment'])

print(df_log)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import zscore
df_logz=df_log.apply(zscore)
print(df_logz)

fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs[0].hist(df_logz['Recency'])
axs[0].set_title('Recency')
axs[1].hist(df_logz['Frequency'])
axs[1].set_title('Frequency')
axs[2].hist(df_logz['Monetary'])
axs[2].set_title('Monetary')

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_logz)
    wcss.append(kmeans.inertia_)
fig, axs = plt.subplots(1, 1, figsize=(5,5))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#snakeplots


def snake_plot(normalised_df_rfm):
    
    # Melt data into long format
    df_melt = pd.melt(normalised_df_rfm.reset_index(), 
                      id_vars=['Account Number', 'Cluster'],
                      value_vars=['Recency', 'Frequency', 'Monetary'], 
                      var_name='Metric', 
                      value_name='Value')

    # Plot a line for each cluster
    plt.figure(figsize=(10, 5))
  

    
    # Set axis labels
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Snake plot of RFM for each cluster')
    sns.pointplot(data=df_melt, x='Metric', y='Value', hue='Cluster')
    
    return
df_logz3=df_logz
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(df_logz3)


# Add the cluster labels to the original DataFrame
df_logz3['Cluster'] = clusters
snake_plot(df_logz3)


df_logz4=df_logz
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(df_logz4)


# Add the cluster labels to the original DataFrame
df_logz4['Cluster'] = clusters
snake_plot(df_logz4)

df_logz5=df_logz
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(df_logz5)


# Add the cluster labels to the original DataFrame
df_logz5['Cluster'] = clusters
snake_plot(df_logz5)

# Analyze the resulting clusters to identify common characteristics and behaviors of customers in each cluster
cluster_summary = df_logz.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
})
print(cluster_summary)

plt.scatter(df_logz['Recency'], df_logz['Frequency'], c=df_logz['Cluster'])
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# create a figure and a 3D axes object
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot the data points
ax.scatter(df_logz['Recency'], df_logz['Frequency'], df_logz['Monetary'], c=df_logz['Cluster'])

# set the axis labels
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')

# show the plot
plt.show()


