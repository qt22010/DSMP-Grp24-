# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 01:10:35 2023

@author: white
"""
import pandas as pd
df10=pd.read_csv(r"C:\Users\qt22010\Downloads\features_seconds.csv")
df=pd.read_csv(r"C:\Users\qt22010\Downloads\features_seconds.csv")
print(df)





df1=pd.read_csv(r"C:\Users\qt22010\Downloads\features_second.csv")
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
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 4))
sns.histplot(df1['Recency'], bins=20, ax=axes[0])
sns.histplot(df1['Frequency'], bins=20, ax=axes[1])
sns.histplot(df1['Monetary'], bins=20, ax=axes[2])
plt.suptitle('Distribution of RFM Metrics')
plt.show()

# Plot the distribution of RFM scores by segment
sns.set_style('whitegrid')
plt.figure(figsize=(9, 6))
sns.histplot(df1, x='RFM_Score', hue='Segment', multiple='stack', bins=50)
plt.title('Distribution of RFM Scores by Segment')
plt.show()

import numpy as np
#

print(df_log.columns)
df_log=df_log.drop(columns=['R_Quartile',
       'F_Quartile', 'M_Quartile', 'RFM_Score', 'Segment'])

print(df_log)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import zscore
df_logz=df_log.apply(zscore)
print(df_logz)

df_logz.to_csv(r"C:\Users\qt22010\Downloads\rfmlogbasestand.csv")


fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs[0].hist(df_logz['Recency'])
axs[0].set_title('Recency')
axs[1].hist(df_logz['Frequency'])
axs[1].set_title('Frequency')
axs[2].hist(df_logz['Monetary'])
axs[2].set_title('Monetary')
df_logz1=df_logz
df_logz = df_logz[["Recency", "Frequency", "Monetary"]]


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_logz)
    wcss.append(kmeans.inertia_)
fig, axs = plt.subplots(1, 1, figsize=(16,16))
plt.plot(range(1, 11), wcss)

plt.xlabel('Number of clusters', fontdict = {'fontsize' : 40})
plt.ylabel('WCSS',  fontdict = {'fontsize' : 40})

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
df_logz3c=df_logz1
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(df_logz3)


# Add the cluster labels to the original DataFrame
df_logz3c['Cluster'] = clusters
snake_plot(df_logz3c)


df_logz4=df_logz
df_logz4c=df_logz1
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(df_logz4)


# Add the cluster labels to the original DataFrame
df_logz4c['Cluster'] = clusters
snake_plot(df_logz4c)

df_logz5=df_logz
df_logz5c=df_logz1
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(df_logz5)


# Add the cluster labels to the original DataFrame
df_logz5c['Cluster'] = clusters
snake_plot(df_logz5c)

df_logz4=df_logz
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(df_logz4)

# Analyze the resulting clusters to identify common characteristics and behaviors of customers in each cluster
"""cluster_summary = df_logz.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
})
print(cluster_summary)

plt.scatter(df_logz4c['Recency'], df_logz4c['Frequency'], c=df_logz4c['Cluster'])
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.show()"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# create a figure and a 3D axes object
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')


df_logz4=df_logz
df_logz4c=df_logz1
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(df_logz4)


# Add the cluster labels to the original DataFrame
df_logz4c['Cluster'] = clusters
print(df_logz4c)
print(df_logz4c["Cluster"].unique())

print(df)

# specify colors for each cluster label
colors = ['r', 'g', 'b', 'm', 'c']

# plot the data points
for i in range(5):
    ax.scatter(df1['Recency'][df_logz4c["Cluster"]==i], 
               df1['Frequency'][df_logz4c["Cluster"]==i], 
               df1['Monetary'][df_logz4c["Cluster"]==i], 
               c=colors[i])
# set the axis labels
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')

colors = plt.cm.cool(df_logz4c)

# show the plot
plt.show()
df1.to_csv(r"C:\Users\qt22010\Downloads\graphs.csv")
df_logz4c.to_csv(r"C:\Users\qt22010\Downloads\infordb.csv")
print(df_logz4c)
from sklearn.metrics import silhouette_score
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(df_logz4)
df_logz4c['Cluster'] = clusters
# calculate silhouette score
silhouette_avg = silhouette_score(df_logz4, clusters)
print("The average silhouette_score is :", silhouette_avg)

kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(df_logz4)
df_logz4c['Cluster'] = clusters
# calculate silhouette score
silhouette_avg = silhouette_score(df_logz4, clusters)
print("The average silhouette_score is :", silhouette_avg)

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(df_logz4)
df_logz4c['Cluster'] = clusters
# calculate silhouette score
silhouette_avg = silhouette_score(df_logz4, clusters)
print("The average silhouette_score is :", silhouette_avg)

kmeans = KMeans(n_clusters=6, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(df_logz4)
df_logz4c['Cluster'] = clusters
# calculate silhouette score
silhouette_avg = silhouette_score(df_logz4, clusters)
print("The average silhouette_score is :", silhouette_avg)

print(df_logz4c)