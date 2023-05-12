# -*- coding: utf-8 -*-
"""
Created on Wed May 10 01:47:12 2023

@author: qt22010
"""
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, make_scorer
df = pd.read_csv(r"C:\Users\qt22010\Downloads\infordb.csv")
df2 = df[(df['Cluster'] == 2) | (df['Cluster'] == 3)]
dfmerge =  df[~((df['Cluster'] == 2) | (df['Cluster'] == 3))]

print(dfmerge.shape, df2.shape)

X = df2[["Recency", "Frequency", "Monetary"]]

df2.drop(columns="Cluster")


   
# Run DBSCAN on the dataset with the best parameters
dbscan_result = DBSCAN(eps=0.9, min_samples=5, n_jobs=-1).fit(X)

# Add the cluster labels to the RFM dataset
df2['Cluster'] = dbscan_result.labels_

# Print the number of clusters and the cluster sizes
print("Number of clusters:", len(df2['Cluster'].unique()))
print("Cluster sizes:")
print(df2['Cluster'].value_counts())

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


df=pd.read_csv(r"C:\Users\qt22010\Downloads\graphs.csv")

# create a figure and a 3D axes object
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df2['Recency'], df2['Frequency'], df2['Monetary'], c=df2['Cluster'])

# set the axis labels
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
# show the plot
plt.show()

df2['Cluster'].replace(0, 'A', inplace=True)
df2['Cluster'].replace(1, 'B', inplace=True)
df2['Cluster'].replace(2, 'C', inplace=True)
df2['Cluster'].replace(3, 'D', inplace=True)

df2['Cluster'].replace('A',2, inplace=True)
df2['Cluster'].replace('B',3, inplace=True)
df2['Cluster'].replace('C',5, inplace=True)
df2['Cluster'].replace('D',6, inplace=True)
print(df2)

graph=pd.concat([dfmerge,df2])
print(graph.shape)
print(graph)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# create a figure and a 3D axes object
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

colors = ['r','g','b','c','m',"y","k"]

# plot the data points
for i in [0,1,2,3,4,5,6,-1]:
    ax.scatter(graph['Recency'][graph["Cluster"]==i], 
               graph['Frequency'][graph["Cluster"]==i], 
               graph['Monetary'][graph["Cluster"]==i], 
               c=colors[i])
# set the axis labels
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')


# show the plot
plt.show()