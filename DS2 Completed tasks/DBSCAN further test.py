# -*- coding: utf-8 -*-
"""
Created on Mon May  8 20:04:33 2023

@author: white
"""
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

rfm_df = pd.read_csv(r"C:\Users\qt22010\Downloads\infordb.csv")


X = rfm_df[["Recency", "Frequency", "Monetary", "Kmeans"]]
print(X.info())
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
transformed = ohe.fit_transform(X[['Kmeans']])
X[ohe.categories_[0]] = transformed.toarray()

X.drop(columns="Kmeans",axis=0)
# Set up the parameter grid for DBSCAN
param_grid = {
    "eps": np.arange(0.1,2,0.1),
    "min_samples": [2,3,4,5,6,7]}

# Set up the grid search for DBSCAN
dbscan_grid = GridSearchCV(
    DBSCAN(),
    param_grid=param_grid,
    scoring=silhouette_score,
    cv=5
)

# Fit the grid search to the data
dbscan_grid.fit(X)

# Get the best DBSCAN parameters
best_eps = dbscan_grid.best_params_["eps"]
best_min_samples = dbscan_grid.best_params_["min_samples"]



# Run DBSCAN on the dataset with the best parameters
dbscan_result = DBSCAN(eps=0.8, min_samples=3, n_jobs=-1).fit(X)

# Add the cluster labels to the RFM dataset
rfm_df['Cluster'] = dbscan_result.labels_



# Print the number of clusters and the cluster sizes
print("Number of clusters:", len(rfm_df['Cluster'].unique()))
print("Cluster sizes:")
print(rfm_df['Cluster'].value_counts())

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


df=pd.read_csv(r"C:\Users\qt22010\Downloads\graphs.csv")

# create a figure and a 3D axes object
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Recency'], df['Frequency'], df['Monetary'], c=rfm_df['Cluster'])

# set the axis labels
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')

# show the plot
plt.show()

print(best_eps)
print(best_min_samples)
    
print(rfm_df)

rfm_df.to_csv(r"C:\Users\qt22010\Downloads\graphs.csv")


print(X.info())