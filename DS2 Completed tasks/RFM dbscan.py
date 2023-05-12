# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:18:37 2023

@author: white
"""

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.model_selection import GridSearchCV
import pandas as pd



# Load the RFM dataset into a pandas DataFrame
rfm_df = pd.read_csv(r"C:\Users\qt22010\Downloads\rfmlogbasestand.csv")

# Specify the relevant columns as features
X = rfm_df[["Recency", "Frequency", "Monetary"]]


from scipy.spatial.distance import pdist
import numpy as np

# Calculate the pairwise distance matrix
distances = pdist(X)

# Calculate the standard deviation of the pairwise distances
std_dev = np.std(distances)

from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go

k = 6
nbrs = NearestNeighbors(n_neighbors=k ).fit(X)
distances, indices = nbrs.kneighbors(X)
distance_desc = sorted(distances[:,k-1], reverse=True)
fig=px.line(x=list(range(1,len(distance_desc )+1)),y= distance_desc )



from kneed import KneeLocator
kneedle = KneeLocator(range(1,len(distance_desc)+1),  #x values
                      distance_desc, # y values
                      S=1.0, #parameter suggested from paper
                      curve="convex", #parameter from figure
                      direction="decreasing") #parameter from figure

# Check if the knee point was found
if kneedle.knee is None:
    print("Warning: Knee point not found")
else:
    # Add a vertical line to the plot at the knee point
    line_shape = go.layout.Shape(type="line", x0=kneedle.knee, y0=0, x1=kneedle.knee, y1=max(distance_desc), line=dict(color='red', width=2, dash='dash'))
    fig.update_layout(width=1100, height=1500, template="plotly_white", font=dict(size=18), shapes=[line_shape])
    # Print the knee point and optimal value of eps
    knee_point = kneedle.knee
    eps = distance_desc[knee_point]
    print(f"Knee point: {knee_point}")
    print(f"Optimal value of eps: {eps:.3f}")
    fig.write_html('knee_plot.html', auto_open=True)
# Show the plot
fig.update_layout(width=1100, height=1500, template="plotly_white", font=dict(size=18), shapes=[line_shape])
# Print the knee point and optimal value of eps
fig.show()

kneedle.plot_knee_normalized(figsize=(12, 8))

kneedle.elbow

kneedle.knee_y





# Set up the parameter grid for DBSCAN
param_grid = {
    "eps": [1.3809],
    "min_samples": [6,7,8,9,10,11,12,13,14,15]}

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
dbscan_result = DBSCAN(eps=best_eps, min_samples=best_min_samples, n_jobs=-1).fit(X)

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
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b', 'm',]
for i in [0,1,-1,2]:
    ax.scatter(df['Recency'][rfm_df["Cluster"]==i], 
               df['Frequency'][rfm_df["Cluster"]==i], 
               df['Monetary'][rfm_df["Cluster"]==i],
               c = colors[i])


# set the axis labels
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')

# show the plot
plt.show()




# show the plot
"""
plt.show()

for i in np.arange(0.1,2,0.1):
    for j in np.arange(3,8,1):
        # Set up the parameter grid for DBSCAN
       
        # Run DBSCAN on the dataset with the best parameters
        dbscan_result = DBSCAN(eps=i, min_samples=j, n_jobs=-1).fit(X)
        
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
        title="My Plot (Param 1: {:.2f}, Param 2: {})".format(i, j)
        plt.title(title)
        # show the plot
        plt.show()
"""
        
        
