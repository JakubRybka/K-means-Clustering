# K-means-Clustering
This project performs K-means clustering on a dataset and evaluates the clustering performance using silhouette scores. K-means clustering is a popular and straightforward unsupervised machine learning algorithm used for partitioning a dataset into a set of distinct, non-overlapping groups called clusters. Each data point belongs to the cluster with the nearest mean, serving as a prototype of the cluster. The program visualizes the clusters for different values of k and plots the silhouette scores to determine the optimal number of clusters.
## Programs
Kmeans.py file stores function implementation of K-means.

Silhouette_analysis.py is a program performing Kmeans clustering for multiple numbers of clusters.

data.json is exemplary numerical data to test clustering.

I have provided exemplary clustering and analysis results in form of plots.
## Usage
Execute the Silhouette_analysis to perform K-means clustering and silhouette analysis on data from file "data.json" (make sure such file exists).

``` python Silhouette_analysis.py ```
