from matplotlib import pyplot as plt
import numpy as np
import json
from Kmeans import *

with open('data.json', 'r') as file:
    data = json.load(file)
data = np.array(data)

x = [0, 0, 1, 1]
y = [0, 1, 0, 1]

f, ax = plt.subplots(2, 2)
a = 3  # lower limit of number of clusters
k = np.arange(a, a+4, 1)
means = []
k_init = data[0]

sil_val = [[] for i in range(len(k))]
for k_val in k:
    clusters, centroids = kmeans(data, k_val, k_init)
    for i in range(k_val):  # Visualization of clusters
        ax[x[int(k_val - a)], y[int(k_val - a)]].plot([x for x, _ in clusters[i]],[y for _, y in clusters[i]], "o")
        ax[x[int(k_val - a)], y[int(k_val - a)]].scatter(centroids[i][0], centroids[i][1], s=80, color='k')
    ax[x[int(k_val - a)], y[int(k_val - a)]].set_title(f"{k_val} Clusters")

    for i in range(k_val):  # calculating silhouette score for every number of clusters
        s = []
        for j in range(len(clusters[i])):
            minimal = float('inf')
            c_cluster = 0
            for p in range(k_val):
                temp = dist(clusters[i][j], centroids[p])
                if p is not i and temp < minimal:
                    c_cluster = p
                    minimal = temp
            d = 0
            for point in clusters[i]:
                d += dist(point, clusters[i][j])
            ac = d/(len(clusters[i])-1)
            d = 0
            for point in clusters[c_cluster]:
                d += dist(point, clusters[i][j])
            b = d/(len(clusters[c_cluster])-1)
            s.append((b-ac)/max(ac,b))
        sil_val[k_val-a].append(sorted(s,reverse=True))
    mean = 0
    for i in sil_val[k_val-a]:
        mean += sum(i)/len(i)
    means.append(mean/len(sil_val[k_val-a]))
# Visualization of data
f.tight_layout()
f.show()
for k_val in k:
    for j in range(k_val):
        plt.plot(sil_val[k_val-a][j])
    plt.title(f"Silhouette score for {k_val} clusters")
    plt.xlabel("Points")
    plt.ylabel("Silhouette score")
    plt.grid()
    plt.show()
plt.plot(k, means, "o")
plt.grid()
plt.title("Average silhouette score")
plt.ylabel("Silhouette score")
plt.xlabel("Number of clusters")
plt.xticks(k)
plt.show()

print(f"Optimal number of clusters: {np.argmax(means)+a}")

