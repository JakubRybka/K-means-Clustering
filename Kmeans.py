def clustering(data, centroids):
    k = len(centroids)
    clusters = [[] for _ in range(k)]
    for elem in data:
        min_dist = float('inf')
        kc = 0
        for k_val in range(k):
            distance = dist(elem, centroids[k_val])**2
            if distance < min_dist:
                min_dist = distance
                kc = k_val
        clusters[kc].append(elem)
    return clusters


def calc_centroids(clusters):
    centroids = []
    for cluster in clusters:
        xm = sum([elem[0] for elem in cluster])/len(cluster)
        ym = sum([elem[1] for elem in cluster]) / len(cluster)
        centroids.append([xm, ym])
    return centroids


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def kmeans(data, k, k_init):  # Takes data number of clusters and initial centroid
    k_points = [k_init]
    for i in range(1, k):
        point = data[0]
        maximal = 0
        for elem in data:
            c_dist = 0
            for j in range(i):  # algorithm to find next centroids
                c_dist += dist(elem, k_points[j]) / i
            if c_dist > maximal:
                maximal = c_dist
                point = elem
        k_points.append(point)
    n_cen = calc_centroids(clustering(data, k_points))
    clusters = clustering(data, n_cen)
    for _ in range(10):
        n_cen = calc_centroids(clustering(data, n_cen))
        clusters = clustering(data, n_cen)
    return clusters, n_cen   # returns data in clusters and centroids
