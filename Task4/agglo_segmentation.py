import numpy as np
import cv2


def euclidDistance(point1, point2):
    """
    Input : 2 lists
    Get euclidean distance of point1 and point2.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def clusters_distance(cluster1, cluster2):
    """
    Input : 2 cluster lists
    Get distance between two clusters.
    """
    return max([euclidDistance(point1, point2) for point1 in cluster1 for point2 in cluster2])


def clusters_distance_2(cluster1, cluster2):
    """
    Get distance between two centroids of the two clusters

    cluster1 and cluster2 are lists of lists of points
    """
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return euclidDistance(cluster1_center, cluster2_center)


class AgglomerativeClustering:

    def __init__(self, pixels, k=2, initial_k=25):
        self.k = k
        self.initial_k = initial_k
        self.pixels = pixels

    def initial_clusters(self, points):
        initial_groups = {}
        d = int(256 / self.initial_k)
        for i in range(self.initial_k):
            j = i * d
            initial_groups[(j, j, j)] = []
        for i, p in enumerate(points):
            if i % 100000 == 0:
                print('processing:', i)
            go = min(initial_groups.keys(), key=lambda c: euclidDistance(p, c))
            initial_groups[go].append(p)
        return [g for g in initial_groups.values() if len(g) > 0]

    def fit(self, points):
        # initially, assign each point to a distinct cluster
        self.clusters_list = self.initial_clusters(points)
        while len(self.clusters_list) > self.k:
            # Find the closest (most similar) pair of clusters
            cluster1, cluster2 = min(
                [(c1, c2) for i, c1 in enumerate(self.clusters_list) for c2 in self.clusters_list[:i]],
                key=lambda c: clusters_distance_2(c[0], c[1]))

            # Remove the two clusters from the clusters list
            self.clusters_list = [c for c in self.clusters_list if c != cluster1 and c != cluster2]

            # Merge the two clusters
            merged_cluster = cluster1 + cluster2

            # Add the merged cluster to the clusters list
            self.clusters_list.append(merged_cluster)

        self.cluster = {}
        for cl_num, cl in enumerate(self.clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num

        self.centers = {}
        for cl_num, cl in enumerate(self.clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)

    def calculate_cluster(self, point):
        """
        Get cluster number of point
        """
        return self.cluster[tuple(point)]

    def calculate_center(self, point):
        """
        Get center of the cluster for each point
        """
        point_cluster_num = self.calculate_cluster(point)
        center = self.centers[point_cluster_num]
        return center
