import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

img_path = "./images/Butterfly.jpg"
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# print(img.shape)
# plt.imshow(RGB_img)
# plt.axis('off')
# plt.show()
pixels = img.reshape((-1, 3))
# print(pixels.shape)


def euclidean_distance(point1, point2):
    """
    Computes euclidean distance of point1 and point2.

    point1 and point2 are lists.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def clusters_distance(cluster1, cluster2):
    """
    Computes distance between two clusters.

    cluster1 and cluster2 are lists of lists of points
    """
    return max([euclidean_distance(point1, point2) for point1 in cluster1 for point2 in cluster2])


def clusters_distance_2(cluster1, cluster2):
    """
    Computes distance between two centroids of the two clusters

    cluster1 and cluster2 are lists of lists of points
    """
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return euclidean_distance(cluster1_center, cluster2_center)


class AgglomerativeClustering:

    def __init__(self, k=2, initial_k=25):
        self.k = k
        self.initial_k = initial_k
    # def __init__(self, pixels, k=2, initial_k=25):
    #     self.k = k
    #     self.initial_k = initial_k
    #     self.pixels = pixels

    def initial_clusters(self, points):
        """
        partition pixels into self.initial_k groups based on color similarity
        """
        groups = {}
        d = int(256 / self.initial_k)
        for i in range(self.initial_k):
            j = i * d
            groups[(j, j, j)] = []
        for i, p in enumerate(points):
            if i % 100000 == 0:
                print('processing pixel:', i)
            go = min(groups.keys(), key=lambda c: euclidean_distance(p, c))
            groups[go].append(p)
        return [g for g in groups.values() if len(g) > 0]

    def fit(self, points):

        # initially, assign each point to a distinct cluster
        print('Computing initial clusters ...')
        self.clusters_list = self.initial_clusters(points)
        print('number of initial clusters:', len(self.clusters_list))
        print('merging clusters ...')

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

            print('number of clusters:', len(self.clusters_list))

        print('assigning cluster num to each point ...')
        self.cluster = {}
        for cl_num, cl in enumerate(self.clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num

        print('Computing cluster centers ...')
        self.centers = {}
        for cl_num, cl in enumerate(self.clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)

    def predict_cluster(self, point):
        """
        Find cluster number of point
        """
        # assuming point belongs to clusters that were computed by fit functions
        return self.cluster[tuple(point)]

    def predict_center(self, point):
        """
        Find center of the cluster that point belongs to
        """
        point_cluster_num = self.predict_cluster(point)
        center = self.centers[point_cluster_num]
        return center


n_clusters = 3
agglo = AgglomerativeClustering(k=n_clusters, initial_k=25)
agglo.fit(pixels)
new_img = [[agglo.predict_center(list(pixel)) for pixel in row] for row in img]
new_img = np.array(new_img, np.uint8)

plt.figure(figsize=(15,15))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.axis('off')
plt.title('Original image')

plt.subplot(1,2,2)
plt.imshow(new_img)
plt.axis('off')
plt.title(f'Segmented image with k={n_clusters}')

plt.show()
