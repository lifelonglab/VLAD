import math
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans


def k_means_summarization(data, max_data_length):
    n_clusters = 5
    if len(data) < n_clusters:
        return np.array(data[max_data_length:])
    single_cluster_length = math.floor(max_data_length / n_clusters)
    clustering_model = KMeans(n_clusters=n_clusters)
    clustering_model.fit(data)
    all_distances = clustering_model.transform(data)
    assigned_clusters = clustering_model.predict(data)

    clusters = defaultdict(list)
    for cluster_id, point, distances in zip(assigned_clusters, data, all_distances):
        clusters[cluster_id].append((point, distances[cluster_id]))

    # choose what data to keep
    selected_data = []
    for cluster in clusters.values():
        cluster.sort(key=lambda x: x[1])  # sort by distance
        best_data_from_cluster = [p for p, _ in cluster[:single_cluster_length]]
        selected_data.append(best_data_from_cluster)

    return np.concatenate(selected_data)

