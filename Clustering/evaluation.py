import time

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score


def ari(y_true, cluster_labels):
    return adjusted_rand_score(y_true, cluster_labels)


def nmi(y_true, cluster_labels):
    return normalized_mutual_info_score(y_true, cluster_labels)


def purity(y_true, cluster_labels):
    total = len(y_true)
    score = 0

    for cluster_id in np.unique(cluster_labels):
        cluster_classes = y_true[cluster_labels == cluster_id]
        if len(cluster_classes) == 0:
            continue
        score += np.bincount(cluster_classes).max()

    return score / total


def silhouette(X, cluster_labels, sample_size=2000, random_state=42):
    if X.shape[0] <= sample_size:
        return silhouette_score(X, cluster_labels)

    return silhouette_score(
        X,
        cluster_labels,
        sample_size=sample_size,
        random_state=random_state,
    )


def measure_runtime(function, *args, **kwargs):
    start_time = time.perf_counter()
    result = function(*args, **kwargs)
    runtime = time.perf_counter() - start_time
    return result, runtime


def distance_preservation(X_original, X_reduced, sample_size=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    n_samples = X_original.shape[0]

    if n_samples > sample_size:
        indices = rng.choice(n_samples, sample_size, replace=False)
        X_original = X_original[indices]
        X_reduced = X_reduced[indices]

    original_distances = pairwise_distances(X_original)
    reduced_distances = pairwise_distances(X_reduced)

    original_vector = original_distances[np.triu_indices_from(original_distances, k=1)]
    reduced_vector = reduced_distances[np.triu_indices_from(reduced_distances, k=1)]

    return np.corrcoef(original_vector, reduced_vector)[0, 1]
