from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def kmeans(X, n_clusters=10, max_iter=300, random_state=42):
    """Return K-means labels and centroids."""
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=max_iter,
        random_state=random_state,
    )
    labels = model.fit_predict(X)

    return labels, model.cluster_centers_


def gmm(X, n_components=10, max_iter=300, random_state=42):
    """Return GMM labels and parameters."""
    model = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        max_iter=max_iter,
        n_init=3,
        reg_covar=1e-4,
        random_state=random_state,
    )
    labels = model.fit_predict(X)

    return labels, model.means_, model.covariances_, model.weights_
