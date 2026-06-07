from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier

from .base import BaseClassifier


class KNNClassifier(BaseClassifier):
    name = "knn"

    def __init__(self, n_neighbors: int = 5, metric: str = "auto", n_jobs: int = -1):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_jobs = n_jobs
        self.model = None

    def _resolve_metric(self, X):
        if self.metric != "auto":
            return self.metric
        # Euclidean is a safe default for standardized dense features and reduced features.
        return "euclidean"

    def fit(self, X, y):
        metric = self._resolve_metric(X)
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            metric=metric,
            algorithm="brute",
            n_jobs=self.n_jobs,
        )
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def params(self) -> dict:
        return {"n_neighbors": self.n_neighbors, "metric": self.metric, "n_jobs": self.n_jobs}
