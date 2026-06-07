from __future__ import annotations

from sklearn.decomposition import TruncatedSVD, PCA

from .base import BaseReducer


class RandomizedSVDReducer(BaseReducer):
    """Randomized SVD via TruncatedSVD(algorithm='randomized').

    This is not centered PCA. It is suitable for sparse text-like matrices.
    """

    name = "randomized_svd"

    def __init__(self, n_components: int, random_state: int = 42, n_iter: int = 5):
        super().__init__(n_components=n_components, random_state=random_state)
        self.n_iter = n_iter
        self.model = TruncatedSVD(
            n_components=n_components,
            algorithm="randomized",
            n_iter=n_iter,
            random_state=random_state,
        )
    # self.model = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)


    def fit(self, X):
        self.model.fit(X)
        return self

    def fit_transform(self, X):
        return self.model.fit_transform(X)

    def transform(self, X):
        return self.model.transform(X)

    def params(self) -> dict:
        p = super().params()
        p.update({"algorithm": "randomized", "n_iter": self.n_iter})
        return p
