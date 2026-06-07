from __future__ import annotations

from scipy import sparse
from sklearn.decomposition import PCA

from .base import BaseReducer, ReductionError


class PCASVDReducer(BaseReducer):
    """PCA using full SVD through scikit-learn's PCA(svd_solver='full')."""

    name = "pca_svd"

    def __init__(self, n_components: int, random_state: int = 42, allow_dense_sparse: bool = False):
        super().__init__(n_components=n_components, random_state=random_state)
        self.allow_dense_sparse = allow_dense_sparse
        self.model = PCA(n_components=n_components, svd_solver="full", random_state=random_state)

    def _prepare(self, X):
        if sparse.issparse(X):
            if not self.allow_dense_sparse:
                raise ReductionError(
                    "PCA full SVD requires dense input. Use --allow-dense-sparse-pca "
                    "or use randomized_svd for sparse datasets."
                )
            return X.toarray()
        return X

    def fit(self, X):
        self.model.fit(self._prepare(X))
        return self

    def fit_transform(self, X):
        return self.model.fit_transform(self._prepare(X))

    def transform(self, X):
        return self.model.transform(self._prepare(X))

    def params(self) -> dict:
        p = super().params()
        p.update({"svd_solver": "full", "allow_dense_sparse": self.allow_dense_sparse})
        return p
