from __future__ import annotations

from scipy import sparse
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection

from .base import BaseReducer


class SparseRandomProjectionReducer(BaseReducer):
    name = "sparse_rp"

    def __init__(self, n_components: int, random_state: int = 42, dense_output: bool | None = None):
        super().__init__(n_components=n_components, random_state=random_state)
        self.dense_output = dense_output
        self.model = None

    def setup(self, dataset_bundle):
        super().setup(dataset_bundle)
        if self.dense_output is None:
            # If dense_output is not explicitly set, determine it based on whether the dataset is sparse
            self.dense_output = not dataset_bundle.is_sparse
        self.model = SparseRandomProjection(
            n_components=self.n_components,
            density="auto",
            random_state=self.random_state,
            dense_output=bool(self.dense_output),
        )
        return self

    def fit(self, X):
        if self.model is None:
            raise RuntimeError("Reducer must be setup before use")
        return self.model.fit(X)

    def fit_transform(self, X):
        if self.model is None:
            raise RuntimeError("Reducer must be setup before use")
        return self.model.fit_transform(X)

    def transform(self, X):
        if self.model is None:
            raise RuntimeError("Reducer must be setup before use")
        return self.model.transform(X)

    def params(self) -> dict:
        p = super().params()
        p.update({"dense_output": self.dense_output})
        return p

class GaussianRandomProjectionReducer(BaseReducer):
    name = "gaussian_rp"

    
    def __init__(self, n_components: int, random_state: int = 42):
        super().__init__(n_components=n_components, random_state=random_state)
        self.model = None

    def setup(self, dataset_bundle):
        super().setup(dataset_bundle)
        self.model = GaussianRandomProjection(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        return self
    
    def fit(self, X):
        if self.model is None:
            raise RuntimeError("Reducer must be setup before use")
        return self.model.fit(X)

    def fit_transform(self, X):
        if self.model is None:
            raise RuntimeError("Reducer must be setup before use")
        return self.model.fit_transform(X)

    def transform(self, X):
        if self.model is None:
            raise RuntimeError("Reducer must be setup before use")
        return self.model.transform(X)

    def params(self) -> dict:
        p = super().params()
        return p
