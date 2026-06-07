from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class ReductionError(RuntimeError):
    pass


class BaseReducer(ABC):
    name: str

    def __init__(self, n_components: Optional[int], random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def setup(self, dataset_bundle):
        """Optional dataset-dependent setup hook."""
        self.dataset_bundle = dataset_bundle
        return self

    def fit(self, X):
        return self

    @abstractmethod
    def fit_transform(self, X):
        raise NotImplementedError

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError

    def output_dim(self, X_transformed) -> int:
        return int(X_transformed.shape[1])

    def params(self) -> dict:
        return {"n_components": self.n_components, "random_state": self.random_state}
