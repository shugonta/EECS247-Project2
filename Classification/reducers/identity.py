from __future__ import annotations

from .base import BaseReducer


class IdentityReducer(BaseReducer):
    name = "identity"

    def __init__(self, n_components=None, random_state: int = 42):
        super().__init__(n_components=n_components, random_state=random_state)

    def fit_transform(self, X):
        return X

    def transform(self, X):
        return X
