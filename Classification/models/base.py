from __future__ import annotations

from abc import ABC, abstractmethod


class ModelError(RuntimeError):
    pass


class BaseClassifier(ABC):
    name: str

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def params(self) -> dict:
        return {}
