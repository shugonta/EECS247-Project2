from __future__ import annotations

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import sparse


from .base import BaseClassifier
import time
import numpy as np

class SVMClassifier(BaseClassifier):
    name = "svm"

    def __init__(self, C: float = 1.0, max_iter: int = 5000, random_state: int = 42, scale_dense: bool = False):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.scale_dense = scale_dense
        self.model = None

    def fit(self, X, y):
        clf = LinearSVC(C=self.C, max_iter=self.max_iter, random_state=self.random_state, dual="auto")
        if self.scale_dense and not sparse.issparse(X):
            self.model = make_pipeline(StandardScaler(), clf)
        else:
            self.model = clf
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def params(self) -> dict:
        return {"C": self.C, "max_iter": self.max_iter, "scale_dense": self.scale_dense}
    
class SGDHingeSVMClassifier(BaseClassifier):
    name = "svm_sgd"

    def __init__(
        self,
        alpha: float = 1e-4,
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state: int = 42,
        scale_dense: bool = False,
        learning_rate: str = "optimal",
        eta0: float = 0.01,
        average: bool = True,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 5,
        n_jobs: int = -1,
        verbose: int = 0,
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.scale_dense = scale_dense
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.average = average
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.model = None

    def fit(self, X, y):
        clf = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            shuffle=True,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            average=self.average,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        # For dense reduced features such as PCA/DCT outputs,
        # scaling often stabilizes SGD optimization.
        # For sparse input, use no centering to avoid densifying the matrix.
        if self.scale_dense and not sparse.issparse(X):
            self.model = make_pipeline(StandardScaler(), clf)
        elif self.scale_dense and sparse.issparse(X):
            self.model = make_pipeline(StandardScaler(with_mean=False), clf)
        else:
            self.model = clf

        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def params(self) -> dict:
        return {
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "scale_dense": self.scale_dense,
            "learning_rate": self.learning_rate,
            "eta0": self.eta0,
            "average": self.average,
            "early_stopping": self.early_stopping,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "n_jobs": self.n_jobs,
        }