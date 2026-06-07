from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
from scipy import sparse

TaskType = Literal["single_label"]


@dataclass
class DatasetBundle:
    """Container used by all dataset loaders."""

    name: str
    X_train: object
    X_test: object
    y_train: np.ndarray
    y_test: np.ndarray
    n_classes: int
    task_type: TaskType = "single_label"
    feature_kind: str = "unknown"  # dense_image, dense_sensor, sparse_text, etc.
    image_shape: Optional[Tuple[int, ...]] = None

    @property
    def train_shape(self):
        return self.X_train.shape

    @property
    def test_shape(self):
        return self.X_test.shape

    @property
    def input_dim(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def is_sparse(self) -> bool:
        return sparse.issparse(self.X_train)
