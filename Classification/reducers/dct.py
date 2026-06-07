from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.fft import dct, dctn

from .base import BaseReducer, ReductionError


def _zigzag_indices_2d(h: int, w: int):
    coords = [(i, j) for i in range(h) for j in range(w)]
    coords.sort(key=lambda x: (x[0] + x[1], x[0]))
    return coords


class DCTReducer(BaseReducer):
    """DCT-based feature reducer.

    - For image datasets, uses 2D DCT per channel and keeps low-frequency coefficients.
    - For non-image datasets, uses 1D DCT over the feature axis and keeps the first k coefficients.
    """

    name = "dct"

    def __init__(self, n_components: int, random_state: int = 42, mode: str = "auto"):
        super().__init__(n_components=n_components, random_state=random_state)
        self.mode = mode
        self.image_shape = None
        self._selected = None

    def setup(self, dataset_bundle):
        super().setup(dataset_bundle)
        self.image_shape = dataset_bundle.image_shape
        return self

    def fit_transform(self, X):
        return self.transform(X)

    def transform(self, X):
        if sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        if self.image_shape is not None and (self.mode in ["auto", "image2d"]):
            return self._transform_images(X)
        return self._transform_1d(X)

    def _transform_1d(self, X):
        Z = dct(X, type=2, norm="ortho", axis=1)
        return Z[:, : self.n_components].astype(np.float32)

    def _transform_images(self, X):
        shape = tuple(self.image_shape)
        n = X.shape[0]
        if len(shape) == 2:
            h, w = shape
            images = X.reshape(n, h, w)
            coeffs = dctn(images, type=2, norm="ortho", axes=(1, 2))
            flat = self._select_image_coeffs(coeffs[..., None], h, w, 1)
        elif len(shape) == 3:
            h, w, c = shape
            images = X.reshape(n, h, w, c)
            coeffs = dctn(images, type=2, norm="ortho", axes=(1, 2))
            flat = self._select_image_coeffs(coeffs, h, w, c)
        else:
            raise ReductionError(f"Unsupported image shape for DCT: {shape}")
        return flat.astype(np.float32)

    def _select_image_coeffs(self, coeffs, h: int, w: int, c: int):
        if self._selected is None:
            coords = []
            for i, j in _zigzag_indices_2d(h, w):
                for ch in range(c):
                    coords.append((i, j, ch))
                    if len(coords) >= self.n_components:
                        break
                if len(coords) >= self.n_components:
                    break
            if len(coords) < self.n_components:
                raise ReductionError("Requested more DCT coefficients than available")
            self._selected = coords
        out = np.empty((coeffs.shape[0], len(self._selected)), dtype=np.float32)
        for idx, (i, j, ch) in enumerate(self._selected):
            out[:, idx] = coeffs[:, i, j, ch]
        return out

    def params(self) -> dict:
        p = super().params()
        p.update({"mode": self.mode})
        return p
