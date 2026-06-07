from __future__ import annotations

from .identity import IdentityReducer
from .random_projection import SparseRandomProjectionReducer, GaussianRandomProjectionReducer
from .pca_svd import PCASVDReducer
from .dct import DCTReducer
from .randomized_svd import RandomizedSVDReducer


def make_reducer(name: str, n_components, seed: int, args):
    name = name.lower()
    if name == "identity":
        return IdentityReducer(random_state=seed)
    if name == "sparse_rp":
        return SparseRandomProjectionReducer(n_components=n_components, random_state=seed, dense_output=None)
    if name == "pca_svd":
        return PCASVDReducer(n_components=n_components, random_state=seed, allow_dense_sparse=args.allow_dense_sparse_pca)
    if name == "dct":
        return DCTReducer(n_components=n_components, random_state=seed, mode=args.dct_mode)
    if name == "randomized_svd":
        return RandomizedSVDReducer(n_components=n_components, random_state=seed, n_iter=args.randomized_svd_iter)
    if name == "gaussian_rp":
        return GaussianRandomProjectionReducer(n_components=n_components, random_state=seed)
    raise ValueError(f"Unknown reducer: {name}")
