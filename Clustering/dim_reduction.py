from scipy.fft import dct
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection


def pca(X, n_components, random_state=42):
    model = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    )
    return model.fit_transform(X)


def rp(X, n_components, random_state=42):
    model = GaussianRandomProjection(
        n_components=n_components,
        random_state=random_state,
    )
    return model.fit_transform(X)


def srp(X, n_components, random_state=42):
    model = SparseRandomProjection(
        n_components=n_components,
        dense_output=True,
        random_state=random_state,
    )
    return model.fit_transform(X)


def dct_reduce(X, n_components):
    X_dct = dct(X, axis=1, norm="ortho")
    return X_dct[:, :n_components]
