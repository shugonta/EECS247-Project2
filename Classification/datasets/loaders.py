from __future__ import annotations

import os
import urllib.request
import zipfile
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from sklearn.datasets import fetch_openml, fetch_rcv1, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import check_random_state

from .base import DatasetBundle


def subsample(X, y, max_samples: Optional[int], seed: int):
    if max_samples is None or X.shape[0] <= max_samples:
        return X, y
    rng = check_random_state(seed)
    idx = rng.choice(X.shape[0], size=max_samples, replace=False)
    if sparse.issparse(X):
        return X[idx], y[idx]
    return X[idx, :], y[idx]


def _safe_train_test_split(X, y, train_size=None, test_size=0.2, seed=42):
    # Stratification can fail when some classes have too few samples.
    try:
        return train_test_split(X, y, train_size=train_size, test_size=test_size, stratify=y, random_state=seed)
    except ValueError:
        return train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=seed)


def load_fashion_mnist(
    data_dir: str,
    seed: int,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
    standardize: bool = True,
) -> DatasetBundle:
    # Load Fashion-MNIST from OpenML.
    dataset = fetch_openml(data_id=40996, as_frame=False, data_home=data_dir)
    X = dataset.data.astype(np.float32) / 255.0
    y = dataset.target.astype(np.int64)

    X_train, X_test, y_train, y_test = _safe_train_test_split(
        X, y, train_size=60000, test_size=10000, seed=seed
    )
    X_train, y_train = subsample(X_train, y_train, max_train, seed)
    X_test, y_test = subsample(X_test, y_test, max_test, seed + 1)

    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    return DatasetBundle(
        name="fashion_mnist",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_classes=10,
        feature_kind="dense_image",
        image_shape=(28, 28),
    )


def channelwise_normalize_rgb_images(
    X_train_img: np.ndarray,
    X_test_img: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize RGB images using channel-wise mean and std from training data.

    Inputs are expected to have shape (N, H, W, C). The mean and std are
    computed over the training samples and spatial dimensions for each channel.
    """
    mean = X_train_img.mean(axis=(0, 1, 2), keepdims=True)
    std = X_train_img.std(axis=(0, 1, 2), keepdims=True) + eps
    X_train_img = (X_train_img - mean) / std
    X_test_img = (X_test_img - mean) / std
    return X_train_img.astype(np.float32), X_test_img.astype(np.float32)


def load_cifar10(
    data_dir: str,
    seed: int,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
    standardize: bool = True,
) -> DatasetBundle:
    # Load CIFAR-10 via torchvision. Keep image shape until normalization,
    # then flatten RGB images to 3072-dimensional vectors.
    try:
        from torchvision.datasets import CIFAR10
    except ImportError as e:
        raise ImportError("CIFAR-10 loader requires torchvision. Install with: pip install torch torchvision") from e

    train_ds = CIFAR10(root=data_dir, train=True, download=True)
    test_ds = CIFAR10(root=data_dir, train=False, download=True)

    X_train_img = train_ds.data.astype(np.float32) / 255.0
    y_train = np.asarray(train_ds.targets, dtype=np.int64)
    X_test_img = test_ds.data.astype(np.float32) / 255.0
    y_test = np.asarray(test_ds.targets, dtype=np.int64)

    X_train_img, y_train = subsample(X_train_img, y_train, max_train, seed)
    X_test_img, y_test = subsample(X_test_img, y_test, max_test, seed + 1)

    if standardize:
        # CIFAR-10 is an RGB image dataset, so normalize each channel using
        # statistics computed from the training split instead of applying a
        # per-pixel StandardScaler after flattening.
        X_train_img, X_test_img = channelwise_normalize_rgb_images(
            X_train_img, X_test_img
        )

    X_train = X_train_img.reshape(X_train_img.shape[0], -1).astype(np.float32)
    X_test = X_test_img.reshape(X_test_img.shape[0], -1).astype(np.float32)

    return DatasetBundle(
        name="cifar10",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_classes=10,
        feature_kind="dense_image",
        image_shape=(32, 32, 3),
    )


def _download_uci_har(data_dir: str) -> str:
    # Download the UCI HAR dataset if it's not already downloaded.
    os.makedirs(data_dir, exist_ok=True)
    outer_zip_path = os.path.join(data_dir, "UCI_HAR_Dataset.zip")
    inner_zip_path = os.path.join(data_dir, "UCI HAR Dataset.zip")
    extract_dir = os.path.join(data_dir, "UCI HAR Dataset")
    if os.path.isdir(extract_dir):
        return extract_dir
    if not os.path.isfile(outer_zip_path):
        url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
        urllib.request.urlretrieve(url, outer_zip_path)
    if not os.path.isfile(inner_zip_path):
        with zipfile.ZipFile(outer_zip_path, "r") as zf:
            zf.extractall(data_dir)
    with zipfile.ZipFile(inner_zip_path, "r") as zf:
        zf.extractall(data_dir)
    return extract_dir


def load_uci_har(
    data_dir: str,
    seed: int,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
    standardize: bool = True,
) -> DatasetBundle:
    # Load the processed 561-feature UCI HAR split.
    root = _download_uci_har(data_dir)
    X_train = np.loadtxt(os.path.join(root, "train", "X_train.txt"), dtype=np.float32)
    y_train = np.loadtxt(os.path.join(root, "train", "y_train.txt"), dtype=np.int64) - 1
    X_test = np.loadtxt(os.path.join(root, "test", "X_test.txt"), dtype=np.float32)
    y_test = np.loadtxt(os.path.join(root, "test", "y_test.txt"), dtype=np.int64) - 1

    X_train, y_train = subsample(X_train, y_train, max_train, seed)
    X_test, y_test = subsample(X_test, y_test, max_test, seed + 1)

    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    return DatasetBundle(
        name="uci_har",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_classes=6,
        feature_kind="dense_sensor",
        image_shape=None,
    )


def multilabel_to_first_label(Y) -> np.ndarray:
    # Convert multilabel CSR matrix to a single label by taking the first positive label.
    Y = Y.tocsr()
    labels = np.empty(Y.shape[0], dtype=np.int64)
    for i in range(Y.shape[0]):
        start, end = Y.indptr[i], Y.indptr[i + 1]
        if start == end:
            labels[i] = -1
        else:
            labels[i] = int(Y.indices[start])
    mask = labels >= 0
    return labels, mask


def load_20newsgroups(
    data_dir: str,
    seed: int,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
    standardize: bool = True,
    max_features: Optional[int] = 50000,
    remove_metadata: bool = True,
) -> DatasetBundle:
    # Load the 20 Newsgroups dataset and convert it to a format suitable for machine learning.
    remove = ("headers", "footers", "quotes") if remove_metadata else ()

    train_data = fetch_20newsgroups(
        subset="train",
        data_home=data_dir,
        shuffle=True,
        random_state=seed,
        remove=remove,
    )

    test_data = fetch_20newsgroups(
        subset="test",
        data_home=data_dir,
        shuffle=True,
        random_state=seed,
        remove=remove,
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        min_df=2,
        dtype=np.float32,
        norm="l2",
    )

    X_train = vectorizer.fit_transform(train_data.data).tocsr()
    X_test = vectorizer.transform(test_data.data).tocsr()

    y_train = np.asarray(train_data.target, dtype=np.int64)
    y_test = np.asarray(test_data.target, dtype=np.int64)

    X_train, y_train = subsample(X_train, y_train, max_train, seed)
    X_test, y_test = subsample(X_test, y_test, max_test, seed + 1)

    # TF-IDF is already normalized by default when norm="l2".
    # If additional scaling is needed, use MaxAbsScaler to preserve sparsity.
    if standardize:
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train).tocsr()
        X_test = scaler.transform(X_test).tocsr()

    return DatasetBundle(
        name="20newsgroups",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_classes=len(train_data.target_names),
        feature_kind="sparse_text",
        image_shape=None,
    )

def load_rcv1(
    data_dir: str,
    seed: int,
    max_train: Optional[int] = 10000,
    max_test: Optional[int] = 2000,
    standardize: bool = True,
) -> DatasetBundle:
    data = fetch_rcv1(data_home=data_dir)
    X = data.data
    labels, mask = multilabel_to_first_label(data.target)
    X = X[mask]
    y = labels[mask]

    X, y = subsample(X, y, None if (max_train is None or max_test is None) else max_train + max_test, seed)
    X_train, X_test, y_train, y_test = _safe_train_test_split(
        X, y, train_size=max_train, test_size=max_test if max_test is not None else 0.2, seed=seed
    )

    if standardize:
        # Preserve sparsity. MaxAbsScaler is suitable for sparse matrices.
        scaler = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return DatasetBundle(
        name="rcv1",
        X_train=X_train,
        X_test=X_test,
        y_train=np.asarray(y_train, dtype=np.int64),
        y_test=np.asarray(y_test, dtype=np.int64),
        n_classes=int(np.max(y) + 1),
        feature_kind="sparse_text",
        image_shape=None,
    )


def load_dataset(
    name: str,
    data_dir: str,
    seed: int,
    max_train: Optional[int],
    max_test: Optional[int],
    standardize: bool = True,
) -> DatasetBundle:
    name = name.lower()
    if name == "fashion_mnist":
        return load_fashion_mnist(data_dir, seed, max_train, max_test, standardize)
    if name == "cifar10":
        return load_cifar10(data_dir, seed, max_train, max_test, standardize)
    if name == "uci_har":
        return load_uci_har(data_dir, seed, max_train, max_test, standardize)
    if name == "rcv1":
        return load_rcv1(data_dir, seed, max_train, max_test, standardize)
    if name == "20newsgroups":
        return load_20newsgroups(data_dir, seed, max_train, max_test, standardize)
    raise ValueError(f"Unknown dataset: {name}")
