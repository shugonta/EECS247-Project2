from collections import Counter
from pathlib import Path
import pickle
import tarfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import fetch_openml

from feature_extraction import load_cifar10_cnn, load_fashion_mnist_hog


DATA_DIR = Path(__file__).resolve().parent / "data"

FASHION_LABELS = [
    "0_tshirt_top",
    "1_trouser",
    "2_pullover",
    "3_dress",
    "4_coat",
    "5_sandal",
    "6_shirt",
    "7_sneaker",
    "8_bag",
    "9_ankle_boot",
]


def download_file(url, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, output_path)


def prepare_fashion_mnist():
    dataset = fetch_openml(data_id=40996, as_frame=False, data_home=DATA_DIR / "openml")
    X = dataset.data.astype(np.float32) / 255.0
    y = dataset.target.astype(np.int64)
    output_dir = DATA_DIR / "openml" / "images" / "fashion_mnist"

    for idx, (image, label) in enumerate(zip(X, y)):
        label_dir = output_dir / FASHION_LABELS[label]
        label_dir.mkdir(parents=True, exist_ok=True)

        image_path = label_dir / f"{idx:05d}.png"
        if image_path.exists():
            continue

        image_array = (image.reshape(28, 28) * 255).astype(np.uint8)
        Image.fromarray(image_array).save(image_path)

    print(f"Fashion-MNIST is ready in {output_dir}")


def prepare_cifar10():
    cifar_dir = DATA_DIR / "cifar10"
    archive_path = cifar_dir / "cifar-10-python.tar.gz"
    extracted_dir = cifar_dir / "cifar-10-batches-py"
    output_dir = cifar_dir / "images"

    download_file(
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        archive_path,
    )

    if not extracted_dir.exists():
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(cifar_dir)

    with open(extracted_dir / "batches.meta", "rb") as file:
        meta = pickle.load(file, encoding="latin1")
    label_names = [f"{idx}_{name}" for idx, name in enumerate(meta["label_names"])]

    image_idx = 0
    for batch_id in range(1, 6):
        with open(extracted_dir / f"data_batch_{batch_id}", "rb") as file:
            batch = pickle.load(file, encoding="latin1")

        for image, label in zip(batch["data"], batch["labels"]):
            label_dir = output_dir / label_names[label]
            label_dir.mkdir(parents=True, exist_ok=True)

            image_path = label_dir / f"{image_idx:05d}.png"
            if not image_path.exists():
                image_array = (
                    image.reshape(3, 32, 32).transpose(1, 2, 0)
                ).astype(np.uint8)
                Image.fromarray(image_array).save(image_path)

            image_idx += 1

    print(f"CIFAR-10 is ready in {output_dir}")


def make_unique_names(names):
    counts = Counter()
    unique_names = []

    for name in names:
        counts[name] += 1
        if counts[name] == 1:
            unique_names.append(name)
        else:
            unique_names.append(f"{name}_{counts[name]}")

    return unique_names


def prepare_uci_har():
    har_dir = DATA_DIR / "uci_har"
    archive_path = har_dir / "UCI_HAR_Dataset.zip"
    extracted_dir = har_dir / "UCI HAR Dataset"
    output_path = har_dir / "uci_har.csv"

    download_file(
        "https://archive.ics.uci.edu/static/public/240/"
        "human+activity+recognition+using+smartphones.zip",
        archive_path,
    )

    if not extracted_dir.exists():
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(har_dir)

    nested_archive_path = har_dir / "UCI HAR Dataset.zip"
    if not extracted_dir.exists() and nested_archive_path.exists():
        with zipfile.ZipFile(nested_archive_path, "r") as zip_ref:
            zip_ref.extractall(har_dir)

    features = pd.read_csv(
        extracted_dir / "features.txt",
        sep=r"\s+",
        header=None,
        names=["feature_id", "feature_name"],
    )
    feature_names = make_unique_names(features["feature_name"].tolist())

    X_train = np.loadtxt(extracted_dir / "train" / "X_train.txt")
    y_train = np.loadtxt(extracted_dir / "train" / "y_train.txt", dtype=np.int64)
    subject_train = np.loadtxt(
        extracted_dir / "train" / "subject_train.txt",
        dtype=np.int64,
    )

    X_test = np.loadtxt(extracted_dir / "test" / "X_test.txt")
    y_test = np.loadtxt(extracted_dir / "test" / "y_test.txt", dtype=np.int64)
    subject_test = np.loadtxt(
        extracted_dir / "test" / "subject_test.txt",
        dtype=np.int64,
    )

    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    subjects = np.concatenate([subject_train, subject_test])
    split = ["train"] * len(y_train) + ["test"] * len(y_test)

    df = pd.DataFrame(X, columns=feature_names)
    df.insert(0, "split", split)
    df.insert(1, "subject", subjects)
    df.insert(2, "activity", y)
    df.to_csv(output_path, index=False)

    print(f"UCI HAR is ready in {har_dir}")


def prepare_fashion_mnist_hog():
    def load_cached_fashion_mnist():
        dataset = fetch_openml(
            data_id=40996,
            as_frame=False,
            data_home=DATA_DIR / "openml",
        )
        X = dataset.data.astype(np.float32) / 255.0
        y = dataset.target.astype(np.int64)
        return X, y

    load_fashion_mnist_hog(DATA_DIR, load_cached_fashion_mnist)
    print(f"Fashion-MNIST HOG is ready in {DATA_DIR / 'openml' / 'hog'}")


def prepare_cifar10_cnn():
    load_cifar10_cnn(DATA_DIR)
    print(f"CIFAR-10 CNN features are ready in {DATA_DIR / 'cifar10' / 'cnn_features'}")


if __name__ == "__main__":
    prepare_fashion_mnist()
    prepare_cifar10()
    prepare_fashion_mnist_hog()
    prepare_cifar10_cnn()
    prepare_uci_har()
    print(f"All datasets are ready in {DATA_DIR}")
