import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from clustering import gmm, kmeans
from dim_reduction import dct_reduce, pca, rp, srp
from evaluation import ari, distance_preservation, measure_runtime, nmi, purity, silhouette
from feature_extraction import load_cifar10_cnn, load_fashion_mnist_hog


RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


# Load Fashion-MNIST from OpenML cache.
def load_fashion_mnist():
    dataset = fetch_openml(
        data_id=40996,
        as_frame=False,
        data_home=DATA_DIR / "openml",
    )
    X = dataset.data.astype(np.float32) / 255.0
    y = dataset.target.astype(np.int64)
    return X, y


# Load CIFAR-10 from local batch files.
def load_cifar10():
    cifar_dir = DATA_DIR / "cifar10" / "cifar-10-batches-py"

    X_parts = []
    y_parts = []
    for batch_id in range(1, 6):
        with open(cifar_dir / f"data_batch_{batch_id}", "rb") as file:
            batch = pickle.load(file, encoding="latin1")
        X_parts.append(batch["data"])
        y_parts.extend(batch["labels"])

    X = np.vstack(X_parts).astype(np.float32) / 255.0
    y = np.array(y_parts, dtype=np.int64)
    return X, y


# Load UCI HAR train/test feature files.
def load_uci_har():
    har_dir = DATA_DIR / "uci_har" / "UCI HAR Dataset"

    X_train = np.loadtxt(har_dir / "train" / "X_train.txt")
    y_train = np.loadtxt(har_dir / "train" / "y_train.txt", dtype=np.int64)
    X_test = np.loadtxt(har_dir / "test" / "X_test.txt")
    y_test = np.loadtxt(har_dir / "test" / "y_test.txt", dtype=np.int64)

    X = np.vstack([X_train, X_test]).astype(np.float32)
    y = np.concatenate([y_train, y_test]) - 1
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clustering experiments and save CSV.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["fashion_mnist", "fashion_mnist_hog", "cifar10", "cifar10_cnn", "uci_har"],
        choices=["fashion_mnist", "fashion_mnist_hog", "cifar10", "cifar10_cnn", "uci_har"],
    )
    parser.add_argument("--k", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--sample_size", type=int, default=10000)
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR / "results.csv"),
    )
    args = parser.parse_args()

    datasets = [
        ("fashion_mnist", load_fashion_mnist),
        ("fashion_mnist_hog", lambda: load_fashion_mnist_hog(DATA_DIR, load_fashion_mnist)),
        ("cifar10", load_cifar10),
        ("cifar10_cnn", lambda: load_cifar10_cnn(DATA_DIR)),
        ("uci_har", load_uci_har),
    ]
    datasets = [(name, loader) for name, loader in datasets if name in args.datasets]

    reduction_methods = [
        ("pca", pca),
        ("rp", rp),
        ("srp", srp),
        ("dct", dct_reduce),
    ]

    rows = []

    for dataset_name, load_data in datasets:
        print(f"\nRunning dataset: {dataset_name}")

        # Load and sample data.
        X, y = load_data()
        if len(y) > args.sample_size:
            _, X, _, y = train_test_split(
                X,
                y,
                test_size=args.sample_size,
                stratify=y,
                random_state=RANDOM_STATE,
            )

        X = StandardScaler().fit_transform(X)
        n_clusters = len(np.unique(y))

        for k in args.k:
            if k > X.shape[1]:
                print(f"  k={k} skipped because original dimension is {X.shape[1]}")
                continue

            print(f"  k={k}")

            # Apply dimensionality reduction.
            reduced_data = {}
            for method_name, reduce_function in reduction_methods:
                if method_name == "dct":
                    X_reduced, reduction_time = measure_runtime(reduce_function, X, k)
                else:
                    X_reduced, reduction_time = measure_runtime(
                        reduce_function,
                        X,
                        k,
                        RANDOM_STATE,
                    )

                reduced_data[method_name] = (X_reduced, reduction_time)

            # Run clustering and calculate metrics.
            for method_name, (X_method, reduction_time) in reduced_data.items():
                distance_corr = distance_preservation(X, X_method)

                kmeans_result, kmeans_time = measure_runtime(
                    kmeans,
                    X_method,
                    n_clusters=n_clusters,
                    random_state=RANDOM_STATE,
                )
                gmm_result, gmm_time = measure_runtime(
                    gmm,
                    X_method,
                    n_components=n_clusters,
                    random_state=RANDOM_STATE,
                )

                for model_name, labels, clustering_time in [
                    ("kmeans", kmeans_result[0], kmeans_time),
                    ("gmm", gmm_result[0], gmm_time),
                ]:
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "method": method_name,
                            "model": model_name,
                            "k": k,
                            "ari": ari(y, labels),
                            "nmi": nmi(y, labels),
                            "purity": purity(y, labels),
                            "silhouette": silhouette(X_method, labels),
                            "distance_preservation": distance_corr,
                            "reduction_time": reduction_time,
                            "clustering_time": clustering_time,
                            "total_time": reduction_time + clustering_time,
                        }
                    )

    results = pd.DataFrame(rows)
    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False)

    print(f"\nSaved CSV: {csv_path}")
