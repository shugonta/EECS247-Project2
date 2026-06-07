import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from clustering import gmm, kmeans
from dim_reduction import dct_reduce, pca, rp, srp
from feature_extraction import load_cifar10_cnn, load_fashion_mnist_hog
from Classification.main import load_cifar10, load_fashion_mnist, load_uci_har


RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
METHODS = ["pca", "rp", "srp", "dct"]


def save_visualization_plots(dataset_name, X_raw, X, y, selected_k, output_root):
    distortion_dir = output_root / "distance_distortion"
    umap_dir = output_root / "umap_cluster_grid"
    heatmap_dir = output_root / "cluster_heatmaps"
    gallery_dir = output_root / "cluster_galleries"

    for output_dir in [
        distortion_dir,
        umap_dir,
        heatmap_dir,
        gallery_dir,
    ]:
        output_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = len(np.unique(y))
    if selected_k > X.shape[1]:
        selected_k = X.shape[1]

    features = {
        "original": X,
        "pca": pca(X, selected_k, RANDOM_STATE),
        "rp": rp(X, selected_k, RANDOM_STATE),
        "srp": srp(X, selected_k, RANDOM_STATE),
        "dct": dct_reduce(X, selected_k),
    }

    labels = {}
    for method_name, X_method in features.items():
        labels[(method_name, "kmeans")] = kmeans(
            X_method,
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
        )[0]
        labels[(method_name, "gmm")] = gmm(
            X_method,
            n_components=n_clusters,
            random_state=RANDOM_STATE,
        )[0]

    # Distance distortion: reduced_distance / original_distance.
    rng = np.random.default_rng(RANDOM_STATE)
    distance_n = min(600, X.shape[0])
    distance_indices = rng.choice(X.shape[0], distance_n, replace=False)
    original_dist = pairwise_distances(X[distance_indices])
    original_vec = original_dist[np.triu_indices_from(original_dist, k=1)]
    nonzero_mask = original_vec > 1e-12
    original_vec = original_vec[nonzero_mask]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    for ax, method_name in zip(axes, METHODS):
        reduced_dist = pairwise_distances(features[method_name][distance_indices])
        reduced_vec = reduced_dist[np.triu_indices_from(reduced_dist, k=1)]
        distortion = reduced_vec[nonzero_mask] / original_vec

        ax.hist(distortion, bins=50, color="#4C78A8", alpha=0.85)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(method_name.upper())
        ax.set_xlabel("reduced distance / original distance")
        ax.set_ylabel("Count")

    fig.suptitle(f"{dataset_name} | Distance Distortion (k={selected_k})")
    fig.tight_layout()
    fig.savefig(distortion_dir / f"{dataset_name}_distortion_k{selected_k}.png")
    plt.close(fig)

    # UMAP grid: true labels, KMeans labels, and GMM labels.
    method_names = ["original"] + METHODS
    umap_n = min(1500, len(y))
    umap_indices = rng.choice(len(y), umap_n, replace=False)
    fig, axes = plt.subplots(5, 3, figsize=(12, 17))

    for row_idx, method_name in enumerate(method_names):
        embedding = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=RANDOM_STATE,
        ).fit_transform(features[method_name][umap_indices])

        plot_labels = [
            ("True Labels", y[umap_indices]),
            ("KMeans", labels[(method_name, "kmeans")][umap_indices]),
            ("GMM", labels[(method_name, "gmm")][umap_indices]),
        ]

        for col_idx, (title, plot_y) in enumerate(plot_labels):
            ax = axes[row_idx, col_idx]
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=plot_y,
                s=7,
                cmap="tab10",
                alpha=0.75,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{method_name.upper()} | {title}")

    fig.suptitle(f"{dataset_name} | UMAP Cluster Grid (k={selected_k})")
    fig.tight_layout()
    fig.savefig(umap_dir / f"{dataset_name}_umap_grid_k{selected_k}.png")
    plt.close(fig)

    # Cluster-class heatmaps.
    for method_name in method_names:
        for model_name in ["kmeans", "gmm"]:
            table = pd.crosstab(labels[(method_name, model_name)], y)

            plt.figure(figsize=(8, 6))
            plt.imshow(table.values, aspect="auto", cmap="Blues")
            plt.colorbar(label="Count")
            plt.xlabel("True class")
            plt.ylabel("Predicted cluster")
            plt.title(f"{dataset_name} | {method_name.upper()} | {model_name}")
            plt.tight_layout()
            plt.savefig(heatmap_dir / f"{dataset_name}_{method_name}_{model_name}_heatmap.png")
            plt.close()

    # Image galleries for image datasets.
    image_dataset = None
    if dataset_name in {"fashion_mnist", "fashion_mnist_hog"}:
        image_dataset = "fashion_mnist"
    elif dataset_name in {"cifar10", "cifar10_cnn"}:
        image_dataset = "cifar10"

    if image_dataset is None:
        return

    for method_name in method_names:
        for model_name in ["kmeans", "gmm"]:
            cluster_labels = labels[(method_name, model_name)]
            cluster_ids = sorted(np.unique(cluster_labels))[:10]
            fig, axes = plt.subplots(
                len(cluster_ids),
                8,
                figsize=(12, 1.7 * len(cluster_ids)),
            )

            if len(cluster_ids) == 1:
                axes = np.array([axes])

            for row_idx, cluster_id in enumerate(cluster_ids):
                cluster_indices = np.where(cluster_labels == cluster_id)[0][:8]
                for col_idx in range(8):
                    ax = axes[row_idx, col_idx]
                    ax.axis("off")
                    if col_idx >= len(cluster_indices):
                        continue

                    sample_idx = cluster_indices[col_idx]
                    image = X_raw[sample_idx]
                    if image_dataset == "fashion_mnist":
                        ax.imshow(image.reshape(28, 28), cmap="gray")
                    elif image_dataset == "cifar10":
                        ax.imshow(image.reshape(3, 32, 32).transpose(1, 2, 0))

                    ax.set_title(f"y={y[sample_idx]}", fontsize=8)
                    if col_idx == 0:
                        ax.set_ylabel(f"Cluster {cluster_id}", fontsize=9)

            fig.suptitle(f"{dataset_name} | {method_name.upper()} | {model_name} samples")
            fig.tight_layout()
            fig.savefig(gallery_dir / f"{dataset_name}_{method_name}_{model_name}_gallery.png")
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create original-data-based clustering visuals.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["fashion_mnist", "fashion_mnist_hog", "cifar10", "cifar10_cnn", "uci_har"],
        choices=["fashion_mnist", "fashion_mnist_hog", "cifar10", "cifar10_cnn", "uci_har"],
    )
    parser.add_argument("--selected_k", type=int, default=100)
    parser.add_argument("--sample_size", type=int, default=5000)
    args = parser.parse_args()

    datasets = [
        ("fashion_mnist", load_fashion_mnist),
        ("fashion_mnist_hog", lambda: load_fashion_mnist_hog(Path(__file__).resolve().parent / "data", load_fashion_mnist)),
        ("cifar10", load_cifar10),
        ("cifar10_cnn", lambda: load_cifar10_cnn(Path(__file__).resolve().parent / "data")),
        ("uci_har", load_uci_har),
    ]
    datasets = [(name, loader) for name, loader in datasets if name in args.datasets]

    for dataset_name, load_data in datasets:
        print(f"\nVisualizing dataset: {dataset_name}")

        X, y = load_data()
        X_display = X
        if dataset_name == "fashion_mnist_hog":
            X_display, _ = load_fashion_mnist()
        elif dataset_name == "cifar10_cnn":
            X_display, _ = load_cifar10()

        if len(y) > args.sample_size:
            indices = np.arange(len(y))
            _, selected_indices, _, _ = train_test_split(
                indices,
                y,
                test_size=args.sample_size,
                stratify=y,
                random_state=RANDOM_STATE,
            )
            X = X[selected_indices]
            X_display = X_display[selected_indices]
            y = y[selected_indices]

        X_raw = X_display.copy()
        X = StandardScaler().fit_transform(X)
        save_visualization_plots(dataset_name, X_raw, X, y, args.selected_k, OUTPUT_DIR)

    print(f"\nSaved visualization plots to {OUTPUT_DIR}")
