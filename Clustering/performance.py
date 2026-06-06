import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METHODS = ["pca", "rp", "srp", "dct"]


def save_metric_vs_k_plots(results, output_root):
    output_dir = output_root / "metric_vs_k"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("ari", "ARI"),
        ("nmi", "NMI"),
        ("purity", "Cluster Purity"),
        ("silhouette", "Silhouette Score"),
        ("distance_preservation", "Distance Preservation"),
    ]

    for dataset_name in sorted(results["dataset"].unique()):
        for model_name in sorted(results["model"].unique()):
            df = results[
                (results["dataset"] == dataset_name)
                & (results["model"] == model_name)
            ]

            for metric_key, metric_label in metrics:
                plt.figure(figsize=(8, 5))
                for method_name in METHODS:
                    method_df = df[df["method"] == method_name].sort_values("k")
                    plt.plot(
                        method_df["k"],
                        method_df[metric_key],
                        marker="o",
                        linewidth=2,
                        label=method_name.upper(),
                    )

                plt.xlabel("Reduced dimension k")
                plt.ylabel(metric_label)
                plt.title(f"{dataset_name} | {model_name} | {metric_label} vs k")
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_dir / f"{dataset_name}_{model_name}_{metric_key}.png")
                plt.close()


def save_runtime_plots(results, output_root):
    output_dir = output_root / "runtime_vs_k"
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in sorted(results["dataset"].unique()):
        for model_name in sorted(results["model"].unique()):
            df = results[
                (results["dataset"] == dataset_name)
                & (results["model"] == model_name)
            ]

            plt.figure(figsize=(8, 5))
            for method_name in METHODS:
                method_df = df[df["method"] == method_name].sort_values("k")
                plt.plot(
                    method_df["k"],
                    method_df["total_time"],
                    marker="o",
                    linewidth=2,
                    label=method_name.upper(),
                )

            plt.xlabel("Reduced dimension k")
            plt.ylabel("Total time (sec)")
            plt.title(f"{dataset_name} | {model_name} | Runtime vs k")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"{dataset_name}_{model_name}_runtime.png")
            plt.close()


def save_runtime_breakdown_plots(results, output_root):
    output_dir = output_root / "runtime_breakdown"
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in sorted(results["dataset"].unique()):
        for model_name in sorted(results["model"].unique()):
            df = results[
                (results["dataset"] == dataset_name)
                & (results["model"] == model_name)
            ]

            for k in sorted(df["k"].unique()):
                k_df = df[df["k"] == k].set_index("method").reindex(METHODS)
                k_df = k_df.dropna(subset=["reduction_time", "clustering_time"])
                if k_df.empty:
                    continue

                plt.figure(figsize=(8, 5))
                x = range(len(k_df))
                plt.bar(x, k_df["reduction_time"], label="Reduction time")
                plt.bar(
                    x,
                    k_df["clustering_time"],
                    bottom=k_df["reduction_time"],
                    label="Clustering time",
                )
                plt.xticks(x, [method.upper() for method in k_df.index])
                plt.ylabel("Time (sec)")
                plt.title(f"{dataset_name} | {model_name} | Runtime breakdown (k={k})")
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_dir / f"{dataset_name}_{model_name}_runtime_breakdown_k{k}.png")
                plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create CSV-based performance plots.")
    parser.add_argument("--csv", type=str, default=str(Path("outputs") / "results.csv"))
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_root = Path(args.output_dir) if args.output_dir else csv_path.parent

    results = pd.read_csv(csv_path)
    save_metric_vs_k_plots(results, output_root)
    save_runtime_plots(results, output_root)
    save_runtime_breakdown_plots(results, output_root)

    print(f"Saved CSV-based plots to {output_root}")
