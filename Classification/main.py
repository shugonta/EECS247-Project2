from __future__ import annotations

import argparse
import os
from typing import List

from Classification.datasets.loaders import load_dataset
from Classification.reducers.factory import make_reducer
from Classification.models.factory import make_model
from Classification.utils.evaluator import ExperimentEvaluator
from Classification.utils.io import append_records_csv


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RP/PCA/DCT/SVD reducers with kNN/SVM/DNN classifiers.")

    p.add_argument("--datasets", nargs="+", default=["fashion_mnist"],
                   choices=["fashion_mnist", "cifar10", "uci_har", "rcv1", "20newsgroups"])
    p.add_argument("--reducers", nargs="+", default=["identity", "sparse_rp"],
                   choices=["identity", "sparse_rp", "pca_svd", "dct", "randomized_svd", "gaussian_rp"])
    p.add_argument("--models", nargs="+", default=["knn"], choices=["knn", "svm", "svm_sgd", "dnn"])
    p.add_argument("--dims", nargs="+", type=int, default=[100],
                   help="Reduced dimensions k. Ignored by identity reducer.")
    p.add_argument("--runs", type=int, default=1,
                   help="Number of repeated runs. Each run is written as a separate CSV record.")
    p.add_argument("--discard-first-run", action="store_true",
                   help="Run one extra iteration and discard the first run to avoid first-run overhead.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--output", type=str, default="results/results.csv")

    p.add_argument("--force-srp-matrix-type", type=str, default=None, choices=["dense", "sparse"],
                   help="Force SparseRandomProjection to output a dense or sparse matrix.")

    p.add_argument("--max-train", type=int, default=None,
                   help="Optional training subset size. Recommended for CIFAR-10/RCV1/kNN.")
    p.add_argument("--max-test", type=int, default=None,
                   help="Optional test subset size. Recommended for CIFAR-10/RCV1/kNN.")
    p.add_argument("--no-standardize", action="store_true")

    # Reducer options
    p.add_argument("--allow-dense-sparse-pca", action="store_true",
                   help="Allow full PCA to densify sparse datasets. Dangerous for RCV1.")
    p.add_argument("--dct-mode", type=str, default="auto", choices=["auto", "image2d", "1d"])
    p.add_argument("--randomized-svd-iter", type=int, default=5)

    # Model options
    p.add_argument("--knn-neighbors", type=int, default=5)
    p.add_argument("--knn-metric", type=str, default="auto",
                   help="auto, euclidean, cosine, manhattan, etc.")
    p.add_argument("--n-jobs", type=int, default=-1)

    p.add_argument("--svm-c", type=float, default=1.0)
    p.add_argument("--svm-max-iter", type=int, default=5000)
    p.add_argument("--svm-scale-dense", action="store_true")
    p.add_argument("--svm-alpha", type=float, default=1e-4)

    p.add_argument("--dnn-hidden-dim", type=int, default=128)
    p.add_argument("--dnn-epochs", type=int, default=20)
    p.add_argument("--dnn-batch-size", type=int, default=256)
    p.add_argument("--dnn-lr", type=float, default=1e-3)
    p.add_argument("--dnn-dropout", type=float, default=0.2)
    p.add_argument("--dnn-early-stopping", type=bool, default=True)
    p.add_argument("--dnn-validation-fraction", type=float, default=0.1)
    p.add_argument("--dnn-patience", type=int, default=3)
    p.add_argument("--dnn-min-delta", type=float, default=0.0)
    p.add_argument("--dnn-weight-decay", type=float, default=0.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--allow-dnn-sparse-to-dense", action="store_true")

    return p.parse_args()


def reducer_dims(reducer_name: str, dims: List[int]):
    if reducer_name == "identity":
        return [None]
    return dims


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    evaluator = ExperimentEvaluator()

    total_runs = args.runs + (1 if args.discard_first_run else 0)
    for run_idx in range(total_runs):
        is_discard = args.discard_first_run and run_idx == 0
        # reported run id for records (shifts down by one when discarding first)
        reported_run_id = run_idx - 1 if args.discard_first_run else run_idx
        seed = args.seed + reported_run_id if not is_discard else args.seed
        for dataset_name in args.datasets:
            if is_discard:
                print(f"\n=== Warm-up (discard) run for dataset={dataset_name}, seed={seed} ===")
            else:
                print(f"\n=== Loading dataset={dataset_name}, run={reported_run_id}, seed={seed} ===")
            dataset = load_dataset(
                dataset_name,
                data_dir=args.data_dir,
                seed=seed,
                max_train=args.max_train,
                max_test=args.max_test,
                standardize=not args.no_standardize,
            )
            print(f"Loaded {dataset.name}: train={dataset.X_train.shape}, test={dataset.X_test.shape}, classes={dataset.n_classes}")

            records = []
            for reducer_name in args.reducers:
                for k in reducer_dims(reducer_name, args.dims):
                    # Skip impossible dimensions early for dense methods where possible.
                    if k is not None and k > dataset.input_dim:
                        print(f"Skipping reducer={reducer_name}, k={k}: k > input_dim={dataset.input_dim}")
                        continue
                    for model_name in args.models:
                        print(f"Running reducer={reducer_name}, k={k}, model={model_name}")
                        reducer = make_reducer(reducer_name, k, seed, args)
                        model = make_model(model_name, dataset.n_classes, seed, args)
                        result = evaluator.evaluate(
                            dataset=dataset,
                            reducer=reducer,
                            model=model,
                            run_id=reported_run_id,
                            seed=seed,
                            reducer_name=reducer_name,
                            model_name=model_name,
                            force_srp_matrix_type=args.force_srp_matrix_type
                        )
                        if not is_discard:
                            records.append(result.record)
                        status = result.record.get("status")
                        acc = result.record.get("accuracy", "")
                        pred_time = result.record.get("model_predict_time_sec", "")
                        # Still print status for visibility; for warm-up runs these are informational only.
                        print(f"  -> status={status}, acc={acc}, pred_time={pred_time}")

            append_records_csv(args.output, records)
            print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
