from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy import sparse

from .metrics import classification_metrics
from .profiling import TimerMemory
from dataclasses import replace


def feature_memory_mb(X) -> float:
    if sparse.issparse(X):
        return float((X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / (1024 * 1024))
    arr = np.asarray(X)
    return float(arr.nbytes / (1024 * 1024))


@dataclass
class EvaluationResult:
    record: Dict[str, Any]


class ExperimentEvaluator:
    """Common evaluation class used by all models and reducers."""

    def force_matrix_type_for_srp(self, X, force_type):
        if force_type is None:
            return X

        print(f"Forcing matrix type to {force_type}")
        if force_type == "dense":
            if sparse.issparse(X):
                print("Converting sparse matrix to dense array...")
                return X.toarray()
            print("Matrix is already dense, returning as is...")
            return np.asarray(X)

        if force_type == "sparse":
            if sparse.issparse(X):
                print("Matrix is already sparse, returning as is...")
                return X.tocsr()
            print("Converting dense array to sparse matrix...")
            return sparse.csr_matrix(X)

        raise ValueError(f"Unknown force_type: {force_type}")

    def evaluate(self, *, dataset, reducer, model, run_id: int, seed: int, reducer_name: str, model_name: str, force_srp_matrix_type: Optional[str] = None):
        recorded_force_srp_matrix_type = force_srp_matrix_type if reducer_name == "sparse_rp" else None
        base = {
            "run_id": run_id,
            "seed": seed,
            "dataset": dataset.name,
            "feature_kind": dataset.feature_kind,
            "model": model_name,
            "reducer": reducer_name,
            "force_srp_matrix_type": recorded_force_srp_matrix_type,
            "requested_dim": reducer.n_components if reducer.n_components is not None else "original",
            "input_dim": dataset.input_dim,
            "n_train": int(dataset.X_train.shape[0]),
            "n_test": int(dataset.X_test.shape[0]),
            "n_classes": int(dataset.n_classes),
        }

        try:
            dataset_for_reducer = None
            if reducer.name == "sparse_rp":
                X_train_for_reduction = self.force_matrix_type_for_srp(
                    dataset.X_train,
                    force_srp_matrix_type,
                )
                X_test_for_reduction = self.force_matrix_type_for_srp(
                    dataset.X_test,
                    force_srp_matrix_type,
                )
                dataset_for_reducer = replace(
                    dataset,
                    X_train=X_train_for_reduction,
                    X_test=X_test_for_reduction,
                )
            else:
                dataset_for_reducer = dataset

            reducer.setup(dataset_for_reducer)

            with TimerMemory() as prof_reduce_fit:
                # Fit the reducer on the training data.
                print(f"Fitting reducer {reducer_name} on training data...")
                reducer.fit(dataset_for_reducer.X_train)
            with TimerMemory() as prof_reduce_train_transform:
                # Transform the training data after fitting.
                print(f"Transforming training data with reducer {reducer_name}...")
                X_train_red = reducer.transform(dataset_for_reducer.X_train)
            with TimerMemory() as prof_reduce_test_transform:
                # Transform the test data using the fitted reducer.
                print(f"Transforming test data with reducer {reducer_name}...")
                X_test_red = reducer.transform(dataset_for_reducer.X_test)
            output_dim = int(X_train_red.shape[1])
            # Warm-up for DNN to avoid measuring one-time import/initialization costs.
            # if model_name == "dnn":
            #     try:
            #         print("Performing DNN warm-up...")
            #         torch, nn, DataLoader, TensorDataset = model._ensure_torch()
            #         # seed and small forward to trigger CUDA / cuDNN and module init
            #         model._set_seed(torch)
            #         device = torch.device(model.config.device if torch.cuda.is_available() or model.config.device == "cpu" else "cpu")
            #         with torch.no_grad():
            #             dummy = torch.zeros((1, 1), dtype=torch.float32).to(device)
            #             tiny_hidden = max(2, min(32, model.config.hidden_dim))
            #             tiny = nn.Sequential(nn.Linear(1, tiny_hidden), nn.ReLU(), nn.Linear(tiny_hidden, model.n_classes)).to(device)
            #             _ = tiny(dummy)
            #             del tiny
            #             if torch.cuda.is_available():
            #                 try:
            #                     torch.cuda.synchronize(device)
            #                     torch.cuda.empty_cache()
            #                 except Exception:
            #                     pass
            #         print("DNN warm-up completed.")
            #     except Exception:
            #         # Warm-up best-effort; ignore failures and proceed.
            #         pass

            with TimerMemory() as prof_fit:
                # Fit the model to the reduced training data
                print(f"Fitting model {model_name} on reduced training data...")
                model.fit(X_train_red, dataset_for_reducer.y_train)
            with TimerMemory() as prof_pred:
                # Make predictions on the reduced test data
                print(f"Making predictions with model {model_name} on reduced test data...")
                y_pred = model.predict(X_test_red)

            metrics = classification_metrics(dataset_for_reducer.y_test, y_pred)

            rec = dict(base)
            rec.update({
                "status": "ok",
                "output_dim": output_dim,
                "x_train_memory_mb": feature_memory_mb(X_train_red),
                "x_test_memory_mb": feature_memory_mb(X_test_red),
                "reduction_fit_time_sec": prof_reduce_fit.result.elapsed_sec,
                "reduction_train_transform_time_sec": prof_reduce_train_transform.result.elapsed_sec,
                "reduction_test_transform_time_sec": prof_reduce_test_transform.result.elapsed_sec,
                "reduction_fit_peak_py_mb": prof_reduce_fit.result.python_peak_mb,
                "reduction_train_transform_peak_py_mb": prof_reduce_train_transform.result.python_peak_mb,
                "reduction_test_transform_peak_py_mb": prof_reduce_test_transform.result.python_peak_mb,
                "model_fit_time_sec": prof_fit.result.elapsed_sec,
                "model_predict_time_sec": prof_pred.result.elapsed_sec,
                "model_fit_peak_py_mb": prof_fit.result.python_peak_mb,
                "model_predict_peak_py_mb": prof_pred.result.python_peak_mb,
                "reducer_params": reducer.params(),
                "model_params": model.params(),
            })
            rec.update(metrics)
            return EvaluationResult(rec)

        except Exception as e:
            rec = dict(base)
            rec.update({
                "status": "failed",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(limit=3),
            })
            return EvaluationResult(rec)
