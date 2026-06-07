from __future__ import annotations

from .knn import KNNClassifier
from .svm import SVMClassifier, SGDHingeSVMClassifier
from .dnn import DNNClassifier, DNNConfig


def make_model(name: str, n_classes: int, seed: int, args):
    name = name.lower()
    if name == "knn":
        return KNNClassifier(n_neighbors=args.knn_neighbors, metric=args.knn_metric, n_jobs=args.n_jobs)
    if name == "svm":
        return SVMClassifier(C=args.svm_c, max_iter=args.svm_max_iter, random_state=seed, scale_dense=args.svm_scale_dense)
    if name == "svm_sgd":
        return SGDHingeSVMClassifier(max_iter=args.svm_max_iter, alpha=args.svm_alpha, random_state=seed)
    if name == "dnn":
        cfg = DNNConfig(
            hidden_dim=args.dnn_hidden_dim,
            epochs=args.dnn_epochs,
            batch_size=args.dnn_batch_size,
            lr=args.dnn_lr,
            dropout=args.dnn_dropout,
            device=args.device,
            random_state=seed,
            allow_sparse_to_dense=args.allow_dnn_sparse_to_dense,
            early_stopping=args.dnn_early_stopping,
            validation_fraction=args.dnn_validation_fraction,
            patience=args.dnn_patience,
            min_delta=args.dnn_min_delta,
            weight_decay=args.dnn_weight_decay,
        )
        return DNNClassifier(n_classes=n_classes, config=cfg)
    raise ValueError(f"Unknown model: {name}")
