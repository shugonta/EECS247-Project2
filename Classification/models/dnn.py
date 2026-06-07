from __future__ import annotations

import random
from dataclasses import dataclass
import time

import numpy as np
from scipy import sparse
import math

from .base import BaseClassifier, ModelError


@dataclass
class DNNConfig:
    hidden_dim: int = 256
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    dropout: float = 0.2
    device: str = "cpu"
    random_state: int = 42
    allow_sparse_to_dense: bool = False

    # Early stopping / regularization options
    early_stopping: bool = False
    validation_fraction: float = 0.1
    patience: int = 3
    min_delta: float = 0.0
    weight_decay: float = 0.0


class DNNClassifier(BaseClassifier):
    name = "dnn"

    def __init__(self, n_classes: int, config: DNNConfig):
        self.n_classes = n_classes
        self.config = config
        self.model = None
        self.input_dim = None
        self.num_parameters = None
        self.best_epoch = None
        self.best_val_acc = None
        self.best_val_loss = None

    def _ensure_torch(self):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as e:
            raise ImportError("DNN requires PyTorch. Install with: pip install torch") from e
        return torch, nn, DataLoader, TensorDataset

    # SparseLinear and Sparse-capable DNN are created dynamically inside fit

    def _to_dense_np(self, X):
        if sparse.issparse(X):
            if not self.config.allow_sparse_to_dense:
                raise ModelError(
                    "DNN received sparse input. Use --allow-dnn-sparse-to-dense "
                    "or skip DNN for sparse datasets."
                )
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def _make_train_val_indices(self, y_np: np.ndarray):
        """Create a stratified validation split for early stopping.

        If early stopping is disabled, returns all indices for training and None for validation.
        """
        n = len(y_np)
        all_indices = np.arange(n)

        if not self.config.early_stopping or self.config.validation_fraction <= 0.0:
            return all_indices, None

        rng = np.random.default_rng(self.config.random_state)
        val_indices = []
        train_indices = []

        for cls in np.unique(y_np):
            cls_idx = np.flatnonzero(y_np == cls)
            rng.shuffle(cls_idx)

            if len(cls_idx) < 2:
                # Cannot split this class safely; keep it in training.
                train_indices.extend(cls_idx.tolist())
                continue

            n_val = int(round(len(cls_idx) * self.config.validation_fraction))
            n_val = max(1, n_val)
            n_val = min(n_val, len(cls_idx) - 1)

            val_indices.extend(cls_idx[:n_val].tolist())
            train_indices.extend(cls_idx[n_val:].tolist())

        if len(val_indices) == 0:
            return all_indices, None

        train_indices = np.asarray(train_indices, dtype=np.int64)
        val_indices = np.asarray(val_indices, dtype=np.int64)
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        return train_indices, val_indices

    def _make_sparse_tensor(self, X_batch, torch, device, n_features: int):
        bsize = X_batch.shape[0]
        coo = X_batch.tocoo()
        if coo.nnz == 0:
            indices = torch.empty((2, 0), dtype=torch.long)
            values = torch.empty((0,), dtype=torch.float32)
        else:
            indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
            values = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(
            indices,
            values,
            (bsize, n_features),
        ).coalesce().to(device)

    def _evaluate_sparse(self, X, y_np, indices, criterion, torch, device):
        if indices is None or len(indices) == 0:
            return None, None

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        batch_size = self.config.batch_size

        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                batch_csr = X[batch_idx]
                x_sparse = self._make_sparse_tensor(batch_csr, torch, device, X.shape[1])
                yb = torch.from_numpy(y_np[batch_idx]).to(device)

                logits = self.model(x_sparse)
                loss = criterion(logits, yb)

                bsize = len(batch_idx)
                total_loss += loss.item() * bsize
                total_correct += (torch.argmax(logits, dim=1) == yb).sum().item()
                total_samples += bsize

        return (
            total_loss / max(total_samples, 1),
            total_correct / max(total_samples, 1),
        )

    def _evaluate_dense_loader(self, loader, criterion, torch, device):
        if loader is None:
            return None, None

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = self.model(xb)
                loss = criterion(logits, yb)

                bsize = yb.size(0)
                total_loss += loss.item() * bsize
                total_correct += (torch.argmax(logits, dim=1) == yb).sum().item()
                total_samples += bsize

        return (
            total_loss / max(total_samples, 1),
            total_correct / max(total_samples, 1),
        )

    def _save_best_state_if_needed(self, torch, epoch, val_loss, val_acc, best_score, epochs_without_improvement):
        if val_acc is None:
            return best_score, epochs_without_improvement, None

        improved = val_acc > best_score + self.config.min_delta
        if improved:
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            }
            self.best_epoch = epoch
            self.best_val_acc = float(val_acc)
            self.best_val_loss = float(val_loss)
            return float(val_acc), 0, best_state

        return best_score, epochs_without_improvement + 1, None

    def fit(self, X, y):
        torch, nn, DataLoader, TensorDataset = self._ensure_torch()
        self._set_seed(torch)
        y_np = np.asarray(y, dtype=np.int64)
        train_idx, val_idx = self._make_train_val_indices(y_np)

        # Sparse training path: build a SparseLinear first layer and remaining dense layers.
        if sparse.issparse(X) and not self.config.allow_sparse_to_dense:
            class SparseLinear(nn.Module):
                def __init__(self, in_features, out_features, bias=True):
                    super().__init__()
                    self.in_features = in_features
                    self.out_features = out_features
                    self.weight = nn.Parameter(torch.empty((out_features, in_features)))
                    if bias:
                        self.bias = nn.Parameter(torch.empty(out_features))
                    else:
                        self.bias = None
                    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                    if self.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                        bound = 1 / math.sqrt(max(1, fan_in))
                        nn.init.uniform_(self.bias, -bound, bound)

                def forward(self, x_sparse):
                    out = torch.sparse.mm(x_sparse, self.weight.t())
                    if self.bias is not None:
                        out = out + self.bias
                    return out

            class SparseDNN(nn.Module):
                def __init__(self, input_dim, config, n_classes):
                    super().__init__()
                    self.sparse_linear = SparseLinear(input_dim, config.hidden_dim)
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(config.dropout)
                    self.fc2 = nn.Linear(config.hidden_dim, max(32, config.hidden_dim // 2))
                    self.relu2 = nn.ReLU()
                    self.fc3 = nn.Linear(max(32, config.hidden_dim // 2), n_classes)

                def forward(self, x_sparse):
                    x = self.sparse_linear(x_sparse)
                    x = self.relu(x)
                    x = self.dropout(x)
                    x = self.fc2(x)
                    x = self.relu2(x)
                    x = self.fc3(x)
                    return x

            self.input_dim = X.shape[1]
            device = torch.device(
                self.config.device
                if torch.cuda.is_available() or self.config.device == "cpu"
                else "cpu"
            )
            self.model = SparseDNN(self.input_dim, self.config, self.n_classes).to(device)
            self.num_parameters = sum(p.numel() for p in self.model.parameters())

            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
            criterion = nn.CrossEntropyLoss()

            batch_size = self.config.batch_size
            rng = np.random.default_rng(self.config.random_state)
            best_score = -np.inf
            best_state = None
            epochs_without_improvement = 0

            for epoch in range(1, self.config.epochs + 1):
                epoch_start = time.perf_counter()
                total_loss = 0.0
                total_correct = 0
                total_samples = 0

                self.model.train()
                epoch_train_idx = np.asarray(train_idx, dtype=np.int64).copy()
                rng.shuffle(epoch_train_idx)

                for start in range(0, len(epoch_train_idx), batch_size):
                    batch_idx = epoch_train_idx[start:start + batch_size]
                    batch_csr = X[batch_idx]
                    bsize = len(batch_idx)
                    x_sparse = self._make_sparse_tensor(batch_csr, torch, device, X.shape[1])
                    yb = torch.from_numpy(y_np[batch_idx]).to(device)

                    optimizer.zero_grad()
                    logits = self.model(x_sparse)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * bsize
                    total_correct += (torch.argmax(logits, dim=1) == yb).sum().item()
                    total_samples += bsize

                avg_loss = total_loss / max(total_samples, 1)
                train_acc = total_correct / max(total_samples, 1)
                val_loss, val_acc = self._evaluate_sparse(X, y_np, val_idx, criterion, torch, device)
                epoch_time = time.perf_counter() - epoch_start

                log_msg = (
                    f"[DNN sparse] epoch={epoch:03d}/{self.config.epochs} "
                    f"loss={avg_loss:.4f} "
                    f"train_acc={train_acc:.4f} "
                )
                if val_acc is not None:
                    log_msg += f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                log_msg += f"time={epoch_time:.2f}s"
                print(log_msg)

                if self.config.early_stopping and val_acc is not None:
                    best_score_new, epochs_without_improvement, candidate_state = self._save_best_state_if_needed(
                        torch,
                        epoch,
                        val_loss,
                        val_acc,
                        best_score,
                        epochs_without_improvement,
                    )
                    if candidate_state is not None:
                        best_state = candidate_state
                    best_score = best_score_new

                    if epochs_without_improvement >= self.config.patience:
                        print(
                            f"[DNN sparse] early stopping at epoch={epoch} "
                            f"best_epoch={self.best_epoch} best_val_acc={self.best_val_acc:.4f}"
                        )
                        break

            if best_state is not None:
                self.model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            return self

        # Dense path
        X_np = self._to_dense_np(X)
        self.input_dim = X_np.shape[1]

        device = torch.device(
            self.config.device
            if torch.cuda.is_available() or self.config.device == "cpu"
            else "cpu"
        )

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, max(32, self.config.hidden_dim // 2)),
            nn.ReLU(),
            nn.Linear(max(32, self.config.hidden_dim // 2), self.n_classes),
        ).to(device)

        self.num_parameters = sum(p.numel() for p in self.model.parameters())

        train_dataset = TensorDataset(
            torch.from_numpy(X_np[train_idx]),
            torch.from_numpy(y_np[train_idx]),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        if val_idx is not None:
            val_dataset = TensorDataset(
                torch.from_numpy(X_np[val_idx]),
                torch.from_numpy(y_np[val_idx]),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )
        else:
            val_loader = None

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        best_score = -np.inf
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.perf_counter()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                batch_size = yb.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (torch.argmax(logits, dim=1) == yb).sum().item()
                total_samples += batch_size

            avg_loss = total_loss / max(total_samples, 1)
            train_acc = total_correct / max(total_samples, 1)
            val_loss, val_acc = self._evaluate_dense_loader(val_loader, criterion, torch, device)
            epoch_time = time.perf_counter() - epoch_start

            log_msg = (
                f"[DNN] epoch={epoch:03d}/{self.config.epochs} "
                f"loss={avg_loss:.4f} "
                f"train_acc={train_acc:.4f} "
            )
            if val_acc is not None:
                log_msg += f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            log_msg += f"time={epoch_time:.2f}s"
            print(log_msg)

            if self.config.early_stopping and val_acc is not None:
                best_score_new, epochs_without_improvement, candidate_state = self._save_best_state_if_needed(
                    torch,
                    epoch,
                    val_loss,
                    val_acc,
                    best_score,
                    epochs_without_improvement,
                )
                if candidate_state is not None:
                    best_state = candidate_state
                best_score = best_score_new

                if epochs_without_improvement >= self.config.patience:
                    print(
                        f"[DNN] early stopping at epoch={epoch} "
                        f"best_epoch={self.best_epoch} best_val_acc={self.best_val_acc:.4f}"
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        return self

    def predict(self, X):
        torch, _nn, DataLoader, TensorDataset = self._ensure_torch()
        device = next(self.model.parameters()).device
        preds = []
        self.model.eval()
        with torch.no_grad():
            if sparse.issparse(X) and not self.config.allow_sparse_to_dense:
                n = X.shape[0]
                batch_size = self.config.batch_size
                for bs in range(0, n, batch_size):
                    be = min(n, bs + batch_size)
                    batch_csr = X[bs:be]
                    x_sparse = self._make_sparse_tensor(batch_csr, torch, device, X.shape[1])
                    logits = self.model(x_sparse)
                    preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                return np.concatenate(preds) if preds else np.array([], dtype=np.int64)

            # fallback dense path
            X_np = self._to_dense_np(X)
            dataset = TensorDataset(torch.from_numpy(X_np))
            loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
            for (xb,) in loader:
                logits = self.model(xb.to(device))
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            return np.concatenate(preds)

    def _set_seed(self, torch):
        random.seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_state)

    def params(self) -> dict:
        return {
            "hidden_dim": self.config.hidden_dim,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "lr": self.config.lr,
            "dropout": self.config.dropout,
            "device": self.config.device,
            "allow_sparse_to_dense": self.config.allow_sparse_to_dense,
            "early_stopping": self.config.early_stopping,
            "validation_fraction": self.config.validation_fraction,
            "patience": self.config.patience,
            "min_delta": self.config.min_delta,
            "weight_decay": self.config.weight_decay,
            "best_epoch": self.best_epoch,
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "num_parameters": self.num_parameters,
        }
