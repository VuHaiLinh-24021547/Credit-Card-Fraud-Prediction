"""Shared PyTorch MLP and class-imbalance loss experimentation pipeline for Member 3."""

import copy
import math
import os
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

NN_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(NN_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_pipeline.nn_data_pipeline import evaluate_auprc


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class FraudMLP(nn.Module):
    """Controlled 4-layer tabular MLP: Linear -> BatchNorm1d -> ReLU -> Dropout."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: tuple[int, ...] = (256, 128, 64, 32),
        dropouts: tuple[float, ...] = (0.30, 0.25, 0.20, 0.10),
        init_pos_prob: float | None = None,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != len(dropouts):
            raise ValueError("hidden_dims and dropouts must have equal length.")

        layers: list[nn.Module] = []
        prev_dim = in_features
        for h_dim, drop in zip(hidden_dims, dropouts):
            lin = nn.Linear(prev_dim, h_dim)
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            nn.init.zeros_(lin.bias)
            layers.extend([
                lin,
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(p=drop),
            ])
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)
        nn.init.xavier_normal_(self.head.weight)
        if init_pos_prob is not None and 0.0 < init_pos_prob < 1.0:
            nn.init.constant_(self.head.bias, math.log(init_pos_prob / (1.0 - init_pos_prob)))
        else:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


class BinaryFocalLoss(nn.Module):
    """Logit-based Binary Focal Loss for imbalanced binary classification."""

    def __init__(self, alpha: float | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        loss = ((1.0 - p_t) ** self.gamma) * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        return loss.mean()


def make_loss(
    loss_type: str,
    y_train: np.ndarray,
    device: torch.device,
    pos_weight: float | str | None = None,
    alpha: float | None = None,
    gamma: float = 2.0,
) -> tuple[nn.Module, float | None, float]:
    """Build loss function strictly from training fold targets (zero data leakage)."""
    n_pos = float(np.sum(y_train == 1))
    n_neg = float(np.sum(y_train == 0))
    ir = n_neg / max(n_pos, 1.0)
    prior = n_pos / max(n_pos + n_neg, 1.0)

    if loss_type == "bce":
        if pos_weight is None or pos_weight == 1.0:
            return nn.BCEWithLogitsLoss(), prior, 1.0
        if pos_weight == "sqrt":
            eff_pw = math.sqrt(ir)
        elif pos_weight == "balanced":
            eff_pw = ir
        else:
            eff_pw = float(pos_weight)
        pw_tensor = torch.tensor([eff_pw], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pw_tensor), None, float(eff_pw)

    if loss_type == "focal":
        return BinaryFocalLoss(alpha=alpha, gamma=gamma), prior, 1.0

    raise ValueError(f"Unsupported loss_type: {loss_type}")


@torch.no_grad()
def predict_proba_tensor(
    model: nn.Module,
    X_tensor: torch.Tensor,
    batch_size: int = 16384,
) -> np.ndarray:
    """Compute sigmoid probabilities in batches."""
    model.eval()
    n = X_tensor.shape[0]
    preds = np.empty(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        logits = model(X_tensor[start:end])
        preds[start:end] = torch.sigmoid(logits).detach().cpu().numpy()
    return preds


def train_mlp_fold(
    fold_data: dict[str, Any],
    loss_cfg: dict[str, Any],
    seed: int = 42,
    hidden_dims: tuple[int, ...] = (256, 128, 64, 32),
    dropouts: tuple[float, ...] = (0.30, 0.25, 0.20, 0.10),
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 4096,
    max_epochs: int = 22,
    patience: int = 5,
    clip_val: float = 20.0,
    verbose: bool = False,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Train FraudMLP on one fold with deterministic outlier clipping and AUPRC early stopping."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr = np.clip(fold_data["X_train"], -clip_val, clip_val).astype(np.float32)
    X_va = np.clip(fold_data["X_val"], -clip_val, clip_val).astype(np.float32)
    y_tr = fold_data["y_train"].astype(np.float32)
    y_va = fold_data["y_val"].astype(np.int64)

    criterion, init_prior, eff_pw = make_loss(
        loss_type=loss_cfg["loss_type"],
        y_train=fold_data["y_train"],
        device=device,
        pos_weight=loss_cfg.get("pos_weight_mode"),
        alpha=loss_cfg.get("alpha"),
        gamma=loss_cfg.get("gamma", 2.0),
    )

    model = FraudMLP(
        in_features=X_tr.shape[1],
        hidden_dims=hidden_dims,
        dropouts=dropouts,
        init_pos_prob=init_prior,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_va_t = torch.from_numpy(X_va).to(device)

    n_train = X_tr_t.shape[0]
    best_auprc = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    history: dict[str, Any] = {"train_loss": [], "val_auprc": [], "effective_pos_weight": eff_pw}

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        seen = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            if end - start < 2:
                continue
            idx = perm[start:end]
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_tr_t[idx])
            loss = criterion(logits, y_tr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = end - start
            epoch_loss += float(loss.item()) * bs
            seen += bs

        avg_loss = epoch_loss / max(seen, 1)
        val_probs = predict_proba_tensor(model, X_va_t)
        val_auprc = evaluate_auprc(y_va, val_probs)
        scheduler.step(val_auprc)

        history["train_loss"].append(avg_loss)
        history["val_auprc"].append(val_auprc)

        if val_auprc > best_auprc + 1e-6:
            best_auprc = val_auprc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch == 1 or epoch % 5 == 0 or no_improve == 0):
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:02d}/{max_epochs} | loss={avg_loss:.5f} "
                f"| val_auprc={val_auprc:.5f} | lr={cur_lr:.2e}"
            )

        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    val_probs = predict_proba_tensor(model, X_va_t)
    history["best_epoch"] = best_epoch
    return val_probs, float(best_auprc), history


def evaluate_binary_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    """Compute AUPRC, ROC-AUC, F1@0.50, optimal F1 threshold, and confusion counts."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_proba, dtype=np.float64)

    auprc = evaluate_auprc(y_true_arr, y_prob_arr)
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    prec, rec, thresholds = precision_recall_curve(y_true_arr, y_prob_arr)
    f1_scores = np.divide(
        2.0 * prec * rec,
        prec + rec,
        out=np.zeros_like(prec),
        where=(prec + rec) != 0,
    )
    best_idx = int(np.argmax(f1_scores))
    best_th = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])

    y_pred_best = (y_prob_arr >= best_th).astype(np.int64)
    f1_default = float(f1_score(y_true_arr, (y_prob_arr >= 0.5).astype(np.int64), zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_best).ravel()

    return {
        "auprc": auprc,
        "roc_auc": roc_auc,
        "f1_at_050": f1_default,
        "best_f1": best_f1,
        "best_threshold": best_th,
        "precision_at_best_f1": float(precision_score(y_true_arr, y_pred_best, zero_division=0)),
        "recall_at_best_f1": float(recall_score(y_true_arr, y_pred_best, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
