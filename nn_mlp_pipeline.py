"""Shared PyTorch MLP architecture training and 5-fold bagged inference pipeline for Member 3."""

import copy
from typing import Any

import numpy as np
import torch

from nn_data_pipeline import N_SPLITS, evaluate_auprc, iter_nn_folds
from nn_imbalance_pipeline import (
    FraudMLP,
    evaluate_binary_metrics,
    make_loss,
    predict_proba_tensor,
    set_seed,
)


def train_mlp_fold_with_test(
    fold_data: dict[str, Any],
    loss_cfg: dict[str, Any],
    arch_cfg: dict[str, Any],
    seed: int = 42,
    batch_size: int = 4096,
    max_epochs: int = 25,
    patience: int = 6,
    clip_val: float = 20.0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Train FraudMLP on one fold with outlier clipping, LR scheduling, and AUPRC early stopping."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr = np.clip(fold_data["X_train"], -clip_val, clip_val).astype(np.float32)
    X_va = np.clip(fold_data["X_val"], -clip_val, clip_val).astype(np.float32)
    X_te = np.clip(fold_data["X_test"], -clip_val, clip_val).astype(np.float32)
    y_tr = fold_data["y_train"].astype(np.float32)
    y_va = fold_data["y_val"].astype(np.int64)

    criterion, init_prior, _ = make_loss(
        loss_type=loss_cfg["loss_type"],
        y_train=fold_data["y_train"],
        device=device,
        pos_weight=loss_cfg.get("pos_weight_mode"),
        alpha=loss_cfg.get("alpha"),
        gamma=loss_cfg.get("gamma", 2.0) if loss_cfg.get("gamma") is not None else 2.0,
    )

    model = FraudMLP(
        in_features=X_tr.shape[1],
        hidden_dims=tuple(arch_cfg["hidden_dims"]),
        dropouts=tuple(arch_cfg["dropouts"]),
        init_pos_prob=init_prior,
    ).to(device)

    lr = float(arch_cfg.get("lr", 2e-3))
    weight_decay = float(arch_cfg.get("weight_decay", 1e-4))
    scheduler_type = arch_cfg.get("scheduler_type", "plateau")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=1e-5
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )
    else:
        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_va_t = torch.from_numpy(X_va).to(device)
    X_te_t = torch.from_numpy(X_te).to(device)

    n_train = X_tr_t.shape[0]
    best_auprc = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    history: dict[str, Any] = {"train_loss": [], "val_auprc": []}

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

        if scheduler_type == "cosine":
            scheduler.step()
        else:
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
    test_probs = predict_proba_tensor(model, X_te_t)
    history["best_epoch"] = best_epoch
    return val_probs, test_probs, float(best_auprc), history


def run_bagged_mlp_cv(
    loss_cfg: dict[str, Any],
    arch_cfg: dict[str, Any],
    seeds: tuple[int, ...] = (42, 2024, 2026),
    train_folds_path: str = "train_folds.csv",
    test_path: str = "test.csv",
    batch_size: int = 4096,
    max_epochs: int = 25,
    patience: int = 6,
    clip_val: float = 20.0,
) -> dict[str, Any]:
    """Run 5-fold stratified CV with multi-seed bagging and return aligned OOF and test predictions."""
    oof_seeds: np.ndarray | None = None
    test_seeds: np.ndarray | None = None
    seed_fold_auprcs: dict[int, list[float]] = {int(s): [] for s in seeds}
    best_epochs: dict[int, list[int]] = {int(s): [] for s in seeds}

    for fold_k, fold_data in iter_nn_folds(train_folds_path, test_path):
        val_ids = fold_data["val_ids"]
        if oof_seeds is None:
            n_train = fold_data["X_train"].shape[0] + fold_data["X_val"].shape[0]
            n_test = fold_data["X_test"].shape[0]
            oof_seeds = np.zeros((len(seeds), n_train), dtype=np.float64)
            test_seeds = np.zeros((len(seeds), n_test), dtype=np.float64)

        for s_idx, seed in enumerate(seeds):
            val_p, test_p, best_ap, hist = train_mlp_fold_with_test(
                fold_data=fold_data,
                loss_cfg=loss_cfg,
                arch_cfg=arch_cfg,
                seed=int(seed) + fold_k,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                clip_val=clip_val,
                verbose=False,
            )
            oof_seeds[s_idx, val_ids] = val_p
            test_seeds[s_idx] += test_p / N_SPLITS
            seed_fold_auprcs[int(seed)].append(best_ap)
            best_epochs[int(seed)].append(int(hist["best_epoch"]))
            print(
                f"  Fold {fold_k} | Seed {seed} | val_auprc={best_ap:.5f} "
                f"(best_epoch={hist['best_epoch']})"
            )

    assert oof_seeds is not None and test_seeds is not None
    oof_nn = oof_seeds.mean(axis=0)
    test_nn = test_seeds.mean(axis=0)

    return {
        "oof_nn": oof_nn,
        "test_nn": test_nn,
        "oof_seeds": oof_seeds,
        "seed_fold_auprcs": seed_fold_auprcs,
        "best_epochs": best_epochs,
    }
