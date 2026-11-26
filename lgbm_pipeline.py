"""Shared leak-free LightGBM cross-validation and evaluation pipeline for Member 2."""

import os
import time
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from nn_data_pipeline import (
    CAT_COLS,
    ENGINEERED_COLS,
    ID_COL,
    N_SPLITS,
    SEED,
    TARGET_COL,
    evaluate_auprc,
)

FOLD_COL = "fold"
TRAIN_LGBM_PATH = os.path.join("processed_data", "train_lgbm.parquet")
TEST_LGBM_PATH = os.path.join("processed_data", "test_lgbm.parquet")


def load_lgbm_datasets(
    train_path: str = TRAIN_LGBM_PATH,
    test_path: str = TEST_LGBM_PATH,
    verify_folds_path: str | None = "train_folds.csv",
    verify_test_raw_path: str | None = "test.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Load Member 1's LightGBM parquet artifacts and run strict integrity checks."""
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    excluded = {ID_COL, FOLD_COL, TARGET_COL}
    feature_cols = [c for c in train_df.columns if c not in excluded]
    cat_cols = [c for c in CAT_COLS if c in feature_cols]

    # Verify identical feature schema between train and test
    test_feature_cols = [c for c in test_df.columns if c != ID_COL]
    if feature_cols != test_feature_cols:
        raise ValueError("Train and test feature columns do not match in name or order.")

    # Verify zero Inf values in numeric columns
    num_tr = train_df[feature_cols].select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    num_te = test_df[feature_cols].select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if np.isinf(num_tr).any() or np.isinf(num_te).any():
        raise ValueError("Infinite values detected in LightGBM numeric features.")

    # Verify exact row alignment with shared train_folds.csv if present
    if verify_folds_path and os.path.exists(verify_folds_path):
        tf = pd.read_csv(verify_folds_path, usecols=[ID_COL, FOLD_COL, TARGET_COL])
        if not np.array_equal(train_df[ID_COL].to_numpy(), tf[ID_COL].to_numpy()):
            raise ValueError("ID misalignment between train_lgbm.parquet and train_folds.csv.")
        if not np.array_equal(train_df[FOLD_COL].to_numpy(), tf[FOLD_COL].to_numpy()):
            raise ValueError("Fold misalignment between train_lgbm.parquet and train_folds.csv.")
        if not np.array_equal(
            train_df[TARGET_COL].astype(int).to_numpy(),
            tf[TARGET_COL].astype(int).to_numpy(),
        ):
            raise ValueError("Target misalignment between train_lgbm.parquet and train_folds.csv.")

    # Verify exact row alignment with raw test.csv if present
    if verify_test_raw_path and os.path.exists(verify_test_raw_path):
        te_raw = pd.read_csv(verify_test_raw_path, usecols=[ID_COL])
        if not np.array_equal(test_df[ID_COL].to_numpy(), te_raw[ID_COL].to_numpy()):
            raise ValueError("ID misalignment between test_lgbm.parquet and test.csv.")

    return train_df, test_df, feature_cols, cat_cols


def compute_scale_pos_weight(y_train: np.ndarray, mode: str | float = "none") -> float:
    """Compute class weight strictly from a training fold target vector."""
    if isinstance(mode, (int, float)) and not isinstance(mode, bool):
        return float(mode)

    y_arr = np.asarray(y_train, dtype=np.int64)
    n_neg = int(np.sum(y_arr == 0))
    n_pos = int(np.sum(y_arr == 1))
    if n_pos == 0:
        raise ValueError("Training fold contains zero positive samples.")

    ratio = n_neg / n_pos
    mode_key = str(mode).lower()
    if mode_key == "none":
        return 1.0
    if mode_key == "balanced":
        return float(ratio)
    if mode_key == "sqrt":
        return float(np.sqrt(ratio))
    raise ValueError(f"Unsupported scale_pos_weight mode: {mode}")


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    verbose: bool = False,
) -> dict[str, Any]:
    """Evaluate continuous fraud probabilities using AUPRC (primary) and auxiliary metrics."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_pred_proba, dtype=np.float64)

    auprc = evaluate_auprc(y_true_arr, y_prob_arr)
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    precisions, recalls, thresholds = precision_recall_curve(y_true_arr, y_prob_arr)
    f1_scores = np.divide(
        2.0 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) > 0,
    )
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1 = float(f1_scores[best_idx])
    best_precision = float(precisions[best_idx])
    best_recall = float(recalls[best_idx])
    f1_at_05 = float(f1_score(y_true_arr, (y_prob_arr >= 0.5).astype(int), zero_division=0))
    cm = confusion_matrix(y_true_arr, (y_prob_arr >= best_threshold).astype(int))

    if verbose:
        print("=" * 56)
        print(f"OOF AUPRC (Primary) : {auprc:.6f}")
        print(f"OOF ROC-AUC         : {roc_auc:.6f}")
        print(f"F1 @ 0.50           : {f1_at_05:.6f}")
        print(
            f"Best F1             : {best_f1:.6f} @ threshold={best_threshold:.4f} "
            f"(P={best_precision:.4f}, R={best_recall:.4f})"
        )
        print(f"Confusion Matrix (Best Thresh):\n{cm}")
        print("=" * 56)

    return {
        "auprc": auprc,
        "roc_auc": roc_auc,
        "f1_at_05": f1_at_05,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def get_default_lgbm_params(seed: int = SEED) -> dict[str, Any]:
    """Return clean baseline LightGBM hyperparameters for binary fraud detection."""
    return {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 1.0,
        "subsample_freq": 0,
        "colsample_bytree": 1.0,
        "min_child_samples": 20,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def run_lgbm_cv(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any] | None = None,
    test_df: pd.DataFrame | None = None,
    cat_cols: list[str] | None = None,
    spw_mode: str | float | None = None,
    stopping_rounds: int = 100,
    seeds: Iterable[int] = (SEED,),
    verbose: bool = True,
) -> dict[str, Any]:
    """Run leak-free Stratified 5-Fold LightGBM CV using train_df['fold']."""
    start_time = time.perf_counter()
    seeds_list = list(seeds)
    base_params = get_default_lgbm_params()
    if params is not None:
        base_params.update(params)

    if cat_cols is None:
        cat_cols = [c for c in CAT_COLS if c in feature_cols]

    X_all = train_df[feature_cols]
    y_all = train_df[TARGET_COL].astype(np.int64).to_numpy()
    folds = train_df[FOLD_COL].to_numpy()
    X_test = test_df[feature_cols] if test_df is not None else None

    oof_preds = np.zeros(len(train_df), dtype=np.float64)
    test_preds = np.zeros(len(test_df), dtype=np.float64) if test_df is not None else None

    fold_auprcs: list[float] = []
    best_iterations: list[int] = []
    gain_importances = np.zeros(len(feature_cols), dtype=np.float64)
    split_importances = np.zeros(len(feature_cols), dtype=np.float64)

    for fold in range(N_SPLITS):
        tr_mask = folds != fold
        va_mask = folds == fold

        X_tr, y_tr = X_all.loc[tr_mask], y_all[tr_mask]
        X_va, y_va = X_all.loc[va_mask], y_all[va_mask]

        fold_val_preds = np.zeros(int(va_mask.sum()), dtype=np.float64)
        fold_test_preds = np.zeros(len(test_df), dtype=np.float64) if X_test is not None else None
        fold_iters: list[int] = []

        for seed in seeds_list:
            fold_params = dict(base_params)
            fold_params["random_state"] = int(seed)
            fold_params["bagging_seed"] = int(seed)
            fold_params["feature_fraction_seed"] = int(seed)

            # Compute fold-local scale_pos_weight strictly on training fold
            if spw_mode is not None:
                fold_params.pop("is_unbalance", None)
                fold_params["scale_pos_weight"] = compute_scale_pos_weight(y_tr, mode=spw_mode)

            model = lgb.LGBMClassifier(**fold_params)
            callbacks = [lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)]

            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="average_precision",
                categorical_feature=cat_cols if cat_cols else "auto",
                callbacks=callbacks,
            )

            best_iter = int(model.best_iteration_ or fold_params.get("n_estimators", 2000))
            fold_iters.append(best_iter)

            fold_val_preds += model.predict_proba(X_va, num_iteration=best_iter)[:, 1] / len(seeds_list)
            if X_test is not None and fold_test_preds is not None:
                fold_test_preds += model.predict_proba(X_test, num_iteration=best_iter)[:, 1] / len(seeds_list)

            gain_importances += model.booster_.feature_importance(importance_type="gain") / (N_SPLITS * len(seeds_list))
            split_importances += model.booster_.feature_importance(importance_type="split") / (N_SPLITS * len(seeds_list))

        oof_preds[va_mask] = fold_val_preds
        if test_preds is not None and fold_test_preds is not None:
            test_preds += fold_test_preds / N_SPLITS

        fold_auprc = evaluate_auprc(y_va, fold_val_preds)
        fold_auprcs.append(fold_auprc)
        avg_iter = int(round(float(np.mean(fold_iters))))
        best_iterations.append(avg_iter)

        if verbose:
            print(f"Fold {fold} | AUPRC: {fold_auprc:.6f} | Best Iter: {avg_iter}")

    elapsed = time.perf_counter() - start_time
    overall = evaluate_predictions(y_all, oof_preds, verbose=verbose)
    mean_auprc = float(np.mean(fold_auprcs))
    std_auprc = float(np.std(fold_auprcs))

    if verbose:
        print(
            f"Mean 5-Fold AUPRC: {mean_auprc:.6f} +/- {std_auprc:.6f} "
            f"| Avg Best Iter: {np.mean(best_iterations):.1f} | Time: {elapsed:.1f}s"
        )

    fi_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "gain": gain_importances,
            "split": split_importances,
            "is_engineered": [c in ENGINEERED_COLS for c in feature_cols],
        }
    ).sort_values("gain", ascending=False).reset_index(drop=True)
    fi_df["gain_pct"] = 100.0 * fi_df["gain"] / max(fi_df["gain"].sum(), 1e-12)

    return {
        "oof_preds": oof_preds,
        "test_preds": test_preds,
        "fold_auprcs": fold_auprcs,
        "mean_auprc": mean_auprc,
        "std_auprc": std_auprc,
        "metrics": overall,
        "best_iterations": best_iterations,
        "feature_importance_df": fi_df,
        "train_time_sec": float(elapsed),
    }