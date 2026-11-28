"""Shared 5-fold LightGBM SHAP & Gain embedded feature selection module."""

import json
import os
import sys
import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DATA_PIPELINE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_pipeline.nn_data_pipeline import (
    CAT_COLS,
    ENGINEERED_COLS,
    N_SPLITS,
    SEED,
    TARGET_COL,
    evaluate_auprc,
)
from lgbm.lgbm_pipeline import get_default_lgbm_params, run_lgbm_cv

FOLD_COL = "fold"
DEFAULT_SELECTED_FEATURES_PATH = os.path.join(DATA_PIPELINE_DIR, "selected_features.json")


def compute_5fold_shap_and_gain(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any] | None = None,
    cat_cols: list[str] | None = None,
    stopping_rounds: int = 100,
    verify_with_shap_pkg: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train 5-fold LightGBM and compute out-of-fold validation mean |SHAP| and Gain importances."""
    t0 = time.perf_counter()
    base_params = get_default_lgbm_params(seed=SEED)
    if params is not None:
        base_params.update(params)

    if cat_cols is None:
        cat_cols = [c for c in CAT_COLS if c in feature_cols]

    X_all = train_df[feature_cols]
    y_all = train_df[TARGET_COL].astype(np.int64).to_numpy()
    folds = train_df[FOLD_COL].to_numpy()

    oof_preds = np.zeros(len(train_df), dtype=np.float64)
    fold_shap_abs = np.zeros((N_SPLITS, len(feature_cols)), dtype=np.float64)
    fold_gain = np.zeros((N_SPLITS, len(feature_cols)), dtype=np.float64)
    fold_split = np.zeros((N_SPLITS, len(feature_cols)), dtype=np.float64)
    fold_auprcs: list[float] = []
    best_iterations: list[int] = []
    shap_verify_max_diff: float | None = None

    for fold in range(N_SPLITS):
        tr_mask = folds != fold
        va_mask = folds == fold
        X_tr, y_tr = X_all.loc[tr_mask], y_all[tr_mask]
        X_va, y_va = X_all.loc[va_mask], y_all[va_mask]

        model = lgb.LGBMClassifier(**base_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="average_precision",
            categorical_feature=cat_cols if cat_cols else "auto",
            callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)],
        )

        best_iter = int(model.best_iteration_ or base_params.get("n_estimators", 2000))
        best_iterations.append(best_iter)

        val_preds = model.predict_proba(X_va, num_iteration=best_iter)[:, 1]
        oof_preds[va_mask] = val_preds
        fold_auprc = evaluate_auprc(y_va, val_preds)
        fold_auprcs.append(fold_auprc)

        t_shap = time.perf_counter()
        contrib = model.booster_.predict(X_va, num_iteration=best_iter, pred_contrib=True)
        shap_vals = contrib[:, :-1]
        fold_shap_abs[fold] = np.mean(np.abs(shap_vals), axis=0)
        shap_sec = time.perf_counter() - t_shap

        if verify_with_shap_pkg and fold == 0:
            try:
                import shap

                explainer = shap.TreeExplainer(model)
                sv_check = explainer.shap_values(X_va.iloc[:200])
                if isinstance(sv_check, list):
                    sv_check = sv_check[1]
                shap_verify_max_diff = float(np.max(np.abs(sv_check - shap_vals[:200])))
                if verbose:
                    print(
                        f"[Verification] Fold 0 native pred_contrib vs shap.TreeExplainer "
                        f"max_abs_diff = {shap_verify_max_diff:.2e}"
                    )
            except Exception as exc:
                if verbose:
                    print(f"[Verification] Skipped shap.TreeExplainer check: {exc}")

        fold_gain[fold] = model.booster_.feature_importance(importance_type="gain")
        fold_split[fold] = model.booster_.feature_importance(importance_type="split")

        if verbose:
            print(
                f"Fold {fold} | Val AUPRC: {fold_auprc:.6f} | "
                f"Best Iter: {best_iter:4d} | Val SHAP time: {shap_sec:.1f}s"
            )

    oof_auprc = evaluate_auprc(y_all, oof_preds)
    elapsed = time.perf_counter() - t0

    mean_shap = fold_shap_abs.mean(axis=0)
    std_shap = fold_shap_abs.std(axis=0)
    mean_gain = fold_gain.mean(axis=0)
    mean_split = fold_split.mean(axis=0)

    ranking_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "is_engineered": [c in ENGINEERED_COLS for c in feature_cols],
            "mean_abs_shap": mean_shap,
            "std_abs_shap": std_shap,
            "mean_gain": mean_gain,
            "mean_split": mean_split,
        }
    )
    ranking_df["shap_pct"] = 100.0 * ranking_df["mean_abs_shap"] / max(ranking_df["mean_abs_shap"].sum(), 1e-12)
    ranking_df["gain_pct"] = 100.0 * ranking_df["mean_gain"] / max(ranking_df["mean_gain"].sum(), 1e-12)
    ranking_df = ranking_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    ranking_df["shap_rank"] = np.arange(1, len(ranking_df) + 1)
    ranking_df["gain_rank"] = ranking_df["mean_gain"].rank(ascending=False, method="min").astype(int)
    ranking_df["cum_shap_pct"] = ranking_df["shap_pct"].cumsum()

    if verbose:
        print(
            f"\nBaseline 5-Fold OOF AUPRC: {oof_auprc:.6f} "
            f"| Mean Fold AUPRC: {np.mean(fold_auprcs):.6f} +/- {np.std(fold_auprcs):.6f} "
            f"| Elapsed: {elapsed:.1f}s"
        )

    return {
        "ranking_df": ranking_df,
        "oof_preds": oof_preds,
        "oof_auprc": oof_auprc,
        "fold_auprcs": fold_auprcs,
        "best_iterations": best_iterations,
        "shap_verify_max_diff": shap_verify_max_diff,
        "elapsed_sec": float(elapsed),
    }


def evaluate_pruning_candidates(
    train_df: pd.DataFrame,
    candidates: list[tuple[str, str, list[str]]],
    all_feature_cols: list[str],
    params: dict[str, Any] | None = None,
    stopping_rounds: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """Re-run 5-fold Stratified LightGBM CV on each pruned feature subset and compare AUPRC."""
    rows: list[dict[str, Any]] = []
    baseline_auprc: float | None = None

    for label, criterion, cols in candidates:
        res = run_lgbm_cv(
            train_df,
            cols,
            params=params,
            stopping_rounds=stopping_rounds,
            verbose=False,
        )
        oof_auprc = float(res["metrics"]["auprc"])
        if baseline_auprc is None:
            baseline_auprc = oof_auprc
        delta_auprc = oof_auprc - baseline_auprc
        dropped = [c for c in all_feature_cols if c not in cols]

        row = {
            "experiment": label,
            "criterion": criterion,
            "n_features": len(cols),
            "n_dropped": len(dropped),
            "pct_dropped": round(100.0 * len(dropped) / len(all_feature_cols), 1),
            "oof_auprc": oof_auprc,
            "delta_auprc": delta_auprc,
            "mean_fold_auprc": float(res["mean_auprc"]),
            "std_fold_auprc": float(res["std_auprc"]),
            "roc_auc": float(res["metrics"]["roc_auc"]),
            "best_f1": float(res["metrics"]["best_f1"]),
            "train_time_sec": float(res["train_time_sec"]),
            "dropped_features": dropped,
        }
        rows.append(row)

        if verbose:
            print(
                f"{label:34s} | feats={len(cols):2d} (-{len(dropped):2d}) | "
                f"OOF AUPRC={oof_auprc:.6f} ({delta_auprc:+.6f}) | "
                f"Mean={res['mean_auprc']:.6f}+/-{res['std_auprc']:.6f} | "
                f"{res['train_time_sec']:.1f}s"
            )

    return pd.DataFrame(rows)


def export_selected_features(
    selected_features: list[str],
    dropped_features: list[str],
    zero_gain_features: list[str],
    ranking_df: pd.DataFrame,
    pruning_df: pd.DataFrame,
    criterion: str,
    baseline_oof_auprc: float,
    selected_oof_auprc: float,
    output_path: str = DEFAULT_SELECTED_FEATURES_PATH,
) -> str:
    """Persist selected feature subset and embedded-selection audit metadata to JSON."""
    payload = {
        "method": "5-Fold LightGBM Embedded Feature Selection (Validation Mean |SHAP| & Gain)",
        "selection_criterion": criterion,
        "n_total_features": len(selected_features) + len(dropped_features),
        "n_selected_features": len(selected_features),
        "n_dropped_features": len(dropped_features),
        "baseline_oof_auprc": float(baseline_oof_auprc),
        "selected_oof_auprc": float(selected_oof_auprc),
        "delta_auprc": float(selected_oof_auprc - baseline_oof_auprc),
        "selected_features": list(selected_features),
        "dropped_features": list(dropped_features),
        "zero_gain_features": list(zero_gain_features),
        "pruning_cv_summary": pruning_df.drop(columns=["dropped_features"], errors="ignore").to_dict(
            orient="records"
        ),
        "feature_ranking": ranking_df.to_dict(orient="records"),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return os.path.abspath(output_path)


def load_selected_features(path: str = DEFAULT_SELECTED_FEATURES_PATH) -> list[str]:
    """Load the team-wide selected feature list exported to selected_features.json."""
    if not os.path.isabs(path) and not os.path.exists(path):
        cand = os.path.join(DATA_PIPELINE_DIR, path)
        if os.path.exists(cand):
            path = cand
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload["selected_features"])
