"""Shared leak-free ensemble blending and nested CV AUPRC validation utilities for Phase 3."""

from typing import Any
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import pearsonr, rankdata, spearmanr

from nn_data_pipeline import N_SPLITS, evaluate_auprc

EPS = 1e-7


def clip_probs(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Clip probabilities strictly inside (eps, 1 - eps) for numerical stability."""
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)


def prior_calibrate_spw(p: np.ndarray, spw: float) -> np.ndarray:
    """Map probabilities trained with scale_pos_weight back to unweighted posterior scale."""
    p_c = clip_probs(p)
    odds = (p_c / (1.0 - p_c)) / float(spw)
    return odds / (1.0 + odds)


def to_percentile_rank(p: np.ndarray) -> np.ndarray:
    """Convert continuous scores into uniform percentile ranks in (0, 1)."""
    arr = np.asarray(p, dtype=np.float64)
    return rankdata(arr, method="average") / (len(arr) + 1.0)


def _prepare_pair(
    p_lgbm: np.ndarray,
    p_nn: np.ndarray,
    method: str,
    lgbm_spw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-transform probability vectors once before sweeping blend weights."""
    p1 = clip_probs(p_lgbm)
    p2 = clip_probs(p_nn)
    if method == "raw_linear":
        return p1, p2
    if method in {"prior_calibrated", "geometric"}:
        return prior_calibrate_spw(p1, spw=lgbm_spw), p2
    if method == "logit":
        return logit(p1) - np.log(float(lgbm_spw)), logit(p2)
    if method == "rank_average":
        return to_percentile_rank(p1), to_percentile_rank(p2)
    raise ValueError(f"Unknown blending method: {method}")


def _combine_prepared(
    t1: np.ndarray,
    t2: np.ndarray,
    weight_lgbm: float,
    method: str,
) -> np.ndarray:
    w = float(weight_lgbm)
    if method in {"raw_linear", "prior_calibrated", "rank_average"}:
        return w * t1 + (1.0 - w) * t2
    if method == "logit":
        return expit(w * t1 + (1.0 - w) * t2)
    if method == "geometric":
        return np.power(t1, w) * np.power(t2, 1.0 - w)
    raise ValueError(f"Unknown blending method: {method}")


def blend_predictions(
    p_lgbm: np.ndarray,
    p_nn: np.ndarray,
    weight_lgbm: float,
    method: str = "logit",
    lgbm_spw: float = 3.359272372588889,
) -> np.ndarray:
    """Combine LightGBM and Neural Network probabilities under a specified blending rule."""
    t1, t2 = _prepare_pair(p_lgbm, p_nn, method=method, lgbm_spw=lgbm_spw)
    return _combine_prepared(t1, t2, weight_lgbm=weight_lgbm, method=method)


def compute_diversity_metrics(
    y_true: np.ndarray,
    oof_lgbm: np.ndarray,
    oof_nn: np.ndarray,
    top_frac: float = 0.02,
) -> dict[str, float]:
    """Quantify correlation and high-risk prediction diversity between GBDT and MLP."""
    y_arr = np.asarray(y_true, dtype=np.int64)
    p_l = np.asarray(oof_lgbm, dtype=np.float64)
    p_n = np.asarray(oof_nn, dtype=np.float64)

    pearson_all = float(pearsonr(p_l, p_n)[0])
    spearman_all = float(spearmanr(p_l, p_n)[0])
    fraud_mask = y_arr == 1
    pearson_fraud = float(pearsonr(p_l[fraud_mask], p_n[fraud_mask])[0])
    spearman_fraud = float(spearmanr(p_l[fraud_mask], p_n[fraud_mask])[0])

    k = max(1, int(round(len(y_arr) * top_frac)))
    top_l = set(np.argsort(p_l)[-k:])
    top_n = set(np.argsort(p_n)[-k:])
    jaccard_top = len(top_l & top_n) / max(1, len(top_l | top_n))

    return {
        "pearson_all": pearson_all,
        "spearman_all": spearman_all,
        "pearson_fraud_only": pearson_fraud,
        "spearman_fraud_only": spearman_fraud,
        "top_2pct_jaccard_overlap": float(jaccard_top),
    }


def optimize_blend_weight(
    y_true: np.ndarray,
    p_lgbm: np.ndarray,
    p_nn: np.ndarray,
    method: str,
    lgbm_spw: float = 3.359272372588889,
    grid_steps: int = 201,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Find optimal LightGBM weight w in [0, 1] maximizing AUPRC."""
    t1, t2 = _prepare_pair(p_lgbm, p_nn, method=method, lgbm_spw=lgbm_spw)
    weights = np.linspace(0.0, 1.0, grid_steps)
    scores = np.array(
        [evaluate_auprc(y_true, _combine_prepared(t1, t2, w, method=method)) for w in weights],
        dtype=np.float64,
    )
    best_idx = int(np.argmax(scores))
    w_grid = float(weights[best_idx])

    lo = max(0.0, w_grid - 0.02)
    hi = min(1.0, w_grid + 0.02)
    res = minimize_scalar(
        lambda w: -evaluate_auprc(y_true, _combine_prepared(t1, t2, w, method=method)),
        bounds=(lo, hi),
        method="bounded",
    )
    best_w = float(res.x) if -float(res.fun) >= scores[best_idx] else w_grid
    best_score = max(float(scores[best_idx]), -float(res.fun))
    return best_w, best_score, weights, scores


def evaluate_nested_blend_cv(
    y_true: np.ndarray,
    folds: np.ndarray,
    oof_lgbm: np.ndarray,
    oof_nn: np.ndarray,
    test_lgbm: np.ndarray,
    test_nn: np.ndarray,
    method: str,
    lgbm_spw: float = 3.359272372588889,
) -> dict[str, Any]:
    """Leak-free nested CV: fit blend weight on 4 folds by AUPRC, evaluate on held-out 5th fold."""
    nested_oof = np.zeros_like(oof_lgbm, dtype=np.float64)
    nested_test = np.zeros_like(test_lgbm, dtype=np.float64)
    fold_weights: list[float] = []
    fold_auprcs: list[float] = []

    for fold in range(N_SPLITS):
        tr_mask = folds != fold
        va_mask = folds == fold

        w_k, _, _, _ = optimize_blend_weight(
            y_true[tr_mask],
            oof_lgbm[tr_mask],
            oof_nn[tr_mask],
            method=method,
            lgbm_spw=lgbm_spw,
            grid_steps=101,
        )
        fold_weights.append(w_k)

        val_blend = blend_predictions(
            oof_lgbm[va_mask],
            oof_nn[va_mask],
            weight_lgbm=w_k,
            method=method,
            lgbm_spw=lgbm_spw,
        )
        nested_oof[va_mask] = val_blend
        fold_auprcs.append(evaluate_auprc(y_true[va_mask], val_blend))

        nested_test += (
            blend_predictions(
                test_lgbm,
                test_nn,
                weight_lgbm=w_k,
                method=method,
                lgbm_spw=lgbm_spw,
            )
            / N_SPLITS
        )

    global_w, global_auprc, grid_w, grid_s = optimize_blend_weight(
        y_true,
        oof_lgbm,
        oof_nn,
        method=method,
        lgbm_spw=lgbm_spw,
        grid_steps=201,
    )
    global_oof = blend_predictions(oof_lgbm, oof_nn, global_w, method=method, lgbm_spw=lgbm_spw)
    global_test = blend_predictions(test_lgbm, test_nn, global_w, method=method, lgbm_spw=lgbm_spw)
    nested_oof_auprc = evaluate_auprc(y_true, nested_oof)

    return {
        "method": method,
        "global_weight_lgbm": global_w,
        "global_oof_auprc": global_auprc,
        "nested_oof_auprc": nested_oof_auprc,
        "nested_mean_fold_auprc": float(np.mean(fold_auprcs)),
        "nested_std_fold_auprc": float(np.std(fold_auprcs)),
        "fold_weights": fold_weights,
        "fold_auprcs": fold_auprcs,
        "nested_oof": nested_oof,
        "nested_test": nested_test,
        "global_oof": global_oof,
        "global_test": global_test,
        "grid_weights": grid_w,
        "grid_scores": grid_s,
    }