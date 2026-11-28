"""Leak-free fold-wise Neural Network & shared data pipeline built from train_folds.csv."""

import os
from typing import Iterator
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SEED = 42
N_SPLITS = 5
ID_COL = "id"
TARGET_COL = "is_fraud"

CONT_COLS = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "merchant_risk_score",
    "prior_disputes",
]

TIME_COLS = ["time_of_day_hour", "day_of_week"]

CAT_COLS = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
]

RISK_FLAG_COLS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]

ENGINEERED_COLS = [
    "amount_to_balance_ratio",
    "is_over_balance",
    "amount_per_txn_24h",
    "travel_speed_kmh",
    "is_brand_new_card",
    "has_cvv_retry",
    "hour_sin",
    "hour_cos",
    "is_night_txn",
    "is_weekend",
    "total_risk_flags",
]

NN_NUM_COLS = CONT_COLS + TIME_COLS + [
    "amount_to_balance_ratio",
    "amount_per_txn_24h",
    "travel_speed_kmh",
    "hour_sin",
    "hour_cos",
    "total_risk_flags",
]

NN_SKEW_COLS = [
    "amount_usd",
    "account_balance_usd",
    "distance_from_home_km",
    "hours_since_last_txn",
    "travel_speed_kmh",
]

NN_BIN_COLS = RISK_FLAG_COLS + [
    "is_over_balance",
    "is_brand_new_card",
    "has_cvv_retry",
    "is_night_txn",
    "is_weekend",
]


def _resolve_root_path(path: str) -> str:
    """Resolve relative paths against cwd first, then PROJECT_ROOT."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(PROJECT_ROOT, path)
    return candidate if os.path.exists(candidate) else path


def evaluate_auprc(y_true, y_pred_proba) -> float:
    """Shared primary evaluation metric (Area Under the Precision-Recall Curve)."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_pred_proba, dtype=np.float64)
    if np.isnan(y_prob_arr).any() or np.isinf(y_prob_arr).any():
        raise ValueError("Predicted probabilities contain NaN or Inf values.")
    return float(average_precision_score(y_true_arr, y_prob_arr))


def ensure_train_folds(
    train_folds_path: str = "train_folds.csv",
    raw_train_path: str = "train.csv",
) -> pd.DataFrame:
    """Load train_folds.csv, or create it deterministically from train.csv if missing."""
    tf_path = _resolve_root_path(train_folds_path)
    if os.path.exists(tf_path):
        df = pd.read_csv(tf_path)
        df[TARGET_COL] = df[TARGET_COL].astype(int)
        return df

    raw_path = _resolve_root_path(raw_train_path)
    train_raw = pd.read_csv(raw_path)
    train_raw[TARGET_COL] = train_raw[TARGET_COL].astype(int)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    train_raw["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_raw, train_raw[TARGET_COL])):
        train_raw.iloc[val_idx, train_raw.columns.get_loc("fold")] = fold_idx
    out_path = os.path.join(PROJECT_ROOT, train_folds_path) if not os.path.isabs(train_folds_path) else train_folds_path
    train_raw.to_csv(out_path, index=False)
    return train_raw


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate deterministic row-level domain features identically for train and test."""
    out = df.copy()
    for flag_col in RISK_FLAG_COLS:
        if out[flag_col].isnull().any():
            out[flag_col] = out[flag_col].astype(float)
        else:
            out[flag_col] = out[flag_col].astype(int)

    out["amount_to_balance_ratio"] = out["amount_usd"] / (out["account_balance_usd"] + 1e-5)
    out["is_over_balance"] = (out["amount_usd"] > out["account_balance_usd"]).astype(int)
    out["amount_per_txn_24h"] = out["amount_usd"] / (out["txn_count_last_24h"] + 1)

    out["travel_speed_kmh"] = out["distance_from_home_km"] / (out["hours_since_last_txn"] + 0.1)
    out["is_brand_new_card"] = (out["card_age_months"] <= 1).astype(int)
    out["has_cvv_retry"] = (out["cvv_retry_count"] > 0).astype(int)

    out["hour_sin"] = np.sin(2.0 * np.pi * out["time_of_day_hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["time_of_day_hour"] / 24.0)
    out["is_night_txn"] = out["time_of_day_hour"].between(0, 5).astype(int)

    dow = out["day_of_week"]
    if pd.api.types.is_numeric_dtype(dow):
        dow_valid = dow.dropna()
        if len(dow_valid) > 0 and dow_valid.min() >= 1 and dow_valid.max() == 7:
            out["is_weekend"] = dow.isin([6, 7]).astype(int)
        else:
            out["is_weekend"] = dow.isin([5, 6]).astype(int)
    else:
        weekend_strings = {
            "saturday", "sunday", "sat", "sun",
            "thứ 7", "thứ bảy", "chủ nhật", "thu 7", "thu bay", "chu nhat", "t7", "cn",
        }
        out["is_weekend"] = dow.astype(str).str.strip().str.lower().isin(weekend_strings).astype(int)

    out["total_risk_flags"] = out[RISK_FLAG_COLS].sum(axis=1)
    return out


def _prep_cat_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    arr = df[cols].astype(object).to_numpy(copy=True)
    arr[pd.isna(arr)] = np.nan
    return arr


def _normalize_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    if pd.api.types.is_numeric_dtype(df["day_of_week"]):
        return df
    out = df.copy()
    dow_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
        "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
    }
    out["day_of_week"] = out["day_of_week"].astype(str).str.lower().map(dow_map)
    return out


def _transform_single_fold(
    train_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
    fold: int,
) -> dict:
    """Fit imputers, log1p, RobustScaler, and OneHotEncoder strictly on fold != k."""
    skew_col_indices = [NN_NUM_COLS.index(c) for c in NN_SKEW_COLS]
    tr_mask = (train_fe["fold"] != fold).to_numpy()
    va_mask = (train_fe["fold"] == fold).to_numpy()

    num_imputer = SimpleImputer(strategy="median")
    X_tr_num = num_imputer.fit_transform(train_fe.loc[tr_mask, NN_NUM_COLS].to_numpy(dtype=np.float64))
    X_va_num = num_imputer.transform(train_fe.loc[va_mask, NN_NUM_COLS].to_numpy(dtype=np.float64))
    X_te_num = num_imputer.transform(test_fe[NN_NUM_COLS].to_numpy(dtype=np.float64))

    X_tr_num[:, skew_col_indices] = np.log1p(np.clip(X_tr_num[:, skew_col_indices], 0.0, None))
    X_va_num[:, skew_col_indices] = np.log1p(np.clip(X_va_num[:, skew_col_indices], 0.0, None))
    X_te_num[:, skew_col_indices] = np.log1p(np.clip(X_te_num[:, skew_col_indices], 0.0, None))

    scaler = RobustScaler()
    X_tr_num = scaler.fit_transform(X_tr_num)
    X_va_num = scaler.transform(X_va_num)
    X_te_num = scaler.transform(X_te_num)

    bin_imputer = SimpleImputer(strategy="most_frequent")
    X_tr_bin = bin_imputer.fit_transform(train_fe.loc[tr_mask, NN_BIN_COLS].to_numpy(dtype=np.float64))
    X_va_bin = bin_imputer.transform(train_fe.loc[va_mask, NN_BIN_COLS].to_numpy(dtype=np.float64))
    X_te_bin = bin_imputer.transform(test_fe[NN_BIN_COLS].to_numpy(dtype=np.float64))

    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_tr_cat_imp = cat_imputer.fit_transform(_prep_cat_matrix(train_fe.loc[tr_mask], CAT_COLS))
    X_va_cat_imp = cat_imputer.transform(_prep_cat_matrix(train_fe.loc[va_mask], CAT_COLS))
    X_te_cat_imp = cat_imputer.transform(_prep_cat_matrix(test_fe, CAT_COLS))

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr_cat = ohe.fit_transform(X_tr_cat_imp)
    X_va_cat = ohe.transform(X_va_cat_imp)
    X_te_cat = ohe.transform(X_te_cat_imp)
    ohe_feature_names = ohe.get_feature_names_out(CAT_COLS).tolist()

    return {
        "X_train": np.hstack([X_tr_num, X_tr_bin, X_tr_cat]).astype(np.float32),
        "y_train": train_fe.loc[tr_mask, TARGET_COL].to_numpy(dtype=np.int64),
        "X_val": np.hstack([X_va_num, X_va_bin, X_va_cat]).astype(np.float32),
        "y_val": train_fe.loc[va_mask, TARGET_COL].to_numpy(dtype=np.int64),
        "X_test": np.hstack([X_te_num, X_te_bin, X_te_cat]).astype(np.float32),
        "feature_names": NN_NUM_COLS + NN_BIN_COLS + ohe_feature_names,
        "val_ids": train_fe.loc[va_mask, ID_COL].to_numpy(dtype=np.int64),
        "test_ids": test_fe[ID_COL].to_numpy(dtype=np.int64),
    }


def get_nn_fold_data(
    fold: int,
    train_folds_path: str = "train_folds.csv",
    test_path: str = "test.csv",
) -> dict:
    """Load and preprocess a single fold k (0..4) directly from train_folds.csv & test.csv."""
    if fold not in range(N_SPLITS):
        raise ValueError(f"fold must be in 0..{N_SPLITS - 1}, got {fold}")
    train_fe = _normalize_day_of_week(build_features(ensure_train_folds(train_folds_path)))
    test_fe = _normalize_day_of_week(build_features(pd.read_csv(_resolve_root_path(test_path))))
    return _transform_single_fold(train_fe, test_fe, fold)


def iter_nn_folds(
    train_folds_path: str = "train_folds.csv",
    test_path: str = "test.csv",
) -> Iterator[tuple[int, dict]]:
    """Yield (fold_k, fold_dict) for folds 0..4 while reading CSVs only once."""
    train_fe = _normalize_day_of_week(build_features(ensure_train_folds(train_folds_path)))
    test_fe = _normalize_day_of_week(build_features(pd.read_csv(_resolve_root_path(test_path))))
    for k in range(N_SPLITS):
        yield k, _transform_single_fold(train_fe, test_fe, k)
