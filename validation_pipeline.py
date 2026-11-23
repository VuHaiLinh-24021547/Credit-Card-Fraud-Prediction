import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def evaluate_fraud_predictions(y_true, y_pred_proba, verbose=True):
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    auprc = average_precision_score(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0
    )

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    f1_default = f1_score(y_true, (y_pred_proba >= 0.5).astype(int), zero_division=0)
    cm = confusion_matrix(y_true, (y_pred_proba >= best_threshold).astype(int))

    if verbose:
        print("=" * 50)
        print(f"AUPRC:          {auprc:.4f}")
        print(f"ROC-AUC:        {roc_auc:.4f}")
        print(f"F1 @ 0.50:      {f1_default:.4f}")
        print(f"Best Threshold: {best_threshold:.4f} => Best F1: {best_f1:.4f}")
        print(f"Confusion Matrix (Best Thresh):\n{cm}")
        print("=" * 50)

    return {
        "auprc": auprc,
        "roc_auc": roc_auc,
        "best_threshold": float(best_threshold),
        "best_f1": float(best_f1),
        "f1_default": float(f1_default),
        "confusion_matrix": cm
    }

def run_cv_pipeline(model_fn, feature_cols, data_path="train_folds.csv", n_splits=5, preprocess_fn=None):
    df = pd.read_csv(data_path)
    oof_predictions = np.zeros(len(df))
    fold_auprcs = []

    for fold in range(n_splits):
        train_mask = (df["fold"] != fold)
        val_mask = (df["fold"] == fold)

        X_train = df.loc[train_mask, feature_cols].copy()
        y_train = df.loc[train_mask, "is_fraud"].values
        X_val = df.loc[val_mask, feature_cols].copy()
        y_val = df.loc[val_mask, "is_fraud"].values

        if preprocess_fn is not None:
            X_train, y_train, X_val = preprocess_fn(X_train, y_train, X_val)

        model = model_fn()
        model.fit(X_train, y_train)

        val_preds = model.predict_proba(X_val)[:, 1]
        oof_predictions[val_mask] = val_preds

        fold_auprc = average_precision_score(y_val, val_preds)
        fold_auprcs.append(fold_auprc)
        print(f"Fold {fold} AUPRC: {fold_auprc:.4f}")

    print(f"\nMean AUPRC: {np.mean(fold_auprcs):.4f} +/- {np.std(fold_auprcs):.4f}")
    overall_metrics = evaluate_fraud_predictions(df["is_fraud"].values, oof_predictions, verbose=True)
    return oof_predictions, overall_metrics
