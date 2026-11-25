import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

NUM_COLS = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
]

SKEWED_NUM_COLS = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "account_balance_usd",
]

BINARY_FLAGS = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]

CAT_COLS = [
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
]

ALL_FEATURES = NUM_COLS + BINARY_FLAGS + CAT_COLS


def set_all_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _calibrate_scores_to_prob(train_scores, eval_scores):
    """
    Fit calibration parameters (mean, std) strictly on train_scores
    and map eval_scores to (0, 1) via logistic sigmoid over z-scores.
    """
    mu = float(np.mean(train_scores))
    sigma = float(np.std(train_scores)) + 1e-8
    z = np.clip((eval_scores - mu) / sigma, -25.0, 25.0)
    probs = 1.0 / (1.0 + np.exp(-z))
    return probs.astype(np.float64), mu, sigma


def make_dl_preprocess_fn(
    log_transform_skewed=True,
    scale_numeric=True,
    encode_mode="onehot",
):
    """
    Return preprocess_fn(X_train, y_train, X_val) compatible with
    validation_pipeline.run_cv_pipeline().

    Strict Zero Data Leakage:
    - Log1p is applied to right-skewed positive numeric columns.
    - StandardScaler and OneHotEncoder/OrdinalEncoder are fit strictly on X_train.
    - X_val is transformed only.
    """
    def preprocess_fn(X_train, y_train, X_val):
        num_cols = [c for c in NUM_COLS if c in X_train.columns]
        bin_cols = [c for c in BINARY_FLAGS if c in X_train.columns]
        cat_cols = [c for c in CAT_COLS if c in X_train.columns]

        tr_parts = []
        va_parts = []
        out_cols = []

        # 1. Numeric features
        if num_cols:
            X_tr_num = X_train[num_cols].to_numpy(dtype=np.float32, copy=True)
            X_va_num = X_val[num_cols].to_numpy(dtype=np.float32, copy=True)
            if log_transform_skewed:
                for idx, col_name in enumerate(num_cols):
                    if col_name in SKEWED_NUM_COLS:
                        X_tr_num[:, idx] = np.log1p(np.maximum(X_tr_num[:, idx], 0.0))
                        X_va_num[:, idx] = np.log1p(np.maximum(X_va_num[:, idx], 0.0))
            if scale_numeric:
                scaler = StandardScaler()
                X_tr_num = scaler.fit_transform(X_tr_num).astype(np.float32)
                X_va_num = scaler.transform(X_va_num).astype(np.float32)
            tr_parts.append(X_tr_num)
            va_parts.append(X_va_num)
            out_cols.extend(num_cols)

        # 2. Binary flags
        if bin_cols:
            X_tr_bin = X_train[bin_cols].astype(np.float32).to_numpy()
            X_va_bin = X_val[bin_cols].astype(np.float32).to_numpy()
            tr_parts.append(X_tr_bin)
            va_parts.append(X_va_bin)
            out_cols.extend(bin_cols)

        # 3. Categorical features
        if cat_cols:
            if encode_mode == "onehot":
                enc = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float32,
                )
                X_tr_cat = enc.fit_transform(X_train[cat_cols])
                X_va_cat = enc.transform(X_val[cat_cols])
                cat_names = [
                    name.replace(" ", "_").replace("-", "_")
                    for name in enc.get_feature_names_out(cat_cols)
                ]
            elif encode_mode == "ordinal":
                enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float32,
                )
                X_tr_cat = enc.fit_transform(X_train[cat_cols])
                X_va_cat = enc.transform(X_val[cat_cols])
                cat_names = list(cat_cols)
            else:
                raise ValueError(f"Unknown encode_mode: {encode_mode}")

            tr_parts.append(X_tr_cat)
            va_parts.append(X_va_cat)
            out_cols.extend(cat_names)

        X_tr_arr = np.hstack(tr_parts).astype(np.float32)
        X_va_arr = np.hstack(va_parts).astype(np.float32)
        y_tr_arr = np.asarray(y_train, dtype=np.int32)

        X_tr_df = pd.DataFrame(X_tr_arr, columns=out_cols)
        X_va_df = pd.DataFrame(X_va_arr, columns=out_cols, index=X_val.index)
        return X_tr_df, y_tr_arr, X_va_df

    return preprocess_fn


class IsolationForestFraudDetector(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper around IsolationForest for Fraud Anomaly Detection.
    Supports:
    - fit_mode='unsupervised': fit on all X_train samples
    - fit_mode='normal_only': fit strictly on normal transactions X_train[y_train == 0]
    """

    def __init__(
        self,
        n_estimators=200,
        max_samples=256,
        contamination=0.0167,
        max_features=1.0,
        fit_mode="unsupervised",
        random_state=42,
        n_jobs=-1,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.fit_mode = fit_mode
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X_arr = np.array(X, dtype=np.float32, copy=True)
        if self.fit_mode == "normal_only" and y is not None:
            y_arr = np.asarray(y)
            X_fit = X_arr[y_arr == 0]
        else:
            X_fit = X_arr

        self.iforest_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.iforest_.fit(X_fit)

        # Calibrate raw anomaly score distribution strictly on X_train
        raw_train_scores = -self.iforest_.score_samples(X_arr)
        self.score_mean_ = float(np.mean(raw_train_scores))
        self.score_std_ = float(np.std(raw_train_scores)) + 1e-8
        self.classes_ = np.array([0, 1])
        return self

    def raw_anomaly_score(self, X):
        X_arr = np.array(X, dtype=np.float32, copy=True)
        return -self.iforest_.score_samples(X_arr)

    def predict_proba(self, X):
        raw_scores = self.raw_anomaly_score(X)
        z = np.clip((raw_scores - self.score_mean_) / self.score_std_, -25.0, 25.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class TabularAutoencoderNet(nn.Module):
    """
    Deep Undercomplete / Denoising Autoencoder for Tabular Fraud Anomaly Detection.
    """

    def __init__(self, input_dim, hidden_dims=(64, 32), latent_dim=16, dropout=0.1):
        super().__init__()
        h1, h2 = hidden_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.SiLU(),
            nn.Linear(h2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.BatchNorm1d(h2),
            nn.SiLU(),
            nn.Linear(h2, h1),
            nn.BatchNorm1d(h1),
            nn.SiLU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x, noise_std=0.0):
        if self.training and noise_std > 0.0:
            x_in = x + torch.randn_like(x) * noise_std
        else:
            x_in = x
        z = self.encoder(x_in)
        x_hat = self.decoder(z)
        return x_hat, z


class AutoencoderFraudDetector(BaseEstimator, ClassifierMixin):
    """
    Semi-Supervised One-Class Deep Autoencoder trained strictly on Normal transactions (y == 0).
    Supports:
    - Standard Undercomplete Autoencoder (noise_std=0.0)
    - Denoising Autoencoder (noise_std > 0.0)
    - Multiple anomaly scoring modes: 'mse', 'mae', 'multi' (MSE + MaxSpike + Cosine + NumFlagWeighted)
    """

    def __init__(
        self,
        hidden_dims=(64, 32),
        latent_dim=16,
        dropout=0.05,
        noise_std=0.0,
        score_mode="mse",
        epochs=12,
        batch_size=2048,
        lr=3e-3,
        weight_decay=1e-4,
        fit_normal_only=True,
        random_state=42,
    ):
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.noise_std = noise_std
        self.score_mode = score_mode
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.fit_normal_only = fit_normal_only
        self.random_state = random_state

    def _compute_reconstruction_details(self, X_arr):
        self.model_.eval()
        n = len(X_arr)
        mse_all = np.empty(n, dtype=np.float32)
        mae_all = np.empty(n, dtype=np.float32)
        max_err = np.empty(n, dtype=np.float32)
        cos_dist = np.empty(n, dtype=np.float32)
        num_flag_mse = np.empty(n, dtype=np.float32)
        latents = np.empty((n, self.latent_dim), dtype=np.float32)

        core_dim = min(19, X_arr.shape[1])  # 13 NUM_COLS + 6 BINARY_FLAGS
        eval_bs = 8192

        with torch.no_grad():
            for start in range(0, n, eval_bs):
                end = min(start + eval_bs, n)
                xb = torch.from_numpy(X_arr[start:end])
                x_hat, zb = self.model_(xb, noise_std=0.0)
                diff = xb - x_hat
                sq_err = diff * diff

                mse_all[start:end] = torch.mean(sq_err, dim=1).numpy()
                mae_all[start:end] = torch.mean(torch.abs(diff), dim=1).numpy()
                max_err[start:end] = torch.max(sq_err, dim=1).values.numpy()
                num_flag_mse[start:end] = torch.mean(sq_err[:, :core_dim], dim=1).numpy()
                cos_sim = F.cosine_similarity(xb, x_hat, dim=1, eps=1e-8)
                cos_dist[start:end] = (1.0 - cos_sim).numpy()
                latents[start:end] = zb.numpy()

        return {
            "mse": mse_all,
            "mae": mae_all,
            "max": max_err,
            "cos": cos_dist,
            "num_flag_mse": num_flag_mse,
            "latents": latents,
        }

    def _aggregate_score(self, details, fit_stats=False):
        if self.score_mode == "mse":
            return np.log1p(details["mse"])
        if self.score_mode == "mae":
            return details["mae"]
        if self.score_mode == "multi":
            s_mse = np.log1p(details["mse"])
            s_core = np.log1p(details["num_flag_mse"])
            s_max = np.log1p(details["max"])
            s_cos = details["cos"]
            if fit_stats:
                self.multi_stats_ = {
                    "mse": (float(np.mean(s_mse)), float(np.std(s_mse)) + 1e-8),
                    "core": (float(np.mean(s_core)), float(np.std(s_core)) + 1e-8),
                    "max": (float(np.mean(s_max)), float(np.std(s_max)) + 1e-8),
                    "cos": (float(np.mean(s_cos)), float(np.std(s_cos)) + 1e-8),
                }
            z_mse = (s_mse - self.multi_stats_["mse"][0]) / self.multi_stats_["mse"][1]
            z_core = (s_core - self.multi_stats_["core"][0]) / self.multi_stats_["core"][1]
            z_max = (s_max - self.multi_stats_["max"][0]) / self.multi_stats_["max"][1]
            z_cos = (s_cos - self.multi_stats_["cos"][0]) / self.multi_stats_["cos"][1]
            return 0.35 * z_mse + 0.35 * z_core + 0.15 * z_max + 0.15 * z_cos
        raise ValueError(f"Unknown score_mode: {self.score_mode}")

    def fit(self, X, y=None):
        set_all_seeds(self.random_state)
        X_arr = np.array(X, dtype=np.float32, copy=True)
        if self.fit_normal_only and y is not None:
            y_arr = np.asarray(y)
            X_fit = X_arr[y_arr == 0]
        else:
            X_fit = X_arr

        input_dim = X_arr.shape[1]
        self.model_ = TabularAutoencoderNet(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        )

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.05
        )

        n_fit = len(X_fit)
        rng = np.random.default_rng(self.random_state)

        self.model_.train()
        for _ in range(self.epochs):
            perm = rng.permutation(n_fit)
            for start in range(0, n_fit, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = torch.from_numpy(X_fit[idx])
                optimizer.zero_grad(set_to_none=True)
                x_hat, _ = self.model_(xb, noise_std=self.noise_std)
                loss = F.mse_loss(x_hat, xb)
                loss.backward()
                optimizer.step()
            scheduler.step()

        train_details = self._compute_reconstruction_details(X_arr)
        raw_train_scores = self._aggregate_score(train_details, fit_stats=True)
        self.score_mean_ = float(np.mean(raw_train_scores))
        self.score_std_ = float(np.std(raw_train_scores)) + 1e-8
        self.classes_ = np.array([0, 1])
        return self

    def transform_anomaly_features(self, X):
        X_arr = np.array(X, dtype=np.float32, copy=True)
        return self._compute_reconstruction_details(X_arr)

    def predict_proba(self, X):
        X_arr = np.array(X, dtype=np.float32, copy=True)
        details = self._compute_reconstruction_details(X_arr)
        raw_scores = self._aggregate_score(details, fit_stats=False)
        z = np.clip((raw_scores - self.score_mean_) / self.score_std_, -25.0, 25.0)
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ResidualTabularBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class TabularMLPNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64, 32), dropout=0.15, use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        h1, h2, h3 = hidden_dims
        if not use_residual:
            self.net = nn.Sequential(
                nn.Linear(input_dim, h1),
                nn.BatchNorm1d(h1),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(h1, h2),
                nn.BatchNorm1d(h2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(h2, h3),
                nn.BatchNorm1d(h3),
                nn.SiLU(),
                nn.Linear(h3, 1),
            )
        else:
            self.in_proj = nn.Sequential(
                nn.Linear(input_dim, h1),
                nn.BatchNorm1d(h1),
                nn.SiLU(),
            )
            self.res1 = ResidualTabularBlock(h1, dropout=dropout)
            self.down = nn.Sequential(
                nn.Linear(h1, h2),
                nn.BatchNorm1d(h2),
                nn.SiLU(),
            )
            self.res2 = ResidualTabularBlock(h2, dropout=dropout)
            self.head = nn.Sequential(
                nn.BatchNorm1d(h2),
                nn.SiLU(),
                nn.Linear(h2, 1),
            )

    def forward(self, x):
        if not self.use_residual:
            return self.net(x).squeeze(-1)
        h = self.in_proj(x)
        h = self.res1(h)
        h = self.down(h)
        h = self.res2(h)
        return self.head(h).squeeze(-1)


class JointSupAENet(nn.Module):
    """
    Dual-Head Joint Supervised Autoencoder (SupAE):
    - Shared Encoder compresses x -> z
    - Decoder reconstructs x_hat from z
    - Classifier Head takes [x, z, (x - x_hat)^2, ||x - x_hat||_2] to predict fraud logit
    """

    def __init__(self, input_dim, hidden_dim=96, latent_dim=24, dropout=0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        cls_in_dim = input_dim + latent_dim + input_dim + 2
        self.classifier = nn.Sequential(
            nn.Linear(cls_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        sq_diff = (x - x_hat) ** 2
        mse_feat = torch.mean(sq_diff, dim=1, keepdim=True)
        max_feat = torch.max(sq_diff, dim=1, keepdim=True).values
        cls_input = torch.cat([x, z, sq_diff, mse_feat, max_feat], dim=1)
        logits = self.classifier(cls_input).squeeze(-1)
        return logits, x_hat


class DeepTabularFraudClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible Supervised & Semi-Supervised Deep Tabular Classifier.
    arch in:
    - 'mlp': Standard 3-layer Tabular MLP
    - 'resnet': Residual Tabular Neural Network
    - 'sup_ae': Joint Supervised Autoencoder (Reconstruction + Fraud Classification)
    loss_type in:
    - 'wbce': Weighted Binary Cross-Entropy (pos_weight)
    - 'focal': Binary Focal Loss (gamma=2.0)
    """

    def __init__(
        self,
        arch="mlp",
        loss_type="wbce",
        pos_weight=1.0,
        focal_gamma=2.0,
        recon_weight=0.35,
        hidden_dims=(128, 64, 32),
        latent_dim=24,
        dropout=0.15,
        epochs=12,
        batch_size=2048,
        lr=3e-3,
        weight_decay=1e-4,
        random_state=42,
    ):
        self.arch = arch
        self.loss_type = loss_type
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma
        self.recon_weight = recon_weight
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state

    def _compute_cls_loss(self, logits, targets):
        if self.loss_type == "wbce":
            pw = torch.tensor(self.pos_weight, dtype=torch.float32)
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        if self.loss_type == "focal":
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            probs = torch.sigmoid(logits)
            p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
            modulating = (1.0 - p_t) ** self.focal_gamma
            if self.pos_weight != 1.0:
                alpha_w = targets * self.pos_weight + (1.0 - targets)
                return torch.mean(alpha_w * modulating * bce)
            return torch.mean(modulating * bce)
        raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def fit(self, X, y):
        set_all_seeds(self.random_state)
        X_arr = np.array(X, dtype=np.float32, copy=True)
        y_arr = np.array(y, dtype=np.float32, copy=True)
        input_dim = X_arr.shape[1]

        if self.arch == "mlp":
            self.model_ = TabularMLPNet(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
                use_residual=False,
            )
        elif self.arch == "resnet":
            self.model_ = TabularMLPNet(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
                use_residual=True,
            )
        elif self.arch == "sup_ae":
            self.model_ = JointSupAENet(
                input_dim=input_dim,
                hidden_dim=self.hidden_dims[0],
                latent_dim=self.latent_dim,
                dropout=self.dropout,
            )
        else:
            raise ValueError(f"Unknown arch: {self.arch}")

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.05
        )

        n_samples = len(X_arr)
        rng = np.random.default_rng(self.random_state)

        self.model_.train()
        for _ in range(self.epochs):
            perm = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = torch.from_numpy(X_arr[idx])
                yb = torch.from_numpy(y_arr[idx])

                optimizer.zero_grad(set_to_none=True)
                if self.arch == "sup_ae":
                    logits, x_hat = self.model_(xb)
                    cls_loss = self._compute_cls_loss(logits, yb)
                    normal_mask = (yb == 0.0).float().unsqueeze(1)
                    rec_loss = torch.mean(normal_mask * ((xb - x_hat) ** 2))
                    loss = cls_loss + self.recon_weight * rec_loss
                else:
                    logits = self.model_(xb)
                    loss = self._compute_cls_loss(logits, yb)

                loss.backward()
                optimizer.step()
            scheduler.step()

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X_arr = np.array(X, dtype=np.float32, copy=True)
        self.model_.eval()
        n = len(X_arr)
        p1 = np.empty(n, dtype=np.float64)
        eval_bs = 8192

        with torch.no_grad():
            for start in range(0, n, eval_bs):
                end = min(start + eval_bs, n)
                xb = torch.from_numpy(X_arr[start:end])
                if self.arch == "sup_ae":
                    logits, _ = self.model_(xb)
                else:
                    logits = self.model_(xb)
                p1[start:end] = torch.sigmoid(logits).numpy()

        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_anomaly_enriched_preprocess_fn(
    use_iforest=True,
    use_autoencoder=True,
    ae_latent_dim=8,
    ae_epochs=8,
    if_estimators=150,
    if_max_samples=2048,
    random_state=42,
):
    """
    Higher-order preprocess_fn for validation_pipeline.run_cv_pipeline().
    Inside each CV fold (strictly fit on X_train, y_train):
    1. Standardizes and One-Hot encodes base features.
    2. Fits IsolationForest (unsupervised & normal-only) to extract global isolation scores.
    3. Fits Denoising Autoencoder on X_train[y_train == 0] to extract reconstruction errors
       (MSE, MAE, MaxSpike, Core Num+Flag MSE, Cosine Distance) and Latent Bottleneck features.
    4. Appends all extracted anomaly meta-features to X_train and X_val for GBDT models.
    """
    base_prep = make_dl_preprocess_fn(
        log_transform_skewed=True,
        scale_numeric=True,
        encode_mode="onehot",
    )

    def preprocess_fn(X_train, y_train, X_val):
        X_tr_df, y_tr_arr, X_va_df = base_prep(X_train, y_train, X_val)
        X_tr_arr = X_tr_df.to_numpy(dtype=np.float32)
        X_va_arr = X_va_df.to_numpy(dtype=np.float32)

        extra_tr = []
        extra_va = []
        extra_cols = []

        if use_iforest:
            # 1. Normal-only Isolation Forest score
            if_norm = IsolationForestFraudDetector(
                n_estimators=if_estimators,
                max_samples=if_max_samples,
                fit_mode="normal_only",
                random_state=random_state,
                n_jobs=-1,
            )
            if_norm.fit(X_tr_arr, y_tr_arr)
            tr_if_norm = if_norm.raw_anomaly_score(X_tr_arr).astype(np.float32)
            va_if_norm = if_norm.raw_anomaly_score(X_va_arr).astype(np.float32)

            # 2. Unsupervised Isolation Forest score
            if_unsup = IsolationForestFraudDetector(
                n_estimators=if_estimators,
                max_samples=256,
                fit_mode="unsupervised",
                random_state=random_state + 7,
                n_jobs=-1,
            )
            if_unsup.fit(X_tr_arr)
            tr_if_unsup = if_unsup.raw_anomaly_score(X_tr_arr).astype(np.float32)
            va_if_unsup = if_unsup.raw_anomaly_score(X_va_arr).astype(np.float32)

            extra_tr.extend([tr_if_norm[:, None], tr_if_unsup[:, None]])
            extra_va.extend([va_if_norm[:, None], va_if_unsup[:, None]])
            extra_cols.extend(["iforest_score_normal", "iforest_score_unsup"])

        if use_autoencoder:
            ae = AutoencoderFraudDetector(
                hidden_dims=(64, 32),
                latent_dim=ae_latent_dim,
                dropout=0.05,
                noise_std=0.10,
                score_mode="multi",
                epochs=ae_epochs,
                batch_size=2048,
                lr=3e-3,
                fit_normal_only=True,
                random_state=random_state,
            )
            ae.fit(X_tr_arr, y_tr_arr)
            tr_det = ae.transform_anomaly_features(X_tr_arr)
            va_det = ae.transform_anomaly_features(X_va_arr)

            tr_multi = ae.predict_proba(X_tr_arr)[:, 1].astype(np.float32)
            va_multi = ae.predict_proba(X_va_arr)[:, 1].astype(np.float32)

            extra_tr.extend([
                tr_det["mse"][:, None],
                tr_det["mae"][:, None],
                tr_det["max"][:, None],
                tr_det["num_flag_mse"][:, None],
                tr_det["cos"][:, None],
                tr_multi[:, None],
                tr_det["latents"],
            ])
            extra_va.extend([
                va_det["mse"][:, None],
                va_det["mae"][:, None],
                va_det["max"][:, None],
                va_det["num_flag_mse"][:, None],
                va_det["cos"][:, None],
                va_multi[:, None],
                va_det["latents"],
            ])
            extra_cols.extend([
                "ae_recon_mse",
                "ae_recon_mae",
                "ae_recon_max",
                "ae_recon_core_mse",
                "ae_cos_dist",
                "ae_multi_prob",
            ] + [f"ae_latent_{i}" for i in range(ae_latent_dim)])

        if extra_tr:
            X_tr_full = np.hstack([X_tr_arr] + extra_tr).astype(np.float32)
            X_va_full = np.hstack([X_va_arr] + extra_va).astype(np.float32)
            all_cols = list(X_tr_df.columns) + extra_cols
            X_tr_out = pd.DataFrame(X_tr_full, columns=all_cols)
            X_va_out = pd.DataFrame(X_va_full, columns=all_cols, index=X_val.index)
            return X_tr_out, y_tr_arr, X_va_out

        return X_tr_df, y_tr_arr, X_va_df

    return preprocess_fn
