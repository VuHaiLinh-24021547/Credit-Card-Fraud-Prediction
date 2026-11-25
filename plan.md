# Kế Hoạch Chi Tiết Triển Khai Xây Dựng Mô Hình Phức Tập: Anomaly Detection & Deep Learning (AI Implementation Plan)

> **Mục tiêu tài liệu:** Đây là bản thiết kế kỹ thuật chi tiết (Specification & Execution Blueprint) dành cho AI Agent / Developer để triển khai thực nghiệm **Xây dựng các mô hình phức tạp — Anomaly Detection / Deep Learning (Autoencoder & Isolation Forest)** trong dự án **Credit Card Fraud Prediction**. Toàn bộ mã nguồn phải tuân thủ tuyệt đối kiến trúc hiện có của repo (`validation_pipeline.py`, `imbalance_pipeline.py`, `requirement.md`).

---

## 0. Quản Lý Nhánh Git & Quy Tắc Dự Án (Git Branching & Repo Rules)

AI Agent trước khi thực hiện bất kỳ thay đổi hay tạo file mới nào **bắt buộc** phải kiểm tra và làm việc trên nhánh riêng:

1. **Tên nhánh làm việc:** `feat/anomaly-deep-learning`
2. **Lệnh khởi tạo / chuyển nhánh:**
   ```bash
   git checkout -b feat/anomaly-deep-learning
   ```
   *(Nếu nhánh đã tồn tại: `git checkout feat/anomaly-deep-learning`)*
3. **Quy tắc Commit & File không đẩy lên Git:**
   - Tuân thủ `.gitignore`: Không commit `train_folds.csv`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, các file trọng số nhị phân tạm (`.pt`, `.pth`, `.onnx`).
   - Tuân thủ `requirement.md`: Mọi module Python dùng chung được sinh ra từ Jupyter Notebook phải sử dụng cell magic `%%writefile <ten_module>.py`.
   - Các file đầu ra cần theo dõi trên nhánh `feat/anomaly-deep-learning`:
     - `plan.md` (Tài liệu kế hoạch chi tiết này)
     - `anomaly_detection_dl.ipynb` (Notebook thực nghiệm đầy đủ: Isolation Forest, Deep Autoencoder, Supervised MLP/ResNet-like Tabular DL, và Hybrid Anomaly-Enriched GBDT; có đầy đủ output và biểu đồ)
     - `anomaly_dl_pipeline.py` (Shared module xuất tự động qua `%%writefile anomaly_dl_pipeline.py` từ notebook để tái sử dụng cho bước Ensembling cuối cùng)

---

## 1. Bối Cảnh Dữ Liệu, Baseline & Ràng Buộc Kỹ Thuật (Dataset, Baselines & Constraints)

### 1.1. Đặc điểm tập dữ liệu (`train.csv` / `train_folds.csv`)
- **Kích thước:** `699,960` dòng $\times$ `26` cột (`train_folds.csv` có thêm cột `fold` $\in \{0, 1, 2, 3, 4\}$, tổng cộng 27 cột).
- **Phân phối nhãn mục tiêu (`is_fraud`):**
  - Lớp bình thường (`is_fraud = 0` - Non-Fraud): **688,271** mẫu (**98.330%**). Mỗi train fold (4/5 dữ liệu) có $\approx 550,617$ mẫu âm.
  - Lớp gian lận / bất thường (`is_fraud = 1` - Fraud / Anomaly): **11,689** mẫu (**1.670%**). Mỗi train fold (4/5 dữ liệu) có $\approx 9,351$ mẫu dương.
  - **Tỷ lệ nhiễm bẩn lý thuyết (Theoretical Contamination Rate):** $\gamma = \frac{11,689}{699,960} \approx 0.01670$.
- **Danh mục 24 Đặc trưng gốc (`ALL_FEATURES` kế thừa từ `imbalance_pipeline.py`):**
  - `NUM_COLS` (13 cột số liên tục/rời rạc): `amount_usd`, `hours_since_last_txn`, `txn_count_last_24h`, `distance_from_home_km`, `card_age_months`, `customer_age`, `account_balance_usd`, `cvv_retry_count`, `velocity_score`, `time_of_day_hour`, `day_of_week`, `merchant_risk_score`, `prior_disputes`.
  - `BINARY_FLAGS` (6 cột cờ nhị phân $0/1$): `is_foreign_transaction`, `is_new_merchant`, `used_vpn`, `ip_country_mismatch`, `billing_shipping_mismatch`, `is_ai_generated_scam_attempt`.
  - `CAT_COLS` (5 cột phân loại chuỗi): `merchant_category`, `card_type`, `auth_method`, `channel`, `device_type` ($\rightarrow$ sau `OneHotEncoder` tạo ra 35 cột nhị phân, tổng số chiều đầu vào $D = 13 + 6 + 35 = 54$ đặc trưng).

### 1.2. Mốc Tham Chiếu (Benchmarks từ các giai đoạn trước)
Để đánh giá khách quan sức mạnh của các mô hình Anomaly Detection & Deep Learning, mọi kết quả trong notebook mới phải được đặt cạnh các mốc chuẩn đã thiết lập trên cùng 5-Fold CV (`train_folds.csv`):
1. **Linear Baseline (`validation_pipeline.ipynb` — Logistic Regression):**
   - `OOF_AUPRC = 0.1909` | `OOF_ROC_AUC = 0.8705` | `Best_F1 = 0.2563`
2. **Unweighted GBDT Baseline (`imbalance_handling.ipynb` — LightGBM `spw=1.0`):**
   - `OOF_AUPRC = 0.4978` | `OOF_ROC_AUC = 0.9455` | `Best_F1 = 0.4919`
3. **Best Tree + Class Weight (`imbalance_handling.ipynb` — CatBoost `SqrtBalanced`):**
   - `OOF_AUPRC = 0.5065` | `OOF_ROC_AUC = 0.9454` | `Best_F1 = 0.4965`

### 1.3. Ràng buộc Tích hợp với `validation_pipeline.py`
Hàm `run_cv_pipeline(model_fn, feature_cols, data_path="train_folds.csv", n_splits=5, preprocess_fn=None)` trong `validation_pipeline.py` yêu cầu đối tượng trả về từ `model_fn()` phải tuân thủ giao diện Scikit-Learn chuẩn:
```python
model = model_fn()
model.fit(X_train, y_train)
val_preds = model.predict_proba(X_val)[:, 1]
```
Do đó, mọi mô hình **Isolation Forest**, **Autoencoder**, hoặc **Hybrid Model** đều phải được bọc (wrap) trong các class tương thích chuẩn `BaseEstimator, ClassifierMixin` có phương thức `.fit(X, y)` và `.predict_proba(X)` trả về ma trận $(N, 2)$ với cột `[:, 1]` là **điểm bất thường đã chuẩn hóa về $[0, 1]$ (calibrated anomaly score)** hoặc **xác suất gian lận dự đoán**.

---

## 2. Cơ Sở Lý Thuyết & Chiến Lược Triển Khai Anomaly Detection / Deep Learning Trên Dữ Liệu Bảng

### 2.1. Hai cách tiếp cận cốt lõi với Anomaly Detection
1. **Unsupervised / Semi-Supervised Anomaly Detection (Phát hiện bất thường độc lập):**
   - **Isolation Forest (Unsupervised):** Xây dựng rừng cây cô lập ngẫu nhiên trên toàn bộ `X_train` (hoặc chỉ trên tập mẫu bình thường `X_train[y_train == 0]` - *One-Class / Semi-Supervised Isolation Forest*). Các điểm gian lận có đặc tính khác biệt sẽ bị cô lập ở độ sâu đường đi trung bình $E(h(x))$ ngắn hơn.
   - **Deep Autoencoder (Semi-Supervised One-Class Learning):** Chỉ huấn luyện mạng Autoencoder nén và tái tạo trên các giao dịch **Non-Fraud (`y_train == 0`)** bên trong mỗi fold:
     $$\mathcal{L}_{\text{AE}}(x) = \|x - \hat{x}\|_2^2 = \frac{1}{D}\sum_{j=1}^{D}(x_j - g_\phi(f_\theta(x))_j)^2$$
     Khi đưa giao dịch gian lận (`y_val == 1`) qua mạng, vì mạng chưa từng học quy luật của gian lận nên **Sai số tái tạo (Reconstruction Error)** sẽ cao đột biến.
2. **Hybrid Anomaly-Enriched Supervised Learning (Trích xuất đặc trưng bất thường & Biểu diễn ẩn cho GBDT / Deep MLP):**
   - Trong các bài toán dữ liệu bảng thực tế (Tabular Fraud Detection), các thuật toán Unsupervised thuần túy (như Isolation Forest hay Autoencoder độc lập) thường chỉ đạt AUPRC ở mức trung bình ($0.10 - 0.35$) vì ranh giới gian lận phụ thuộc phi tuyến phức tạp vào nhãn giám sát.
   - Tuy nhiên, khi trích xuất **Anomaly Score của Isolation Forest**, **Reconstruction Error theo từng nhóm biến của Autoencoder**, và **Latent Bottleneck Representation ($z = f_\theta(x)$)** làm đặc trưng đầu vào bổ sung (Meta-Features) cho mô hình cây (**LightGBM / CatBoost**) hoặc kết hợp huấn luyện **Supervised Tabular Deep Learning (MLP / ResNet-Tabular / Supervised Autoencoder)**, hiệu năng tổng thể (`OOF_AUPRC` & `Best_F1`) sẽ đạt bước nhảy vọt và tạo nguồn dự đoán đa dạng cao cho bước Ensemble.

### 2.2. Nguyên tắc Chống Rò Rỉ Dữ Liệu Tuyệt Đối (Strict Zero Data Leakage)
Bên trong mỗi Fold $k \in \{0, 1, 2, 3, 4\}$:
1. **Biến đổi phân phối & Chuẩn hóa (`QuantileTransformer` / `RobustScaler` + `StandardScaler`):**
   - Các biến số như `amount_usd`, `distance_from_home_km`, `account_balance_usd` có phân phối lệch phải (right-skewed) với đuôi rất dài. Nếu đưa trực tiếp vào Autoencoder (hàm mất mát MSE), một vài biến có độ lệch lớn sẽ thống trị hoàn toàn Reconstruction Error.
   - **Giải pháp:** Áp dụng biến đổi `np.log1p` cho các biến dương lệch phải kết hợp `StandardScaler` (hoặc `RobustScaler`) **chỉ `.fit()` trên `X_train`**, sau đó `.transform()` trên `X_train` và `X_val`.
2. **Huấn luyện Semi-Supervised Autoencoder:**
   - Bên trong hàm `.fit(X_train, y_train)` của fold hiện tại, lọc riêng tập `X_train_normal = X_train[y_train == 0]` để huấn luyện Autoencoder. Tuyệt đối không dùng bất kỳ thông tin nào từ `X_val` hay `y_val`.
3. **Chuẩn hóa Điểm Bất Thường về $[0, 1]$ (`MinMaxScaler` / Sigmoid Calibration):**
   - Bộ biến đổi điểm bất thường $s(x) \in \mathbb{R} \mapsto \hat{p}(x) \in [0, 1]$ chỉ được khớp tham số (min/max hoặc trung bình/độ lệch chuẩn) trên `X_train`, sau đó áp dụng lên `X_val` để đảm bảo phân phối điểm dự đoán giữa các fold đồng nhất khi ghép thành vector `oof_predictions` 699,960 phần tử.

---

## 3. Kiến Trúc Chi Tiết Các Nhóm Mô Hình Trong `anomaly_dl_pipeline.py`

### 3.1. Nhóm 1: Mô hình Cô lập Bất thường Dạng Cây — `IsolationForestFraudDetector`
Xây dựng class `IsolationForestFraudDetector(BaseEstimator, ClassifierMixin)` bọc quanh `sklearn.ensemble.IsolationForest`:
- **Cơ chế tính điểm bất thường:**
  - `IsolationForest.score_samples(X)` trả về giá trị âm của điểm bất thường (càng âm càng bất thường).
  - Đảo dấu $s(x) = -\text{score\_samples}(X)$ để điểm càng cao $\Rightarrow$ khả năng gian lận càng lớn.
  - Chuẩn hóa $s(x)$ về đoạn $[0, 1]$ bằng `MinMaxScaler` (hoặc biến đổi Logistic Sigmoid trên Z-score: $\hat{p}(x) = \sigma\left(\frac{s(x) - \mu_{\text{tr}}}{\sigma_{\text{tr}}}\right)$) khớp trên tập train của fold.
- **Hai chế độ huấn luyện (`fit_mode`):**
  1. `fit_mode="unsupervised"` (Standard): Huấn luyện Isolation Forest trên toàn bộ `X_train` (bao gồm cả $98.33\%$ Non-Fraud và $1.67\%$ Fraud).
  2. `fit_mode="normal_only"` (Semi-Supervised / One-Class Novelty Detection): Chỉ huấn luyện cây cô lập trên `X_train[y_train == 0]` để các cây học thuần túy cấu trúc phân bố của giao dịch hợp lệ, giúp điểm cô lập của giao dịch gian lận sắc nét hơn.

### 3.2. Nhóm 2: Mạng Học Sâu Tự Mã Hóa — `AutoencoderFraudDetector` (Deep Autoencoder & Denoising / Variational Variants)
Sử dụng **PyTorch** (hoặc cài đặt `torch` vào `.venv`; hỗ trợ chạy cực nhanh với Mini-batch SGD + `AdamW` + Cosine/ReduceLROnPlateau Scheduler trên float32):
- **Kiến trúc Mạng Autoencoder Đa tầng (Tabular Deep Autoencoder):**
  - Đầu vào: Vector đặc trưng đã chuẩn hóa $x \in \mathbb{R}^{54}$.
  - **Encoder ($f_\theta$):**
    $$\text{Linear}(54 \to 64) \to \text{BatchNorm1d} \to \text{SiLU/GELU} \to \text{Dropout}(p) \to \text{Linear}(64 \to 32) \to \text{BatchNorm1d} \to \text{SiLU/GELU} \to \text{Linear}(32 \to d_{\text{latent}})$$
    với số chiều nút cổ chai (Bottleneck dimension) $d_{\text{latent}} \in \{8, 16\}$.
  - **Decoder ($g_\phi$):**
    $$\text{Linear}(d_{\text{latent}} \to 32) \to \text{BatchNorm1d} \to \text{SiLU/GELU} \to \text{Linear}(32 \to 64) \to \text{BatchNorm1d} \to \text{SiLU/GELU} \to \text{Linear}(64 \to 54)$$
- **Các biến thể Autoencoder cần khảo sát:**
  1. **Standard Undercomplete Autoencoder (`AE-Standard`):** Huấn luyện trên `X_train[y_train == 0]` với hàm mất mát MSE/SmoothL1.
  2. **Denoising Autoencoder (`DAE`):** Cộng nhiễu Gaussian $\epsilon \sim \mathcal{N}(0, \sigma_{\text{noise}}^2 I)$ hoặc Masking Noise vào đầu vào $\tilde{x} = x + \epsilon$ và buộc mạng khôi phục lại $x$ sạch ban đầu, giúp chống học vẹt (identity mapping) và làm mượt đa tạp (manifold) của lớp giao dịch hợp lệ.
  3. **Feature-Weighted / Hard-Mining Reconstruction Score (`AE-Weighted`):**
     - Thay vì lấy trung bình cộng đồng đều sai số tái tạo trên cả 54 cột (nơi các cột One-Hot ít quan trọng có thể làm loãng tín hiệu), gán trọng số sai số tái tạo theo mức độ phân biệt hoặc tách riêng:
       - $e_{\text{mse}}(x) = \frac{1}{D}\sum_{j=1}^D (x_j - \hat{x}_j)^2$ (Mean Squared Reconstruction Error)
       - $e_{\text{mae}}(x) = \frac{1}{D}\sum_{j=1}^D |x_j - \hat{x}_j|$ (Mean Absolute Reconstruction Error)
       - $e_{\text{max}}(x) = \max_{1 \le j \le D} (x_j - \hat{x}_j)^2$ (Maximum Feature Anomaly Spike)
       - $e_{\text{cos}}(x) = 1 - \frac{x \cdot \hat{x}}{\|x\|_2 \|\hat{x}\|_2}$ (Cosine Reconstruction Distance)

### 3.3. Nhóm 3: Mạng Học Sâu Có Giám Sát & Bán Giám Sát — `DeepTabularFraudClassifier` & `HybridAutoencoderMLP`
Bên cạnh Autoencoder thuần túy không dùng nhãn `y_train == 1`, triển khai thêm 2 kiến trúc Deep Learning khai thác trực tiếp tín hiệu giám sát:
1. **Dual-Loss Supervised Autoencoder (`SupAE` / Joint Reconstruction + Classification Network):**
   - Mạng có chung nhánh `Encoder` $z = f_\theta(x)$, sau đó rẽ làm 2 đầu ra:
     - **Nhánh 1 (`Decoder` $g_\phi(z)$):** Tái tạo lại $\hat{x}$ để giữ cấu trúc hình học của dữ liệu.
     - **Nhánh 2 (`Classifier Head` $h_\psi([z, x - \hat{x}, \|x - \hat{x}\|_2^2])$):** Nhận đồng thời vector ẩn $z$, vector hiệu số tái tạo $(x - \hat{x})$ và tổng sai số tái tạo để dự đoán xác suất gian lận $\hat{y} = \sigma(\text{logit})$ thông qua hàm mất mát kết hợp:
       $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE / Focal}}(y, \hat{y}) + \lambda_{\text{rec}} \cdot \mathbb{I}(y=0) \cdot \|x - \hat{x}\|_2^2$$
2. **Residual Tabular Neural Network (`ResNet-Tabular` / Deep MLP với Focal Loss / Weighted BCE):**
   - Kiến trúc mạng nơ-ron sâu cho dữ liệu bảng với kết nối tắt (Skip/Residual Connections: $h_{l+1} = h_l + \text{Block}(h_l)$), `BatchNorm1d`, `SiLU`, `Dropout`, huấn luyện với `pos_weight = sqrt(IR)` hoặc **Focal Loss** ($\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$) để tập trung vào các giao dịch gian lận khó phân loại.

### 3.4. Nhóm 4: Kiến Trúc Lai — `AnomalyEnrichedGBDT` (Feature Engineering bằng Isolation Forest + Autoencoder cho GBDT)
Xây dựng bộ trích xuất đặc trưng bất thường trong từng fold (`make_anomaly_enriched_preprocess_fn`):
- Bên trong mỗi fold:
  1. Fit `IsolationForest` trên `X_train` $\rightarrow$ trích xuất `iforest_score_all` và `iforest_score_normal`.
  2. Fit `Autoencoder` trên `X_train[y_train == 0]` $\rightarrow$ trích xuất bộ đặc trưng bất thường:
     - `ae_recon_mse`: Sai số tái tạo bậc hai trung bình toàn cục.
     - `ae_recon_mae`: Sai số tái tạo tuyệt đối trung bình.
     - `ae_recon_max`: Đỉnh sai số tái tạo lớn nhất trong các đặc trưng của giao dịch.
     - `ae_recon_num_mse` & `ae_recon_flag_mse`: Sai số tái tạo tính riêng trên nhóm biến số liên tục và nhóm cờ nhị phân rủi ro cao.
     - `ae_cos_dist`: Khoảng cách Cosine giữa vector gốc và vector tái tạo.
     - `ae_latent_0 ... ae_latent_k`: Các chiều biểu diễn nén tại nút cổ chai (Bottleneck features).
  3. Ghép các đặc trưng mới này vào bảng đặc trưng gốc của `X_train` và `X_val`, sau đó huấn luyện **LightGBM** và **CatBoost (`SqrtBalanced`)**.

---

## 4. Ma Trận Thực Nghiệm Chi Tiết (Experimental Matrix)

Toàn bộ thực nghiệm được đánh giá bằng `run_cv_pipeline` trên 5 folds của `train_folds.csv`:

### 4.1. Giai đoạn 1: Thực nghiệm Isolation Forest (EXP-IF)

| Mã TN | Cấu hình Mô hình | Chi tiết Kỹ thuật | Mục tiêu Khoa học |
| :--- | :--- | :--- | :--- |
| **EXP-IF.1** | `IsolationForest` (Unsupervised Default) | `n_estimators=200, max_samples=256, contamination=0.0167, fit_mode="unsupervised"` | Đánh giá khả năng phát hiện gian lận hoàn toàn không giám sát với `max_samples=256` kinh điển của bài báo gốc Liu et al. |
| **EXP-IF.2** | `IsolationForest` (Large Subsample) | `n_estimators=300, max_samples=4096, contamination=0.0167, fit_mode="unsupervised"` | Kiểm tra xem việc tăng kích thước mẫu con mỗi cây (`4096` trên `560k` dòng) có giúp bắt được các cụm gian lận tinh vi hơn không. |
| **EXP-IF.3** | `IsolationForest` (Semi-Supervised Normal-Only) | `n_estimators=300, max_samples=4096, fit_mode="normal_only"` | Chỉ dựng cây cô lập trên giao dịch `Non-Fraud (y=0)` để loại bỏ hiện tượng "masking/swamping" do mẫu gian lận gây ra khi chia cây. |
| **EXP-IF.4** | `IsolationForest` (Feature Subspace) | `n_estimators=300, max_samples=4096, max_features=0.7, fit_mode="normal_only"` | Kết hợp lấy mẫu con đặc trưng (`max_features=0.7`) để tăng tính đa dạng giữa các cây cô lập trong không gian 54 chiều. |

### 4.2. Giai đoạn 2: Thực nghiệm Deep Autoencoder — Anomaly Detection (EXP-AE)

| Mã TN | Cấu hình Mô hình | Chi tiết Kỹ thuật | Mục tiêu Khoa học |
| :--- | :--- | :--- | :--- |
| **EXP-AE.1** | `Standard Autoencoder` ($d_{\text{latent}}=16$) | Huấn luyện trên `y_train==0`, kiến trúc `54 -> 64 -> 32 -> 16 -> 32 -> 64 -> 54`, Loss MSE | Đo lường khả năng phát hiện bất thường qua sai số tái tạo MSE với nút cổ chai 16 chiều. |
| **EXP-AE.2** | `Tight-Bottleneck Autoencoder` ($d_{\text{latent}}=8$) | Huấn luyện trên `y_train==0`, nén mạnh xuống `8` chiều ẩn | Kiểm tra giả thuyết: nén chặt hơn buộc mạng chỉ giữ lại các tương quan phổ biến nhất của giao dịch hợp lệ, làm lộ rõ giao dịch gian lận hơn. |
| **EXP-AE.3** | `Denoising Autoencoder (DAE)` ($d_{\text{latent}}=12, \sigma=0.15$) | Cộng nhiễu Gaussian $\sigma=0.15$ vào đầu vào khi train trên `y_train==0` | Đánh giá hiệu quả của Denoising Autoencoder trong việc học biên phân phối bền vững hơn so với AE tiêu chuẩn. |
| **EXP-AE.4** | `Multi-Metric Autoencoder Anomaly Score` | Kết hợp tổ hợp chuẩn hóa của `MSE + MaxSpike + CosineDistance` từ DAE | Kiểm chứng xem việc kết hợp đỉnh sai số cục bộ (`MaxSpike`) và góc lệch vector (`Cosine`) có cải thiện AUPRC so với MSE đơn thuần không. |

### 4.3. Giai đoạn 3: Thực nghiệm Deep Learning Có Giám Sát & Bán Giám Sát (EXP-DL)

| Mã TN | Cấu hình Mô hình | Chi tiết Kỹ thuật | Mục tiêu Khoa học |
| :--- | :--- | :--- | :--- |
| **EXP-DL.1** | `Tabular MLP Baseline` (Weighted BCE) | Mạng MLP 3 tầng ẩn (`128 -> 64 -> 32`) + `BatchNorm` + `SiLU` + `Dropout(0.2)`, `pos_weight=1.0` | Thiết lập mốc Baseline cho mạng Nơ-ron sâu có giám sát trên bộ dữ liệu 700k giao dịch. |
| **EXP-DL.2** | `Tabular ResNet-MLP` (`pos_weight=sqrt(IR)`) | Kiến trúc Residual Blocks cho dữ liệu bảng + `CosineAnnealingLR` + `pos_weight=7.67` | Đánh giá tác động của kết nối Residual và trọng số $\sqrt{\text{IR}}$ lên mạng học sâu. |
| **EXP-DL.3** | `Tabular ResNet-MLP` (Focal Loss $\gamma=2.0$) | Thay BCE bằng **Focal Loss** để tự động giảm trọng số các mẫu Non-Fraud dễ đoán | So sánh trực tiếp Focal Loss với Weighted BCE trên dữ liệu mất cân bằng cực đoan ($1.67\%$). |
| **EXP-DL.4** | `Joint Supervised Autoencoder (SupAE)` | Huấn luyện đồng thời `Decoder` (MSE trên `y=0`) + `Classifier Head` (nhận `[z, |x - x_hat|, recon_errors]`) | Kiểm chứng sức mạnh của kiến trúc lai End-to-End: vừa học tái tạo bất thường vừa học phân lớp giám sát. |

### 4.4. Giai đoạn 4: Thực nghiệm Hybrid Anomaly-Enriched GBDT & Ensemble (EXP-HYB)

| Mã TN | Cấu hình Mô hình | Chi tiết Kỹ thuật | Mục tiêu Khoa học |
| :--- | :--- | :--- | :--- |
| **EXP-HYB.1** | `LightGBM + IsolationForest Features` | Bổ sung điểm bất thường của Isolation Forest vào 24 đặc trưng gốc | Đo mức cải thiện AUPRC khi cung cấp độ sâu cô lập toàn cục cho cây GBDT. |
| **EXP-HYB.2** | `LightGBM + Autoencoder Features` | Bổ sung các chỉ số sai số tái tạo (`MSE, MAE, Max, Cosine`) & `Latent Vectors` của DAE cho LightGBM | Kiểm tra giá trị gia tăng của biểu diễn Deep Autoencoder đối với mô hình cây quyết định. |
| **EXP-HYB.3** | `LightGBM + Full Anomaly Suite (IF + AE)` | Kết hợp toàn bộ đặc trưng gốc + Isolation Forest scores + Autoencoder Reconstruction & Latent features | Đánh giá sức mạnh tổng hợp của cả 2 họ Anomaly Detection trên LightGBM. |
| **EXP-HYB.4** | `CatBoost (SqrtBalanced) + Full Anomaly Suite` | Đưa bộ đặc trưng Anomaly (IF + AE) vào mô hình Top 1 hiện tại (`CatBoost SqrtBalanced`) | Kiểm tra xem các đặc trưng bất thường có đẩy kỷ lục `OOF_AUPRC` vượt mốc `0.5065` của Giai đoạn trước hay không. |
| **EXP-HYB.5** | `Cross-Paradigm Ensemble (GBDT + Deep Learning)` | Blending dự đoán OOF của **Anomaly-Enriched CatBoost/LightGBM** với **Deep Learning (`SupAE` & `ResNet-MLP`)** | Chứng minh tính bổ trợ giữa **Decision Trees** và **Deep Neural Networks**: hai họ mô hình có cơ chế phân hoạch không gian khác hẳn nhau khi kết hợp sẽ đạt `OOF_AUPRC` và `Best_F1` cao nhất dự án. |

---

## 5. Hệ Thống Đánh Giá, Bảng Tổng Hợp & Trực Quan Hóa Bắt Buộc

### 5.1. Bảng Leaderboard Tổng Hợp (`results_df`)
Mọi thực nghiệm chạy qua `run_cv_pipeline` phải ghi nhận đầy đủ các chỉ số chuẩn vào bảng tổng hợp:
- `Experiment_ID`, `Paradigm` (`Unsupervised-Tree`, `SemiSupervised-AE`, `Supervised-DL`, `Hybrid-GBDT`, `Ensemble`)
- `Model`, `Technique_Details`
- `Mean_Fold_AUPRC`, `Std_Fold_AUPRC`
- `OOF_AUPRC` *(Thước đo xếp hạng chính)*
- `OOF_ROC_AUC`
- `F1_at_0.50`, `Best_F1`, `Best_Threshold`
- `Precision_at_Best_F1`, `Recall_at_Best_F1`, `TP`, `FP`, `TN`, `FN`
- `Train_Time_Sec`

### 5.2. Các Biểu Đồ Phân Tích Chuyên Sâu Trong Notebook (`anomaly_detection_dl.ipynb`)
1. **Biểu đồ 1 (Phân phối Điểm Bất Thường & Sai Số Tái Tạo — KDE / Histogram Log-Scale):**
   - So sánh trực quan phân phối **Isolation Forest Anomaly Score** và **Autoencoder Reconstruction Error** giữa 2 lớp `Non-Fraud (0)` và `Fraud (1)` trên tập Out-of-Fold.
   - Chỉ ra vùng đuôi phân phối (extreme tail) nơi các giao dịch gian lận tập trung với sai số tái tạo cao gấp nhiều lần giao dịch thường.
2. **Biểu đồ 2 (Không gian Biểu diễn Ẩn 2D — PCA / t-SNE / UMAP của Autoencoder Bottleneck & Reconstruction Error):**
   - Trực quan hóa các điểm giao dịch trong không gian đặc trưng ẩn của Autoencoder, tô màu theo nhãn thực tế (`Non-Fraud` vs `Fraud`) và kích thước điểm theo `Reconstruction Error`.
3. **Biểu đồ 3 (Overlaid Precision-Recall & ROC Curves Across Paradigms):**
   - Vẽ đè đường cong PR và ROC của các đại diện tốt nhất từ mỗi trường phái:
     - *Isolation Forest (Best)*
     - *Deep Autoencoder Unsupervised (Best)*
     - *Supervised Deep Learning (`SupAE` / `ResNet-MLP`)*
     - *Baseline LightGBM / CatBoost*
     - *Hybrid Anomaly-Enriched GBDT*
     - *Cross-Paradigm Ensemble (Tree + Deep Learning)*
4. **Biểu đồ 4 (Feature Importance của các Đặc trưng Bất thường trong Mô hình Hybrid GBDT):**
   - Thống kê mức đóng góp (Feature Importance / Gain) của `ae_recon_mse`, `ae_recon_max`, `iforest_score`, và các chiều `ae_latent_*` khi đứng chung với 24 đặc trưng gốc, chứng minh các đặc trưng từ Autoencoder & Isolation Forest thực sự mang lại thông tin mới cho mô hình cây.

---

## 6. Đóng Gói Shared Module (`anomaly_dl_pipeline.py`) Theo `requirement.md`

Trong notebook `anomaly_detection_dl.ipynb`, sử dụng cell magic `%%writefile anomaly_dl_pipeline.py` để xuất toàn bộ các class và hàm tiện ích dùng chung cho toàn dự án:

```python
%%writefile anomaly_dl_pipeline.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, MinMaxScaler

# 1. Hàm tiền xử lý chuyên biệt cho Anomaly Detection & Deep Learning (Log1p + StandardScaler + OneHot)
def make_dl_preprocess_fn(log_transform_skewed=True, encode_mode="onehot"):
    ...

# 2. Wrapper chuẩn Scikit-Learn cho Isolation Forest (hỗ trợ cả 'unsupervised' và 'normal_only')
class IsolationForestFraudDetector(BaseEstimator, ClassifierMixin):
    ...

# 3. Kiến trúc mạng PyTorch Tabular Autoencoder & Wrapper chuẩn Scikit-Learn
class TabularAutoencoderNet(nn.Module):
    ...

class AutoencoderFraudDetector(BaseEstimator, ClassifierMixin):
    ...

# 4. Kiến trúc mạng Supervised / Joint Autoencoder-Classifier & ResNet-Tabular
class JointSupAENet(nn.Module):
    ...

class DeepTabularFraudClassifier(BaseEstimator, ClassifierMixin):
    ...

# 5. Bộ trích xuất đặc trưng bất thường (Isolation Forest + Autoencoder) bên trong từng Fold CV
def make_anomaly_enriched_preprocess_fn(...):
    ...
```

---

## 7. Checklist Các Bước Thực Thi Dành Cho AI Agent (Step-by-Step Execution)

Khi triển khai từ bản kế hoạch này, AI Agent thực hiện tuần tự các bước sau:

- [x] **Bước 1 (Kiểm tra môi trường, Nhánh Git & Thư viện Deep Learning):**
  - Chuyển sang nhánh `feat/anomaly-deep-learning` (`git checkout -b feat/anomaly-deep-learning`).
  - Kiểm tra sự tồn tại của `train_folds.csv` (nếu chưa có thì tái tạo bằng đúng `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
  - Kiểm tra `torch` (PyTorch) trong `.venv`. Nếu chưa cài đặt, tiến hành cài đặt bản nhẹ/tương thích phần cứng vào `.venv` để huấn luyện Autoencoder và các mạng Deep Learning với tốc độ tối ưu.
- [x] **Bước 2 (Xây dựng & Xuất `anomaly_dl_pipeline.py`):**
  - Viết đầy đủ các class `IsolationForestFraudDetector`, `AutoencoderFraudDetector`, `DeepTabularFraudClassifier`, và `make_anomaly_enriched_preprocess_fn` đảm bảo tương thích 100% với `validation_pipeline.run_cv_pipeline()` và không rò rỉ dữ liệu.
  - Chạy kiểm thử nhanh (Sanity Check) trên 1 fold nhỏ để xác nhận đầu ra `.predict_proba(X_val)[:, 1]` nằm trong $[0, 1]$, không có `NaN/Inf`.
- [x] **Bước 3 (Thực thi Giai đoạn 1 & 2 — Isolation Forest & Deep Autoencoder):**
  - Chạy chuỗi thực nghiệm `EXP-IF.1` $\rightarrow$ `EXP-IF.4` và `EXP-AE.1` $\rightarrow$ `EXP-AE.4` qua 5-Fold CV.
  - Phân tích sự khác biệt giữa huấn luyện trên toàn tập (`unsupervised`) và chỉ huấn luyện trên giao dịch hợp lệ (`normal_only`).
- [x] **Bước 4 (Thực thi Giai đoạn 3 & 4 — Supervised Deep Learning, Hybrid Anomaly-Enriched GBDT & Ensemble):**
  - Chạy chuỗi thực nghiệm `EXP-DL.1` $\rightarrow$ `EXP-DL.4` và `EXP-HYB.1` $\rightarrow$ `EXP-HYB.5`.
  - Đánh giá mức tăng trưởng `OOF_AUPRC` và `Best_F1` khi kết hợp đặc trưng Anomaly Detection và khi Ensemble giữa GBDT với Deep Learning.
- [x] **Bước 5 (Tổng hợp Leaderboard, Trực quan hóa 4 Biểu đồ & Viết Kết luận):**
  - Hoàn thiện `anomaly_detection_dl.ipynb` với đầy đủ bảng xếp hạng, 4 biểu đồ phân tích chuyên sâu và khuyến nghị chiến lược cho giai đoạn Ensembling cuối cùng của dự án.
