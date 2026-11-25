# Báo Cáo Phân Tích Lỗi (Error Analysis) & Khuyến Nghị Feature Engineering Cho Member 1

* **Người thực hiện:** Member 3 (Pipeline, Validation & Post-Processing Lead)
* **Dữ liệu đánh giá:** `train_folds.csv` (`699,960` giao dịch, Stratified 5-Fold CV cố định `seed=42`, chống rò rỉ dữ liệu 100%)
* **Mô hình đối chiếu (24 biến gốc):** `CatBoost (SqrtBalanced)` (`OOF AUPRC = 0.5062`, `Best F1 = 0.4966` @ ngưỡng `0.6845`) & `LightGBM` (`OOF AUPRC = 0.4995`, `Best F1 = 0.4927` @ ngưỡng `0.2362`)

---

## 1. Bức tranh tổng quan về tập lỗi (`FP` & `FN`)

Ngay cả trên các mô hình GBDT tốt nhất với toàn bộ 24 biến gốc, hiệu năng hiện đang chạm trần quanh `AUPRC ≈ 0.506`:
* **False Negatives (`FN` — Bỏ sót gian lận):** `CatBoost` bỏ sót **`6,363 / 11,689` ca gian lận (`54.44%`)**; trong đó có tới **`6,015` ca gian lận cứng đầu (`51.46%`) cả `CatBoost` lẫn `LightGBM` đều bỏ sót (`Hard FN`)**.
* **False Positives (`FP` — Báo động giả):** `CatBoost` báo nhầm **`4,436` giao dịch hợp lệ** (`Precision = 54.56%`), trong đó **`3,818` ca cả hai mô hình cùng báo nhầm (`Hard FP`)**.
* **Nguyên nhân cốt lõi từ phân tích `SMD` (Standardized Mean Difference):** Khi xét đơn biến từng cột số liên tục giữa **`TP` vs `FP`**, độ lệch chuẩn hóa gần như bằng `0` (`|SMD| < 0.09` ở hầu hết các biến). Cây quyết định độ sâu vừa phải chia tách từng biến độc lập nên không gom được tín hiệu cộng hưởng yếu và dễ phạt oan khách thật khi có 1 cờ rủi ro đơn lẻ.

---

## 2. Mổ xẻ sâu 4 Lỗ hổng chính gây ra `FN` và `FP`

### Lỗ hổng 1 — "Gian lận ẩn mình không bật cờ" (`Zero/Single-Flag Stealth Fraud` -> Gây 91.3% tổng số `FN`)
* Khi `flag_count = 0` (`549,155` giao dịch, chứa `3,870` ca Fraud): **Tỷ lệ bỏ sót (`FNR`) lên tới `79.53%`** (`3,078 FN`).
* Khi `flag_count = 1` (`137,966` giao dịch, chứa `4,981` ca Fraud): **Tỷ lệ bỏ sót (`FNR`) là `54.79%`** (`2,729 FN`).
* Cộng gộp lại, vùng `0–1` cờ chiếm tới **`5,807 / 6,363` (`91.26%`) toàn bộ số ca bỏ sót gian lận!**
* Tuy không bật cờ nhị phân và `cvv_retry_count = 0` (`75.64%` FNR), nhóm `FN` này vẫn có **sự gia tăng đồng thời mức độ vừa phải** trên nhiều biến liên tục so với giao dịch sạch (`TN`):
  * `velocity_score` trung bình: `FN = 25.39` vs `TN = 18.72`
  * `merchant_risk_score` trung bình: `FN = 44.44` vs `TN = 34.91`
  * `customer_age` trung bình: `FN = 57.55` vs `TN = 51.24`
  * `txn_count_last_24h` trung bình: `FN = 3.49` vs `TN = 2.97`

### Lỗ hổng 2 — Bẫy báo động giả do "1 lỗi gõ CVV" hoặc "Không xác thực nhưng khách thật" (Gây >70% `FP`)
* **`cvv_retry_count = 1`:** Gây ra **`2,496 / 4,436` ca `FP` (`56.27%` tổng số báo động giả!)** với tỷ lệ báo nhầm `FDR = 44.33%` — rất nhiều khách hàng hợp lệ chỉ gõ nhầm CVV đúng 1 lần bị mô hình gắn nhãn gian lận.
* **`auth_method = No Authentication`:** Gây ra **`1,972` ca `FP` (`44.45%` tổng `FP`)**. Ngược lại, ở các phương thức xác thực mạnh (`Biometric`: `FNR = 70.73%`, `PIN`: `FNR = 67.91%`, `OTP`: `FNR = 62.22%`, `3D Secure`: `FNR = 61.68%`), mô hình lại chủ quan hạ điểm quá thấp dẫn tới bỏ sót `4,683` ca `FN` (ví dụ lừa đảo chiếm đoạt / lừa nạn nhân tự xác thực OTP/Biometric).

### Lỗ hổng 3 — Bỏ sót biến `customer_age` trong EDA ban đầu & Tương tác lừa đảo người cao tuổi
* Trong `eda.ipynb`, biến **`customer_age`** (cùng `card_age_months`, `prior_disputes`) chưa được đưa vào danh sách `num_cols` phân tích phân phối và tương quan.
* Thực tế trên GBDT, **`customer_age` đứng Top 7 biến quan trọng nhất toàn tập dữ liệu**:
  * Tỷ lệ gian lận tăng gấp **7 lần** theo độ tuổi: từ **`0.407%`** (nhóm `18–25` tuổi) lên **`2.875%`** (nhóm `69–76` tuổi).
  * Khi khách hàng `>= 66` tuổi gặp `is_ai_generated_scam_attempt = True`, tỷ lệ gian lận vọt lên **`21.15%`** (so với `5.04%` ở người trẻ `<= 25`). Khi `>= 66` tuổi + `No Authentication`, tỷ lệ gian lận đạt **`9.95%`**.

### Lỗ hổng 4 — Đường cong chữ U phi tuyến của `amount_usd` & Nhiễu từ 7 biến phẳng
* **Đường cong chữ U trên `amount_usd`:** Tương quan tuyến tính Pearson của `amount_usd` rất thấp (`r = 0.0194`) vì rủi ro gian lận có **dạng chữ U**:
  * Giao dịch siêu nhỏ `< $9.22` (thử thẻ / *card-testing*): **`1.634%` Fraud**.
  * Vùng chi tiêu thường nhật `$27.25 – $32.48`: Chạm đáy an toàn **`1.196%` Fraud**.
  * Giao dịch giá trị lớn `> $89.80`: Tăng vọt lên **`2.920%` Fraud**.
* **7 biến nhiễu có phân phối gian lận phẳng lì (gây chia nhánh nhiễu tạo `FP`):**
  * `distance_from_home_km` (`1.57% – 1.76%` phẳng trên mọi decile), `account_balance_usd` (`1.61% – 1.89%`), `card_age_months` (`1.60% – 1.83%`), `time_of_day_hour` (`1.66% – 1.69%`), `day_of_week` (`1.58% – 1.78%`), `device_type` (`1.61% – 1.73%`), `channel` (`1.55% – 1.74%`).

---

## 3. Khuyến nghị cụ thể gửi Member 1 (Đã kiểm chứng thực nghiệm tăng AUPRC trên 5-Fold CV)

Khi thử nghiệm trực tiếp trên 5-Fold `LightGBM`, bộ đặc trưng đề xuất dưới đây đẩy **OOF AUPRC tăng lên `0.5018+`**, cắt giảm hàng trăm ca `FP`, và **3 biến mới tạo chiếm luôn Top 1, Top 2, Top 3 Split Gain toàn mô hình**:

### A. Các nhóm Feature CẦN BỔ SUNG (Feature Engineering)

1. **[Ưu tiên P0] Nhóm điểm Cân bằng Rủi ro – Tin cậy (`Net Risk Minus Trust` — Top 1 Gain):**
   * `flag_count`: Tổng số cờ nhị phân đang bật (`0` đến `6`).
   * `weighted_flag_risk`: Tổng có trọng số các cờ theo mức độ nguy hiểm (`2.5*ip_country_mismatch + 2.4*is_ai_generated_scam_attempt + 2.1*billing_shipping_mismatch + 2.0*is_foreign_transaction + 1.4*used_vpn + 1.3*is_new_merchant`). *(Lưu ý: Nếu dùng trọng số học từ dữ liệu thì bắt buộc chỉ tính trên Fold Train).*
   * `auth_trust_score`: Điểm tin cậy xác thực (`Biometric: +2.5`, `PIN: +1.8`, `OTP / 3D Secure: +1.0`, `No Authentication: -2.0`).
   * `net_risk_minus_trust = weighted_flag_risk + 2.2 * cvv_retry_count + 0.5 * prior_disputes - auth_trust_score`: Giúp giảm mạnh `FP` cho khách thật có xác thực mạnh dù lỡ gõ sai CVV 1 lần hoặc đi nước ngoài.

2. **[Ưu tiên P0] Nhóm tương tác Độ tuổi nhạy cảm (`Senior Vulnerability` — Top 3 Gain):**
   * Đưa đầy đủ `customer_age` và `prior_disputes` vào pipeline.
   * `age_x_merchrisk = customer_age * merchant_risk_score`
   * `senior_x_risk = max(customer_age - 50, 0) * (weighted_flag_risk + 1.0)`

3. **[Ưu tiên P1] Nhóm Động học Giao dịch & Cộng hưởng liên tục (`Velocity Kinematics` — Top 2 Gain, bắt `FN` không cờ):**
   * `vel_x_merchrisk = velocity_score * merchant_risk_score`
   * `continuous_risk_sum = (velocity_score / 18.5) + (merchant_risk_score / 34.0) + (txn_count_last_24h / 3.0)` (tổng hợp tín hiệu cho các ca `flag_count = 0`).
   * `burst_intensity = txn_count_last_24h / (hours_since_last_txn + 0.2)` và `log_hours_since_last = log1p(hours_since_last_txn)`.

4. **[Ưu tiên P1] Nhóm biến đổi Phi tuyến chữ U cho Số tiền (`U-Shaped Amount`):**
   * `log_amount = log1p(amount_usd)` (thay cho scaling tuyến tính đơn thuần vì phân phối lệch phải rất nặng).
   * `u_amount_deviation = abs(log1p(amount_usd) - log1p(28.0))` (đo độ lệch khỏi vùng chi tiêu an toàn `$28`).
   * `is_micro_txn = (amount_usd < 10.0).astype(int)` (bắt hành vi *card-testing* giao dịch nhỏ).

### B. Các Feature CẦN LOẠI BỎ / THẬN TRỌNG (Feature Selection & Anti-Overfitting)

1. **Đề xuất cắt bỏ (hoặc kiểm tra kỹ bằng SHAP / Permutation Importance ở Phase 3) 7 biến nhiễu phẳng:**
   * `distance_from_home_km`, `account_balance_usd`, `card_age_months`, `time_of_day_hour`, `day_of_week`, `device_type`, `channel`.
   * *Bằng chứng thực nghiệm:* Chỉ riêng việc loại bỏ 7 biến này trên Fold 0 đã giúp `LightGBM` **giảm `18.5%` lượng False Positives (`1,072` xuống `874`)** và tăng ngay `+0.0033` AUPRC, `+0.0069` Best F1!
2. **KHÔNG tạo các biến ghép chéo phân loại nhiều nhãn thô (`High-Cardinality Categorical Crosses`):**
   * Ví dụ: `merchant_category + "_" + auth_method` hoặc `channel + "_" + device_type`.
   * *Bằng chứng thực nghiệm:* Làm `LightGBM` bị overfit vào các nhánh ít mẫu, kéo tụt **`-0.0103` AUPRC**!
3. **Nguyên tắc chống Data Leakage bắt buộc:**
   * Mọi phép `StandardScaler` / `RobustScaler` hay thống kê gom nhóm (`mean`, `std`, Target Encoding) **bắt buộc chỉ `.fit()` trên `fold != val_fold`** của `train_folds.csv`.
