# Góp ý Feature Engineering cho Member 1 (từ Phân tích lỗi)

**Mô hình tham chiếu:** CatBoost, 24 biến gốc, 5-Fold CV trên `train_folds.csv`
→ AUPRC = 0.506 | Best F1 = 0.497
→ Bỏ sót **6,363 / 11,689** ca gian lận (FN) | Báo nhầm **4,436** giao dịch sạch (FP)

---

## 1. Mô hình đang sai ở đâu?

| # | Vấn đề | Số liệu chính |
|---|---|---|
| 1 | **Gian lận "không bật cờ" bị bỏ sót** | 91% số FN có 0–1 cờ rủi ro. Khi không có cờ nào, mô hình bỏ sót 80% ca gian lận. Tuy vậy các ca này vẫn có `velocity_score`, `merchant_risk_score`, `txn_count_last_24h` cao hơn bình thường một chút. |
| 2 | **Khách thật bị báo nhầm vì 1 dấu hiệu đơn lẻ** | 56% số FP chỉ do nhập sai CVV 1 lần; 44% do `No Authentication`. Mô hình không "trừ điểm" cho khách đã xác thực mạnh (Biometric, PIN). |
| 3 | **Bỏ quên `customer_age`** | Chưa có trong EDA nhưng nằm Top 7 biến quan trọng. Tỷ lệ gian lận tăng từ 0.4% (≤25 tuổi) lên 2.9% (≥69 tuổi); người ≥66 tuổi gặp AI scam → 21% gian lận. |
| 4 | **`amount_usd` có dạng chữ U** | Giao dịch rất nhỏ (<$9) và rất lớn (>$90) rủi ro cao, vùng $27–32 an toàn nhất → tương quan tuyến tính gần 0, cần biến đổi lại. |
| 5 | **7 biến gần như vô dụng gây nhiễu** | `distance_from_home_km`, `account_balance_usd`, `card_age_months`, `time_of_day_hour`, `day_of_week`, `device_type`, `channel`: tỷ lệ gian lận gần như không đổi (~1.6%) ở mọi mức giá trị. |

---

## 2. Đề xuất cụ thể

### Nên thêm
| Ưu tiên | Feature | Công thức gợi ý | Sửa lỗi |
|---|---|---|---|
| P0 | Điểm rủi ro trừ độ tin cậy | `flag_count` = tổng 6 cờ; `auth_trust` (Biometric +2.5, PIN +1.8, OTP/3DS +1, No Auth −2); `net_risk = weighted_flags + 2.2*cvv_retry_count + 0.5*prior_disputes − auth_trust` | FP (#2) |
| P0 | Tương tác tuổi | `customer_age * merchant_risk_score`; `max(customer_age − 50, 0) * (weighted_flags + 1)` | FN (#3) |
| P1 | Cộng dồn tín hiệu yếu | `velocity_score * merchant_risk_score`; `velocity/18.5 + merchant_risk/34 + txn_count_24h/3`; `txn_count_24h / (hours_since_last_txn + 0.2)` | FN (#1) |
| P1 | Biến đổi số tiền | `log1p(amount)`; `abs(log1p(amount) − log1p(28))`; `amount < 10` | FN (#4) |

### Nên bỏ / tránh
- **Bỏ 7 biến nhiễu ở mục #5** → riêng Fold 0 giảm ~18% FP.
- **Không ghép chéo biến phân loại** (vd `merchant_category × auth_method`) → làm AUPRC giảm 0.01 do overfit.

---

## 3. Kết quả kiểm chứng (LightGBM, 5-Fold CV)

| Cấu hình | AUPRC | FP |
|---|---|---|
| 24 biến gốc | 0.4995 | 4,986 |
| + feature mới, bỏ 7 biến nhiễu | **0.5018** | **4,805** |

3 feature mới (`net_risk`, `velocity × merchant_risk`, `age × merchant_risk`) đứng **Top 1–3** về độ quan trọng.

---

## 4. Lưu ý khi làm
- Dùng chung `train_folds.csv`, báo kết quả bằng **CV AUPRC**.
- Mọi scaler / thống kê theo nhóm / target encoding chỉ được `fit` trên fold train.
- Trọng số các cờ trong `weighted_flags` (2.5, 2.4, 2.1, 2.0, 1.4, 1.3 cho `ip_country_mismatch`, `is_ai_generated_scam_attempt`, `billing_shipping_mismatch`, `is_foreign_transaction`, `used_vpn`, `is_new_merchant`) đang lấy từ EDA toàn tập → nếu muốn chặt chẽ tuyệt đối thì tính lại trong từng fold.
- Code mẫu đầy đủ: mục 7 trong `validation_pipeline.ipynb` (hàm `build_error_driven_features`).
