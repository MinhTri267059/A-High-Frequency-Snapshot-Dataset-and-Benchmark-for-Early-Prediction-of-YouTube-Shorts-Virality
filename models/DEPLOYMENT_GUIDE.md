# 🚀 Hướng dẫn Triển khai Model XGBoost (YouTube Views Prediction)

Tài liệu này dành cho lập trình viên (Dev) ở máy khác muốn tích hợp hệ thống dự đoán lượt xem vào ứng dụng Web hoặc API.

---

## 📁 1. Các tệp tin cần thiết
Để hệ thống hoạt động, bạn cần 3 tệp tin chính trong thư mục `models/`:
1.  **`best_xgboost_views_24h.pkl`**: Model dự đoán lượt xem tại mốc 24h.
2.  **`best_xgboost_views_48h.pkl`**: Model dự đoán lượt xem tại mốc 48h.
3.  **`features_list.pkl`**: Danh sách tên các đặc trưng (features) theo đúng thứ tự mà model yêu cầu.

---

## ⚙️ 2. Quy trình Xử lý Dữ liệu

Dữ liệu đầu vào **BẮT BUỘC** phải trải qua 2 bước biến đổi quan trọng để đảm bảo độ chính xác:

### A. Tiền xử lý (Preprocessing) - Chuẩn hóa Log1p
Model được huấn luyện trên thang đo Logarit để giảm độ nhiễu của các video viral.
-   **Quy tắc**: Với tất cả các giá trị lượt xem (views) hoặc tốc độ (velocity) ở đầu vào, hãy áp dụng công thức: `input = log(1 + x)`.
-   **Các cột cần áp dụng**: `views_at_0h`, `views_at_1h`, `views_at_2h`, `views_at_6h`, `channel_avg_views`, và các cột `vel_...`.

### B. Căn chỉnh Đặc trưng (Feature Alignment)
Hệ thống sử dụng hơn 400 đặc trưng. Code của bạn phải đảm bảo thứ tự các cột truyền vào model khớp hoàn toàn với `features_list.pkl`.

### C. Hậu xử lý (Post-processing) - Nghịch đảo Log
Kết quả model trả về (`y_pred`) là giá trị Logarit. Bạn cần chuyển nó về số view thực tế:
-   **Công thức**: `thực_tế = exp(y_pred) - 1`.

---

## 💻 3. Code mẫu tích hợp (Python)

Dưới đây là đoạn mã gợi ý để triển khai trên môi trường Production:

```python
import pickle
import numpy as np
import pandas as pd

# 1. Load Model và Feature List
with open('models/best_xgboost_views_24h.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/features_list.pkl', 'rb') as f:
    features_list = pickle.load(f)

def predict_views(raw_input_data):
    """
    raw_input_data: dict chứa thông tin video (views_at_2h, duration,...)
    """
    # Tạo DataFrame từ input
    df = pd.DataFrame([raw_input_data])
    
    # Bước 1: Tiền xử lý Log1p cho các cột views/velocity
    cols_to_log = [c for c in df.columns if 'views' in c or 'vel' in c]
    for col in cols_to_log:
        df[col] = np.log1p(df[col].clip(0))
        
    # Bước 2: Căn chỉnh thứ tự đặc trưng (Mapping)
    # Những cột thiếu sẽ được gán bằng 0
    df_final = df.reindex(columns=features_list, fill_value=0)
    
    # Bước 3: Dự đoán
    log_pred = model.predict(df_final)
    
    # Bước 4: Chuyển đổi về View thực tế (Expm1)
    final_pred = np.expm1(log_pred)
    
    return int(max(0, final_pred[0]))

# --- Ví dụ sử dụng ---
data = {
    'views_at_6h': 1500,
    'channel_avg_views': 5000,
    'duration_sec': 600,
    # ... thêm các features khác từ hệ thống crawl
}
# print(f"Dự đoán lượt xem sau 24h: {predict_views(data)}")
```

---

## ⚠️ Lưu ý quan trọng
-   **Thư viện**: Đảm bảo cài đặt đúng phiên bản `xgboost`, `pandas`, `numpy` và `scikit-learn` tương đồng với môi trường huấn luyện.
-   **Xử lý lỗi**: Các giá trị âm (nếu có) nên được `clip(0)` trước khi qua log1p để tránh lỗi `NaN`.
