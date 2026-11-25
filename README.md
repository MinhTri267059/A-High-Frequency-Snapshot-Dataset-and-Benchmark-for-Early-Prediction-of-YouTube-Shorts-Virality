# 🚀 YouTube Shorts Virality Prediction & Analysis

## 📖 Tổng quan (Overview)
Dự án này sử dụng Machine Learning để **dự đoán sớm khả năng viral của một video YouTube Shorts** tại thời điểm 24h và 48h, chỉ dựa trên **dữ liệu tương tác ở 6 giờ đầu tiên** sau khi đăng tải. Bằng việc kết hợp các dữ liệu dạng bảng (tabular engagement data), lịch sử kênh (channel history), và trích xuất đặc trưng văn bản từ tiêu đề (NLP embeddings), hệ thống cung cấp dự đoán có độ chính xác cao giúp các nhà sáng tạo nội dung tối ưu hoá chiến lược phân phối.

## 🔬 Tính học thuật & Phương pháp luận (Academic Methodology)

### 1. Vấn đề rò rỉ dữ liệu (Data Leakage Audit)
Để đảm bảo tính khoa học và minh bạch, một quá trình kiểm tra **Data Leakage** nghiêm ngặt được áp dụng. Tất cả các nhãn mục tiêu trong tương lai (Target Variables như `views_24h`, `views_48h`, `views_7d`) đều bị loại bỏ hoàn toàn (dropped/masked) khỏi ma trận đặc trưng (Feature Matrix $X$) trước khi đưa vào huấn luyện mô hình.
Mô hình hoàn toàn **chỉ sử dụng thông tin khả dụng tại thời điểm 6 giờ**:
- **Early engagement (0-6h):** Lượt xem, lượt thích, bình luận tại mốc 2h, 4h, 6h.
- **Channel history:** Thống kê lịch sử hiệu suất của kênh (trung vị lượt xem, tỷ lệ viral trong quá khứ).
- **Video metadata:** Thời gian đăng (giờ, ngày trong tuần - được mã hoá lượng giác sin/cos), độ dài video, số lượng tag.
- **Title embeddings:** 384 chiều vector trích xuất từ tiêu đề sử dụng mô hình ngôn ngữ (DistilBERT).

### 2. Định nghĩa Bài toán Phân loại (Framing the Problem)
Thay vì chỉ giải quyết bài toán Hồi quy (Regression) để đoán số view, dự án định nghĩa một bài toán Phân loại nhị phân (Binary Classification) khó hơn:
- **"Viral"** được định nghĩa bằng một ngưỡng (Threshold) phân vị cao. Ví dụ, video được dán nhãn là 1 (Viral) nếu số view ở giờ 24 vượt qua mốc `56,785 views` (tương ứng với top 10% các video thành công nhất).
- Ngưỡng này được tính toán **hoàn toàn độc lập trên tập Train** (Train-set strict isolation) để đảm bảo không bị rò rỉ phân phối phân vị từ tập Test vào quá trình tạo nhãn.
- Dataset mang tính mất cân bằng dữ liệu cực cao (Highly Imbalanced Data), với lớp Tích cực (Viral) thường chỉ chiếm khoảng `10%`.

### 3. Mô hình So sánh (Model Benchmarking)
Dự án thực hiện đánh giá chéo (Walk-Forward Time-Series CV) và so sánh nhiều phương pháp luận khác nhau:
1. **Heuristic Rule (Baseline 1):** Dựa trên luật tĩnh đơn giản (views_at_6h >= ngưỡng tĩnh).
2. **Logistic Regression (Baseline 2):** Mô hình tuyến tính tiêu chuẩn.
3. **Random Forest (Baseline 3):** Mô hình Ensemble dựa trên Bagging.
4. **Hybrid LSTM (Baseline 4):** Mô hình Deep Learning kết hợp Recurrent Neural Network để xử lý đặc trưng chuỗi thời gian (views lúc 2h, 4h, 6h) và Multi-Layer Perceptron (MLP) cho dữ liệu tĩnh.
5. **XGBoost (Ours):** Mô hình Gradient Boosting tối ưu hoá được chọn làm mô hình chính nhờ khả năng xử lý mạnh mẽ trên dữ liệu dạng bảng và khả năng chống chịu dữ liệu mất cân bằng.

### 4. Đánh giá Kết quả (Evaluation Results)
Do dữ liệu mất cân bằng, các chỉ số truyền thống như Accuracy không phản ánh đúng thực tế. Chúng tôi sử dụng các metric khắt khe hơn:
- **PR-AUC (Precision-Recall Area Under Curve)**
- **ROC-AUC**
- **Brier Score (độ chính xác của calibration dự đoán)**
- **F1-Macro / F1-Viral**

**Kết quả thực nghiệm:**
- Trải qua bài kiểm tra chéo theo thời gian (3-fold Walk-Forward CV) chặt chẽ, **XGBoost** cho thấy khả năng vượt trội ở **PR-AUC trung bình (0.7769)**, **ROC-AUC (0.9634)** và **Brier Score (0.0564)**, xử lý rất tốt lớp thiểu số.
- Khi đánh giá Hold-out tĩnh, XGBoost giữ PR-AUC đạt `0.8198`.
- Mô hình **LSTM** đạt kết quả khá (PR-AUC 0.7827, ROC-AUC 0.9612), tuy nhiên do độ dài chuỗi thời gian khả dụng quá ngắn (chỉ 3 mốc 2h, 4h, 6h), kiến trúc Deep Learning chưa tận dụng được tối đa ưu thế. Giải pháp dạng Tree-based (XGBoost) vẫn duy trì sự ổn định, tốc độ học nhanh, và khả năng giải thích rõ ràng hơn.
- Giá trị **SHAP (SHapley Additive exPlanations)** được phân tích để minh bạch hoá mô hình, chứng minh rằng số view tại 6h và lịch sử kênh đóng vai trò chủ đạo định hướng dự đoán của XGBoost.

---

## 🛠️ Cấu trúc dự án (Project Structure)

Mã nguồn được chuẩn hoá để dễ dàng tái lập các thực nghiệm khoa học:
- `01_clean_data.py`: Tiền xử lý, làm sạch và nối dữ liệu thô.
- `02_preprocess_features.py`: Trích xuất đặc trưng (Feature Engineering & DistilBERT Embeddings).
- `03_data_split.py`: Phân chia tập dữ liệu (Train/Test) theo thời gian.
- `04_train_xgb_classifier.py`: Huấn luyện mô hình XGBoost phân loại.
- `05_evaluate_classifiers.py`: So sánh đa mô hình (XGBoost vs RF vs LR vs LSTM).
- `06_optuna_classifier.py`: Tối ưu hoá siêu tham số với Optuna.
- `07_train_xgb_regression.py` & `08_evaluate_regression.py`: Huấn luyện & đánh giá cho bài toán Hồi quy liên tục.
- `09_train_lstm_regression.py` & `10_compare_lstm_vs_xgboost.py`: Các thực nghiệm mở rộng so sánh Deep Learning vs Tree-based trong Hồi quy.

---

## 🚀 Ứng dụng thực tiễn (Deployment)

**Lưu ý quan trọng:** 
*   **KHÔNG CẦN HUẤN LUYỆN LẠI (TRAIN):** Các phiên bản model tốt nhất đã được đóng gói sẵn trong thư mục `web_deployment_package/models/` dưới dạng các file `.pkl`. Bạn chỉ cần cài đặt môi trường và chạy Inference.
*   Dự án bao gồm một **Backend API** và một **Dashboard** (Giao diện React) để hiển thị dự đoán theo thời gian thực.

### Hướng dẫn khởi chạy nhanh
**1. Backend (Python API)**
```bash
cd web_deployment_package
python -m venv venv
source venv/bin/activate  # macOS/Linux (Hoặc venv\Scripts\activate trên Windows)
pip install -r requirements.txt
python main.py  # Chạy API ở cổng 8000
```

**2. Frontend (React Dashboard)**
```bash
cd yt-dashboard
npm install
npm run dev
```

**3. Docker (Tuỳ chọn)**
Nếu bạn không muốn cài đặt cục bộ, hãy sử dụng Docker để khởi chạy API Backend nhanh chóng:
```bash
cd web_deployment_package
docker build -t youtube-predictor .
docker run -p 8000:8000 youtube-predictor
```

---
## How to cite

If you found our work, dataset, or benchmark useful in your research, please cite our paper:

M.-T. Truong and H.-N. Tran. [A High-Frequency Snapshot Dataset and Benchmark for Early Prediction of YouTube Shorts Virality](https://link.springer.com/chapter/10.1007/978-981-92-8104-6_43). *Future Data and Security Engineering (FDSE 2026)*, Communications in Computer and Information Science, vol. 3238, pp. 580–594, Springer, 2027. doi: [10.1007/978-981-92-8104-6_43](https://doi.org/10.1007/978-981-92-8104-6_43).

```bibtex
@inproceedings{truong2027high,
  author    = {Truong, Minh-Tri and Tran, Hung-Nghiep},
  title     = {A High-Frequency Snapshot Dataset and Benchmark for Early Prediction of YouTube Shorts Virality},
  booktitle = {Future Data and Security Engineering. FDSE 2026},
  editor    = {Dang, Tran Khanh and K{\"u}ng, Josef and Chung, Tai M. and Tran, Minh-Triet and Pastor, Oscar},
  series    = {Communications in Computer and Information Science},
  volume    = {3238},
  pages     = {580--594},
  publisher = {Springer},
  address   = {Singapore},
  year      = {2027},
  doi       = {10.1007/978-981-92-8104-6_43}
}
