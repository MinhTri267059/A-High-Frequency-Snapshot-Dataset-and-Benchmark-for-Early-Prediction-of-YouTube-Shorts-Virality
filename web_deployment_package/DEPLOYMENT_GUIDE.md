# 🚀 Hướng dẫn Triển khai: Docker + NLP + React

Tài liệu này hướng dẫn cách build và triển khai hệ thống dự đoán YouTube Views sử dụng **XGBoost** kết hợp với **Title Embedding (NLP)**, sẵn sàng kết nối với giao diện **React**.

---

## 🌍 1. Cấu trúc Gói triển khai
Thư mục `web_deployment_package/` bao gồm:
-   `Dockerfile`: Tệp cấu hình build Image Docker (đã tối ưu pre-cache model NLP).
-   `main.py`: Mã nguồn API (FastAPI) xử lý dự đoán.
-   `requirements.txt`: Danh sách thư viện (NLP, API, ML).
-   `best_xgboost_views_24h.pkl` & `48h.pkl`: Các file model đã huấn luyện.
-   `features_list.pkl`: Danh sách 409 đặc trưng bắt buộc.

---

## 🏗️ 2. Triển khai với Docker

### Bước 1: Build Docker Image
Chạy lệnh sau tại thư mục có `Dockerfile`:
```bash
docker build -t youtube-predictor .
```
*(Quá trình này sẽ mất vài phút vì Docker sẽ tự động tải model NLP `all-MiniLM-L6-v2` và nén vào Image).*

### Bước 2: Chạy Container
```bash
docker run -p 8000:8000 youtube-predictor
```
API sẽ lắng nghe tại cổng `8000`.

---

## ⚛️ 3. Kết nối với React

Giao diện React của bạn có thể gửi yêu cầu dự đoán bằng lệnh `fetch` hoặc `axios`.

### Dữ liệu gửi lên (Payload JSON):
```json
{
  "title": "Cách làm video viral 2026",
  "views_at_6h": 5000,
  "channel_avg_views": 10000,
  "duration_sec": 600
}
```

### Code mẫu trong React:
```javascript
const handlePredict = async () => {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      title: "Tiêu đề video của bạn",
      views_at_6h: 1200,
      channel_avg_views: 4500,
      duration_sec: 240
    })
  });
  const data = await response.json();
  console.log("Dự đoán 24h:", data.prediction_24h);
};
```

---

## ⚙️ 4. Cơ chế hoạt động (Dành cho Dev)
1.  **NLP Encoding**: API nhận `title`, dùng `SentenceTransformer` chuyển thành vecto 384 chiều.
2.  **Preprocessing**: Áp dụng `log1p` cho các trường views giống như lúc huấn luyện.
3.  **Alignment**: Tự động điền 0 cho các đặc trưng thiếu và sắp xếp theo đúng thứ tự của `features_list.pkl`.
4.  **Inference**: Chạy qua model XGBoost và trả về kết quả đã được nghịch đảo log (`expm1`).

---

## ⚠️ Lưu ý kỹ thuật
- **CORS**: File `main.py` hiện đang để `allow_origins=["*"]`. Hãy đổi thành domain React của bạn khi đưa lên Production.
- **Dung lượng**: Image Docker sẽ nặng khoảng 1GB do chứa thư viện PyTorch (CPU) và Model NLP.
