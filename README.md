# 🚀 YouTube Shorts View Predictor

Dự án này sử dụng Machine Learning (XGBoost) kết hợp với Xử lý ngôn ngữ tự nhiên (NLP) để dự đoán số lượt xem của YouTube Shorts sau 24 giờ và 48 giờ. 

**Lưu ý quan trọng cho máy mới:** 
*   **KHÔNG CẦN HUẤN LUYỆN LẠI (TRAIN):** Các phiên bản model tốt nhất đã được đóng gói sẵn trong thư mục `web_deployment_package/` dưới dạng các file `.pkl`. Bạn chỉ cần cài đặt môi trường và chạy.
*   Dự án bao gồm một **Backend API** (Xử lý model) và một **Dashboard** (Giao diện người dùng).

---

## 📂 Cấu trúc dự án để di chuyển
Khi gửi sang máy khác, hãy đảm bảo bạn gửi toàn bộ thư mục gốc, đặc biệt là:
*   `web_deployment_package/`: Chứa mã nguồn API và các file Model (`.pkl`).
*   `yt-dashboard/`: Chứa mã nguồn Giao diện (React).

---

## 🛠️ Hướng dẫn cài đặt môi trường (Dành cho máy mới)

### 1. Cài đặt cho Backend (Python API)
Máy mới cần cài đặt Python 3.9 trở lên.

*   **Bước 1: Tạo môi trường ảo (Khuyên dùng)**
    ```bash
    cd web_deployment_package
    python -m venv venv
    source venv/bin/activate  # Trên macOS/Linux
    # Hoặc: venv\Scripts\activate  # Trên Windows
    ```
*   **Bước 2: Cài đặt thư viện**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
*   **Bước 3: Cấu hình API Key**
    Tạo file `.env` trong thư mục `web_deployment_package/` (nếu chưa có):
    ```env
    YOUTUBE_API_KEY=YOUR_API_KEY_HERE
    ```

### 2. Cài đặt cho Frontend (React Dashboard)
Máy mới cần cài đặt Node.js (phiên bản 18+).

```bash
cd yt-dashboard
npm install
```

---

## 🚀 Cách khởi chạy hệ thống

### Bước 1: Khởi động Backend
Mở Terminal 1:
```bash
cd web_deployment_package
# Đảm bảo đã activate môi trường ảo
python main.py
```
*API sẽ chạy tại cổng 8000.*

### Bước 2: Khởi động Frontend
Mở Terminal 2:
```bash
cd yt-dashboard
npm run dev
```
*Truy cập link hiển thị ở terminal (thường là http://localhost:5173).*

---

## 🐳 Sử dụng Docker (Nếu không muốn cài môi trường thủ công)
Nếu máy mới có cài [Docker](https://www.docker.com/), bạn có thể chạy nhanh mà không cần cài Python/Node:

1.  Mở terminal tại `web_deployment_package`.
2.  Chạy: `docker build -t youtube-predictor .`
3.  Chạy: `docker run -p 8000:8000 youtube-predictor`
*(Yêu cầu file .env có chứa API Key vẫn phải tồn tại trước khi build).*

---

## 🧠 Giải đáp thắc mắc
*   **Tại sao không cần train?** Vì model đã được học từ dữ liệu lịch sử và lưu lại thành file vật lý. Máy mới chỉ việc "nạp" file này lên để tính toán cho dữ liệu thực tế (Inference).
*   **Lượt xem lấy từ đâu?** API sẽ dùng YouTube API Key của bạn để lấy thông tin lượt xem thực tế hiện tại của video, sau đó mới đưa vào model để dự đoán tương lai.

---
*Phát triển bởi Đội ngũ Predict Views YouTube Short*
