FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy tệp yêu cầu và cài đặt dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- QUAN TRỌNG: Pre-cache model NLP ---
# Bước này tải model all-MiniLM-L6-v2 về Image trong lúc build
# Để khi khởi chạy container không cần tải lại từ internet
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy toàn bộ code và model vào container
COPY . .

# Mở port cho FastAPI
EXPOSE 8000

# Lệnh khởi chạy API
CMD ["uvicorn", "main.py:app", "--host", "0.0.0.0", "--port", "8000"]
