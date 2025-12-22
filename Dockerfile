# Sử dụng base image Python chính thức
FROM python:3.10-slim

# Thiết lập môi trường làm việc bên trong container
WORKDIR /app

# Copy file yêu cầu và cài đặt dependencies
COPY requirements.txt .

# Cài đặt các dependencies cần thiết, bao gồm PyTorch và Ultralytics
# Sử dụng --no-cache-dir để giảm kích thước image
RUN pip install --no-cache-dir -r requirements.txt

# Copy ứng dụng và các mô hình cần thiết vào container
# GIẢ ĐỊNH: Các file model (best.pt, sam2_b.pt) nằm cùng cấp với Dockerfile
COPY app_gui.py .
COPY best.pt .
COPY sam2_b.pt .

# Lệnh khởi động (ENTRYPOINT)
# LƯU Ý QUAN TRỌNG: Lệnh này chỉ giữ container chạy, vì ứng dụng GUI (Tkinter) 
# không thể chạy trực tiếp. Bạn phải chạy thủ công bên trong container.
CMD ["tail", "-f", "/dev/null"]