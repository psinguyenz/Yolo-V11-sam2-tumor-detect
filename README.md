# 🧠 Brain Tumor Detection: YOLOv11 & SAM2 Pipeline

[![Docker Hub](https://img.shields.io/badge/Docker_Hub-Pulls_100+-blue.svg)](https://hub.docker.com/r/psinguyenz/tumor-detector)
[![Model: YOLOv11](https://img.shields.io/badge/Model-YOLOv11-green)](https://github.com/ultralytics/ultralytics)
[![Framework: SAM2](https://img.shields.io/badge/Segmentation-SAM2-orange)](https://github.com/facebookresearch/segment-anything-2)

Dự án xây dựng quy trình phát hiện và phân đoạn khối u não từ ảnh y tế. Hệ thống kết hợp sức mạnh phát hiện vật thể nhanh chóng của **YOLOv11** và khả năng phân đoạn chính xác của **SAM2 (Segment Anything Model 2)**.


## 🌟 Key Features
- **Hybrid Workflow**: Sử dụng YOLOv11 để tạo Bounding Box (Detection) và dùng đó làm Prompt cho SAM2 để trích xuất Mask (Segmentation).
- **Medical Precision**: Đạt chỉ số **Recall: 0.645**, tối ưu cho việc tránh bỏ lỡ các dấu hiệu khối u trong chẩn đoán hình ảnh.
- **Easy Deployment**: Toàn bộ môi trường phức tạp của SAM2 đã được đóng gói vào **Docker Image**, giúp triển khai ngay lập tức mà không lo lỗi xung đột thư viện.
- **Desktop Application**: Tích hợp giao diện đồ họa (GUI) đơn giản để người dùng upload ảnh và nhận kết quả trực quan.

## 📊 Performance
- **YOLOv11 Recall**: 0.645
- **Segmentation**: SAM2 cung cấp độ chi tiết cao cho các khối u có hình dạng phức tạp, hỗ trợ bác sĩ đo lường kích thước khối u chính xác hơn.

## 🚀 Quick Start with Docker
Do kích thước mô hình SAM2 và các phụ thuộc rất lớn, việc cài đặt thủ công có thể gặp nhiều khó khăn. Khuyến khích sử dụng Docker image đã được tối ưu:

```bash
# Pull image từ Docker Hub
docker pull psinguyenz/tumor-detector:latest

# Chạy container (Yêu cầu cấu hình display nếu muốn dùng GUI)
docker run -it psinguyenz/tumor-detector:latest
```

🛠️ Tech Stack
- Detection: YOLOv11 (Ultralytics)
- Segmentation: SAM2 (Meta AI)
- Deployment: Docker, Docker Hub

📂 Project Structure
```bash
├── src/                 # Mã nguồn xử lý chính
├── app_gui.py           # Giao diện người dùng 
├── best.pt              # Trọng số mô hình YOLOv11 đã được train
├── Dockerfile           # Cấu hình đóng gói hệ thống
├── requirements.txt     # Các thư viện cần thiết
└── README.md            # Tài liệu hướng dẫn dự án
```
