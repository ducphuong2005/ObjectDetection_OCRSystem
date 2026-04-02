# Sử dụng base image Python 3.10
FROM python:3.10-slim

# Cài đặt biến môi trường cho HF Spaces
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Ho_Chi_Minh \
    MAX_JOBS=2

# Cài đặt các gói hệ thống cần thiết (bao gồm Git để clone detectron2 và g++ để build C++)
COPY packages.txt .
RUN apt-get update && \
    apt-get install -y git build-essential ninja-build libgl1 libglib2.0-0 && \
    xargs -a packages.txt apt-get install -y && \
    rm -rf /var/lib/apt/lists/*

# Cài đặt PyTorch Core trước (ép dùng bản CPU để nhẹ và phù hợp môi trường HF Spaces)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Cài đặt Detectron2 (vì đã có torch phía trên, ta tắt build-isolation để detectron2 nhận được torch)
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation

# Cài đặt các thư viện còn lại
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Thiết lập User (Hugging Face Spaces yêu cầu không chạy dưới quyền root)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Copy toàn bộ mã nguồn vào app
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Mở cổng 7860
EXPOSE 7860

# Khởi chạy ứng dụng (PORT có thể bị ghi đè bởi Render)
CMD ["python", "run_space.py"]
