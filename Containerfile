FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ /app
# COPY scripts/ /app/scripts/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional required packages from train.py
RUN pip install --no-cache-dir trl

# Set the entry point
ENTRYPOINT ["python3", "train.py"]
