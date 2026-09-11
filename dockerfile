# syntax=docker/dockerfile:1

FROM python:3.11-slim

# Configure non-interactive installation and unbuffered logging
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/workspace"

# Install system dependencies (OpenMP for LightGBM, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Create directory tree for runtime mounts and persistent outputs
RUN mkdir -p /workspace/data/raw \
             /workspace/data/processed \
             /workspace/model/models/lightgbm \
             /workspace/reports

# Copy dependency definition first for layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source tree and configuration files
COPY configs/ ./configs/
COPY model/ ./model/
COPY train.py test.py evaluate_only.py export_predictions.py ./

# Default container command
CMD ["bash"]