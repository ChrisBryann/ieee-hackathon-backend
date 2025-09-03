# Use NVIDIA CUDA 12.6 base image with Ubuntu 24.04 (gets python Python 3.12 by default)
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS development

WORKDIR /usr/ocr/src

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-distutils \
    python3-venv \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
# RUN python -m pip install --upgrade pip

# Create a virtual environment
RUN python -m venv env
ENV PATH="/env/bin:$PATH"

# Copy requirements to install dependencies
COPY ./ocr-server/requirements.txt .

RUN pip install -r requirements.txt

# Copy code
COPY . .

# Run application
CMD ["python3", "api.py"]