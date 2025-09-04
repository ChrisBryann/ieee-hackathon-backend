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
    # python3-distutils \ 
    # deprecated in python 3.12+
    python3-venv \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python and pip
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
# RUN python -m pip install --upgrade pip

# Copy requirements to install dependencies
COPY ./ocr-server/requirements.txt .

# Create a virtual environment and install everything in one RUN to maintain activation
# make sure to install paddlepaddle-gpu here because it gets installation from an index-url (-i)
# if we let it install from requirements.txt, it will look for the version in offical site (which does not exist)
RUN python -m venv env && \
    . ./env/bin/activate && \
    pip install --upgrade pip && \
    pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ && \ 
    pip install -r requirements.txt

ENV PATH="/env/bin:$PATH"

# Copy code
COPY . .

# Run application
CMD ["python3", "api.py"]