# Use an official Ubuntu base image from Docker Hub
FROM ubuntu:24.04

# Optional: uncomment this block to use the Alibaba PyPI mirror when building in mainland China.
# RUN mkdir -p /etc/pip && \
#     printf '[global]\nindex-url = https://mirrors.aliyun.com/pypi/simple/\ntrusted-host = mirrors.aliyun.com\n' > /etc/pip.conf
# HuggingFace: use default huggingface.co via proxy (7 MB/s) instead of hf-mirror.com (2 MB/s, SSL errors)
ENV HF_HUB_ETAG_TIMEOUT=120
ENV HF_HUB_DOWNLOAD_TIMEOUT=600

# Set non-interactive mode for apt-get to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install common build and ML dependency packages
RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        wget curl unzip sudo \
        build-essential git cmake \
        libatlas-base-dev libblas-dev liblapack-dev libopenblas-dev \
        gfortran libsm6 libxext6 libxrender-dev libgl1 \
        python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

# Install Julia 1.10 LTS (required by sbibm for ODE simulators like Lotka-Volterra, SIR)
ENV JULIA_VERSION=1.10.7
RUN curl -fsSL "https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" | \
    tar -xz -C /usr/local && \
    ln -s /usr/local/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia

# Deterministically install both 3.11 and 3.12, users can choose between them
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.11 python3.11-venv python3.11-dev \
        python3.12 python3.12-venv python3.12-dev \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set default python version to 3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2
# users can switch to 3.11 by running `update-alternatives --set python3 /usr/bin/python3.11`

# Install jupyter for Alcatraz kernel support
# Use --break-system-packages since we're in a container and PEP 668 restrictions don't apply
RUN http_proxy= https_proxy= pip install --break-system-packages jupyter ipykernel

# you would then
# 1. make a /submission dir 
# 2. copy the submission there
# 3. run bash /submission/reproduce.sh
