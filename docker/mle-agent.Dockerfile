FROM hub.byted.org/ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    gettext \
    git \
    libgl1 \
    libsm6 \
    libxext6 \
    nano \
    openssh-server \
    p7zip-full \
    python-is-python3 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    sudo \
    tmux \
    unzip \
    vim \
    wget \
    zip \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /opt/aisci/pyproject.toml
RUN python3 - <<'PY' >/tmp/aisci-mle-requirements.txt
import tomllib
from pathlib import Path

payload = tomllib.loads(Path("/opt/aisci/pyproject.toml").read_text(encoding="utf-8"))
for dependency in payload.get("project", {}).get("dependencies", []):
    print(dependency)
PY
RUN python3 -m pip install --break-system-packages --no-cache-dir -r /tmp/aisci-mle-requirements.txt

RUN git config --global user.email "agent@example.com" && \
    git config --global user.name "agent"

WORKDIR /home/code
CMD ["sleep", "infinity"]
