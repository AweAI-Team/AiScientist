#!/usr/bin/env bash
set -euo pipefail

PROXY_ARGS=""
if [ -n "$http_proxy" ]; then
  PROXY_ARGS="--build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy --build-arg no_proxy=$no_proxy,.ubuntu.com"
fi

docker build --network host --no-cache --platform=linux/amd64 $PROXY_ARGS -t aisci-paper:latest -f docker/paper-agent.Dockerfile .
