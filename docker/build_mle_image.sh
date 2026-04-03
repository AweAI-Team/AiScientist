#!/usr/bin/env bash
set -euo pipefail

HTTP_PROXY_VALUE="${http_proxy:-${HTTP_PROXY:-}}"
HTTPS_PROXY_VALUE="${https_proxy:-${HTTPS_PROXY:-}}"
NO_PROXY_VALUE="${no_proxy:-${NO_PROXY:-}}"
BASE_IMAGE="${AISCI_DOCKER_BASE_IMAGE:-ubuntu:24.04}"

BUILD_ARGS=(
  --network host
  --no-cache
  --platform=linux/amd64
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"
)

if [ -n "${HTTP_PROXY_VALUE}" ]; then
  BUILD_ARGS+=(
    --build-arg "http_proxy=${HTTP_PROXY_VALUE}"
    --build-arg "https_proxy=${HTTPS_PROXY_VALUE}"
    --build-arg "no_proxy=${NO_PROXY_VALUE},.ubuntu.com"
    --build-arg "HTTP_PROXY=${HTTP_PROXY_VALUE}"
    --build-arg "HTTPS_PROXY=${HTTPS_PROXY_VALUE}"
    --build-arg "NO_PROXY=${NO_PROXY_VALUE},.ubuntu.com"
  )
fi

docker build "${BUILD_ARGS[@]}" -t aisci-mle:test -f docker/mle-agent.Dockerfile .
