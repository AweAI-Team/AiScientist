#!/bin/bash

set -e

PROXY_ARGS=""
if [ -n "$http_proxy" ]; then
  PROXY_ARGS="--build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy --build-arg no_proxy=$no_proxy,.ubuntu.com"
fi

docker build --network host --no-cache --platform=linux/amd64 $PROXY_ARGS -t pb-env -f paperbench/Dockerfile.base . &
docker build --network host --no-cache --platform=linux/amd64 $PROXY_ARGS -f paperbench/reproducer.Dockerfile -t pb-reproducer . &
wait

# 用于jupyter 端口转发
docker pull docker.1ms.run/alpine/socat
docker tag docker.1ms.run/alpine/socat alpine/socat