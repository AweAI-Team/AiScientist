FROM mlebench-env

ARG SUBMISSION_DIR
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
ARG LOGS_DIR
ENV LOGS_DIR=${LOGS_DIR}
ARG CODE_DIR
ENV CODE_DIR=${CODE_DIR}
ARG AGENT_DIR
ENV AGENT_DIR=${AGENT_DIR}

RUN mkdir -p ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

ARG CONDA_ENV_NAME=agent

# Build-time proxy: required for pip (PyPI) and tiktoken vocab downloads.
# Pass via: docker build --build-arg HTTP_PROXY=... --build-arg HTTPS_PROXY=...
# ARG-declared values become ENV only within the same RUN layer when explicitly
# exported, so we use a single ENV directive here.
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY}

COPY requirements.txt ${AGENT_DIR}/requirements.txt

RUN conda run -n ${CONDA_ENV_NAME} pip install \
      -r ${AGENT_DIR}/requirements.txt && \
    conda clean -afy

# Pre-download tiktoken encoding files at build time while the build-time proxy
# is available.  This caches the vocab data inside the image so containers can
# use tiktoken offline at runtime — avoids connection errors when
# openaipublic.blob.core.windows.net is unreachable inside the container.
# The step is non-fatal: if the proxy is unavailable at build time we print a
# warning and continue; the code gracefully falls back to bulk prune_messages.
RUN conda run -n ${CONDA_ENV_NAME} python3 -c \
      "import tiktoken; \
       tiktoken.get_encoding('o200k_base'); \
       tiktoken.get_encoding('cl100k_base'); \
       print('tiktoken encodings pre-downloaded OK')" \
    || echo "WARNING: tiktoken pre-download failed — per-message truncation will fall back to bulk prune"

# Clear build-time proxy to avoid leaking it into the final image env.
# Runtime proxy is injected via config.yaml env_vars instead.
ENV HTTP_PROXY="" \
    HTTPS_PROXY="" \
    http_proxy="" \
    https_proxy=""

COPY . ${AGENT_DIR}
# So nonroot can traverse dirs (e.g. subagents/) when exec_run runs start.sh; base image entrypoint only does chmod a+rw.
RUN chmod -R a+rX ${AGENT_DIR}
