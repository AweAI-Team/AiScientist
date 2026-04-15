FROM mlebench-env

# Paths inside the container (must match agents/README.md / run_agent mounts).
# Defaults apply when --build-arg is omitted; note: `docker build --build-arg FOO=`
# passes an empty value and would still clear FOO — use shell `${FOO:-...}` when invoking build.
ARG SUBMISSION_DIR=/home/submission
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
# where to put any logs, will be extracted
ARG LOGS_DIR=/home/logs
ENV LOGS_DIR=${LOGS_DIR}
# where to put any code, will be extracted
ARG CODE_DIR=/home/code
ENV CODE_DIR=${CODE_DIR}
# where to put any other agent-specific files, will not be necessarily extracted
ARG AGENT_DIR=/home/agent
ENV AGENT_DIR=${AGENT_DIR}

RUN mkdir -p ${SUBMISSION_DIR} ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

ARG CONDA_ENV_NAME=agent
ARG REQUIREMENTS=${AGENT_DIR}/requirements.txt

# copy just the requirements file, so that we can cache conda separately from the agent files
COPY requirements.txt ${AGENT_DIR}/requirements.txt

# Requirements for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# create conda environment and install the requirements to it
# Pin httpx<0.28 to fix compatibility with openai library ('proxies' kwarg removed in httpx 0.28)
RUN conda run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt && \
    conda run -n ${CONDA_ENV_NAME} pip install 'httpx<0.28' && \
    conda clean -afy

# put all the agent files in the expected location
COPY . ${AGENT_DIR}

# Patch vendored aideml: route glm-* and gemini-3-* through OpenAI/Azure clients.
RUN AIDE_BACKEND="$(conda run -n ${CONDA_ENV_NAME} python -c "import aide.backend, os; print(os.path.dirname(aide.backend.__file__))")" && \
    cp "${AGENT_DIR}/aideml_patches/backend/__init__.py" "${AIDE_BACKEND}/__init__.py" && \
    cp "${AGENT_DIR}/aideml_patches/backend/backend_openai.py" "${AIDE_BACKEND}/backend_openai.py"
