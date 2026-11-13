# Volumes: /corpus (read-only), /data, /models. Built and smoke-tested with Docker 29 on Linux (amd64).
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[vectors,mcp]" && cd / && rm -rf /src
RUN useradd --create-home --uid 10001 memidx && mkdir -p /data /models && chown memidx:memidx /data /models && chmod 700 /data /models
USER memidx
WORKDIR /data
# MEMIDX_GIT_SAFE_DIRECTORY=1: the repository mounted at /corpus belongs to a user of the host, not to the
# memidx user inside the container, so git would refuse it ("dubious ownership"). Mounting a repository is
# the operator's explicit choice, so this image trusts its owner. memidx still disables fsmonitor and the
# untracked cache for its git calls, so the repository's own config cannot start programs through them.
# Vectors stay off unless you add MEMIDX_VECTORS=1 and MEMIDX_MODEL=<preset> (see `memidx model list`).
ENV MEMIDX_CORPUS=/corpus \
    MEMIDX_DB=/data/index.db \
    MEMIDX_MODEL_CACHE=/models \
    MEMIDX_LOG=/data/queries.jsonl \
    MEMIDX_MCP_HOST=0.0.0.0 \
    MEMIDX_GIT_SAFE_DIRECTORY=1 \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1
VOLUME ["/data", "/models"]
EXPOSE 8800 8801
# Exec form: memidx is PID 1 and handles SIGTERM itself (stops the HTTP server and the MCP child process).
STOPSIGNAL SIGTERM
ENTRYPOINT ["memidx"]
CMD ["serve", "--host", "0.0.0.0", "--mcp"]
