# Untested: this file was written without a Docker build. Volumes: /corpus (read-only), /data, /models.
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY pyproject.toml README.md THIRD_PARTY_NOTICES.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[vectors,mcp]" && cd / && rm -rf /src
RUN useradd --create-home --uid 10001 memidx && mkdir -p /data /models && chown memidx:memidx /data /models
USER memidx
WORKDIR /data
ENV MEMIDX_CORPUS=/corpus \
    MEMIDX_DB=/data/index.db \
    MEMIDX_MODEL_CACHE=/models \
    MEMIDX_LOG=/data/queries.jsonl \
    MEMIDX_MCP_HOST=0.0.0.0 \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1
VOLUME ["/data", "/models"]
EXPOSE 8800 8801
# Exec form: memidx is PID 1 and handles SIGTERM itself (stops the HTTP server and the MCP child process).
STOPSIGNAL SIGTERM
ENTRYPOINT ["memidx"]
CMD ["serve", "--host", "0.0.0.0", "--mcp"]
