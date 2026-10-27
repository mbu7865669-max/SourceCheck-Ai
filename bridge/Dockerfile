# syntax=docker/dockerfile:1
# =============================================================================
# SourceCheck AI - OpenCode bridge image
#
# Runs bridge/opencode_bridge.py (stdlib only) behind an OpenAI-compatible
# endpoint on :18383, backed by the pinned OpenCode CLI. The CLI is what
# actually talks to opencode zen's free models, so this container needs egress
# to the internet but no API keys.
# =============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BRIDGE_HOST=0.0.0.0 \
    BRIDGE_PORT=18383 \
    OPENCODE_BRIDGE_CWD=/work

# curl + ca-certificates fetch the CLI; git lets opencode treat /work as a repo.
# The two hidden dirs are the OpenCode state/config paths that compose mounts
# named volumes onto: pre-creating them here keeps them owned by the service
# user, because Docker creates a missing mount point as root and the bridge
# could then not write its own session database.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash bridge \
    && mkdir -p /work \
        /home/bridge/.local/share/opencode \
        /home/bridge/.config/opencode \
    && chown -R bridge:bridge /work /home/bridge/.local /home/bridge/.config

USER bridge
ENV HOME=/home/bridge

# Pin the CLI to the version the bridge's compatibility patch targets. Installed
# into the service account's home so it stays writable without root.
ARG OPENCODE_VERSION=2.1.4
RUN curl -fsSL https://opencode.ai/v2/install | bash -s -- --version "${OPENCODE_VERSION}" --no-modify-path \
    && /home/bridge/.opencode/bin/opencode --version

ENV PATH="/home/bridge/.opencode/bin:${PATH}" \
    OPENCODE_BIN=/home/bridge/.opencode/bin/opencode

WORKDIR /work

# The bridge is a single stdlib file - no build step, no virtualenv.
COPY --chown=bridge:bridge opencode_bridge.py /app/opencode_bridge.py

EXPOSE 18383

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=6 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:18383/health', timeout=4).status == 200 else 1)"

CMD ["python", "/app/opencode_bridge.py"]
