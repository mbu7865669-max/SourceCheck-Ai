# syntax=docker/dockerfile:1
# =============================================================================
# SourceCheck AI - web gateway image
#
# Serves the FastAPI gateway (`python main.py serve`): the Web UI plus
# POST /answer, which it proxies to the agent service. Model routing lives in
# the agent container - this one only needs AGENT_ENDPOINT to point at it.
# The agent image (Dockerfile.agent) is built from the same source tree, so
# shipping tests here keeps `docker compose run --rm web pytest -q` able to
# verify every layer of the stack in one shot.
# =============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Dependencies first so source-only edits reuse the layer.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Application code. `tests/` ships in the image on purpose so
# `docker compose run --rm web pytest -q` can verify the exact bytes that run.
COPY main.py agent_service.py ./
COPY app ./app
COPY bridge ./bridge
COPY tests ./tests

# Unprivileged service account: the code never writes to disk, so root buys nothing.
RUN useradd --system --no-create-home --uid 10001 app
USER app

EXPOSE 8000

# Liveness probe - /livez only, so Docker doesn't fan out to agent + bridge.
HEALTHCHECK --interval=15s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/livez', timeout=4).status == 200 else 1)"

CMD ["python", "main.py", "serve"]
