"""
Main application entry point - the web gateway.

Exposes two ways to drive the agent:

* **CLI**  - `python main.py "your question"` for a single ad-hoc query (in-process).
* **HTTP** - a FastAPI gateway for automated graders and programmatic clients.
  `POST /answer` is proxied to the agent service (see `agent_service.py`), which
  runs the retrieval -> verification -> reasoning pipeline in its own container.

Run the service:

    python main.py serve
    # equivalent to: uvicorn main:app --host 0.0.0.0 --port 8000

Endpoints:

    GET  /health      liveness + agent reachability (+ bridge details via the agent)
    GET  /            Web UI in a browser, service metadata for API clients
    GET  /ui          the Web UI, unconditionally
    POST /answer      proxy to the agent service (AGENT_ENDPOINT, default :8001)

Model selection is driven by `MODEL_PROVIDER` in the environment / `.env` of the
**agent** service: `mock` (offline default), `opencode` (free models via the
bridge), `local_rocm`, or `local_server`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from app.agent import SourceCheckAgent
from app.config import settings
from app.model import get_model
from app.schemas import AgentResponse, AnswerRequest

logger = logging.getLogger("sourcecheck")

# Reserved CLI words that mean "start the HTTP service" rather than "answer this".
# Keeping them in one place documents the ambiguity instead of hiding it.
_SERVE_WORDS = frozenset({"serve", "server", "api"})

# Browser UI: one self-contained page served as-is - no template engine, no CDN
# dependencies, so it works the same in a container with no outbound access to
# asset hosts.
_UI_FILE = Path(__file__).resolve().parent / "app" / "static" / "index.html"

# `no-cache` means "revalidate", not "don't store": browsers keep the page and get
# a cheap 304 on reload, yet never run a stale UI after a redeploy.
_UI_HEADERS = {"Cache-Control": "no-cache"}


def _ui_response(request: Request) -> Response:
    """Serve the UI, answering `If-None-Match` with a 304.

    FileResponse emits an ETag but does not honour conditional requests itself,
    so passing `stat_result` makes it compute the ETag up front for comparison.
    """
    page = FileResponse(
        _UI_FILE, media_type="text/html", headers=_UI_HEADERS, stat_result=_UI_FILE.stat()
    )
    etag = page.headers["etag"]
    candidates = {tag.strip() for tag in request.headers.get("if-none-match", "").split(",")}
    if etag in candidates or "*" in candidates:
        return Response(status_code=304, headers={"ETag": etag, **_UI_HEADERS})
    return page


def _parse_accept(header: str) -> Dict[str, tuple]:
    """Map each explicitly listed media range to `(quality, position)`."""
    table: Dict[str, tuple] = {}
    for position, part in enumerate(header.split(",")):
        segments = [segment.strip() for segment in part.split(";")]
        media = segments[0].lower()
        quality = 1.0
        for param in segments[1:]:
            if param.startswith("q="):
                try:
                    quality = float(param[2:])
                except ValueError:
                    quality = 0.0
        table[media] = (quality, position)
    return table


def wants_html(accept_header: str) -> bool:
    """True when an Accept header prefers the Web UI over the JSON metadata.

    Split out because it decides what `GET /` returns, and getting it wrong means
    a browser sees JSON or an API client sees HTML. Browsers send
    `text/html,application/xhtml+xml,...` with no `application/json` at all, while
    curl and httpx send `*/*` or `application/json`. So: HTML only when it was
    explicitly asked for; when both types appear, the higher `q` wins, ties going
    to whichever was listed first.
    """
    table = _parse_accept((accept_header or "").lower())
    if "text/html" not in table:
        return False
    html_quality, html_position = table["text/html"]
    json_quality, json_position = table.get("application/json", (0.0, len(table)))
    if html_quality != json_quality:
        return html_quality > json_quality
    return html_position < json_position


# ==============================================================================
# Application factory
# ==============================================================================


def create_app() -> FastAPI:
    """Build the FastAPI gateway. Kept as a factory so tests can mount it."""
    app = FastAPI(
        title="SourceCheck AI",
        description="Autonomous web information retrieval agent with grounded, cited answers.",
        version="1.1.0",
    )
    # The UI is ~50KB of inline HTML/CSS/JS and compresses ~5x; small JSON is left alone.
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.get("/livez", include_in_schema=False)
    def livez() -> Dict[str, str]:
        """Process liveness for the container HEALTHCHECK - no downstream probes.

        `/health` fans out to the next hop(s) on every call; polling it from
        Docker every few seconds was pure overhead and log noise, since it
        never fails on a downstream outage anyway.
        """
        return {"status": "ok", "service": "web"}

    @app.get("/health")
    def health() -> Dict[str, Any]:
        """Liveness probe for this gateway, plus whatever the agent reports.

        The agent owns the model/bridge view of the world, so its health payload
        (provider, search provider, bridge) is merged in when it is reachable -
        that keeps the Web UI status pill honest without the gateway having to
        know how the pipeline is wired. When the agent is down the gateway is
        still alive, so it reports `degraded` rather than failing its own probe.
        """
        agent_state = _agent_health()
        payload: Dict[str, Any] = {"status": "ok", "service": "web", "agent": agent_state}

        body = agent_state.pop("body", None)
        if agent_state.get("reachable") and isinstance(body, dict):
            for key in ("provider", "search_provider", "bridge"):
                if key in body:
                    payload[key] = body[key]
            return payload

        payload["status"] = "degraded"
        # Synthesize the fields the UI's status pill reads, so an unreachable
        # agent shows "bridge unreachable" instead of a pill stuck on "checking".
        payload.setdefault("provider", "unknown")
        payload["bridge"] = {
            "endpoint": "",
            "reachable": False,
            "hint": f"Agent service unreachable at {settings.AGENT_ENDPOINT}",
        }
        return payload

    @app.get("/")
    def root(request: Request) -> Response:
        """`/` is the front door: browsers get the Web UI, API clients get metadata.

        Content negotiation on Accept rather than a redirect, so the URL in the
        address bar is the URL that answers - and `curl http://host/` still
        returns the machine-readable service description it always did.
        """
        if wants_html(request.headers.get("accept", "")):
            return _ui_response(request)
        return JSONResponse(
            {
                "service": "SourceCheck AI",
                "endpoints": {
                    "health": "/health",
                    "models": "/models",
                    "answer": "POST /answer",
                    "ui": "GET /ui",
                },
                "model_provider": settings.MODEL_PROVIDER,
                "agent": {"endpoint": settings.AGENT_ENDPOINT},
                "ui": "/ui",
            }
        )

    @app.get("/ui", include_in_schema=False)
    def ui(request: Request) -> Response:
        """Serve the browser UI. Not in OpenAPI: it is a page, not an API surface."""
        return _ui_response(request)

    @app.get("/models")
    def models() -> Response:
        """Free models the agent can use, fetched through the agent (never the bridge directly)."""
        try:
            upstream = _fetch_agent_models()
        except httpx.HTTPError as exc:
            logger.error("Agent service unreachable at %s: %s", settings.AGENT_ENDPOINT, exc)
            raise HTTPException(
                status_code=503, detail=f"Agent service unreachable at {settings.AGENT_ENDPOINT}"
            ) from exc

        if upstream.status_code >= 400:
            raise HTTPException(status_code=upstream.status_code, detail=_upstream_detail(upstream))
        try:
            return JSONResponse(content=upstream.json())
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="Invalid response from agent service") from exc

    @app.post("/answer", response_model=AgentResponse)
    def answer(req: AnswerRequest) -> Response:
        """Proxy the question to the agent service and pass its response through.

        Validation happens here first (cheap 422s for malformed input), then the
        agent runs the pipeline. The agent's own error status codes - notably 503
        for "model backend down" with its precise detail - are preserved so API
        clients see the same contract they saw when the pipeline ran in-process.
        """
        payload = req.model_dump(exclude_none=True)

        try:
            upstream = _call_agent(payload)
        except httpx.HTTPError as exc:
            # Connect errors and timeouts both land here: the gateway itself is
            # healthy, the pipeline behind it is not, which is exactly a 503.
            logger.error("Agent service unreachable at %s: %s", settings.AGENT_ENDPOINT, exc)
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Agent service unreachable at {settings.AGENT_ENDPOINT} "
                    f"({exc.__class__.__name__}: {exc})"
                ),
            ) from exc

        if upstream.status_code >= 400:
            detail = _upstream_detail(upstream)
            # 4xx is the caller's mistake (e.g. an unknown model), not a fault in the stack.
            level = logging.ERROR if upstream.status_code >= 500 else logging.INFO
            logger.log(level, "Agent returned HTTP %s: %s", upstream.status_code, detail)
            raise HTTPException(status_code=upstream.status_code, detail=detail)

        try:
            return JSONResponse(status_code=upstream.status_code, content=upstream.json())
        except ValueError as exc:
            logger.error("Agent returned a non-JSON body: %s", exc)
            raise HTTPException(status_code=502, detail="Invalid response from agent service") from exc

    @app.exception_handler(HTTPException)
    def _json_errors(_request, exc: HTTPException) -> JSONResponse:
        """Keep error bodies JSON-shaped so clients never have to parse text."""
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app


def _agent_health_url() -> str:
    """Liveness URL of the agent service."""
    return f"{settings.AGENT_ENDPOINT.rstrip('/')}/health"


def _agent_health() -> Dict[str, Any]:
    """Probe the agent service. Never raises - health must not fail.

    The raw health body is returned under `body` for the /health route to merge;
    callers that only care about reachability can ignore it.
    """
    endpoint = settings.AGENT_ENDPOINT.rstrip("/")
    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get(f"{endpoint}/health")
            response.raise_for_status()
            body = response.json()
        return {"endpoint": endpoint, "reachable": True, "body": body}
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return {
            "endpoint": endpoint,
            "reachable": False,
            "hint": "Start it with: docker compose up -d",
            "error": str(exc),
        }


def _call_agent(payload: Dict[str, Any]) -> httpx.Response:
    """POST to the agent service. Split out so tests can stub the network hop."""
    with httpx.Client(timeout=settings.AGENT_TIMEOUT_SECONDS) as client:
        return client.post(f"{settings.AGENT_ENDPOINT.rstrip('/')}/answer", json=payload)


def _fetch_agent_models() -> httpx.Response:
    """GET the agent's model catalogue. Split out so tests can stub the network hop."""
    with httpx.Client(timeout=8.0) as client:
        return client.get(f"{settings.AGENT_ENDPOINT.rstrip('/')}/models")


def _upstream_detail(upstream: httpx.Response) -> Any:
    """Extract the agent's error detail, whatever shape it arrived in."""
    try:
        detail = upstream.json().get("detail")
    except Exception:  # noqa: BLE001 - fall back to raw text
        detail = None
    if detail is None:
        detail = (upstream.text or f"agent returned HTTP {upstream.status_code}").strip()
    return detail


# ASGI entrypoint for `uvicorn main:app`.
app = create_app()


# ==============================================================================
# CLI mode
# ==============================================================================


def run_cli(argv: Optional[list[str]] = None) -> int:
    """Answer one question from the command line and print the structured result."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="SourceCheck AI - answer a question from the command line, or serve the API.",
        epilog='examples:\n'
               '  python main.py "What year did the first human walk on the Moon?"\n'
               '  python main.py serve            # start the HTTP API on :8000\n'
               '  python main.py --provider opencode "your question"',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "question",
        nargs="?",
        help='The question to answer. Use the reserved word "serve" to start the HTTP API.',
    )
    parser.add_argument("--json", action="store_true", help="Emit raw JSON instead of a summary")
    parser.add_argument(
        "--provider",
        default=None,
        help="Override MODEL_PROVIDER for this run (mock, opencode, local_rocm, local_server)",
    )
    parser.add_argument("--serve", action="store_true", help="Start the HTTP service instead")
    args = parser.parse_args(argv)

    # `python main.py serve` and `python main.py --serve` are both accepted. Without
    # this, the bare positional swallowed "serve" as the *question* and the agent went
    # off and web-searched the word "serve" instead of listening on a port.
    if args.question is not None and args.question.strip().lower() in _SERVE_WORDS:
        args.serve = True
        args.question = None

    if args.serve:
        import uvicorn

        uvicorn.run(
            "main:app",
            host=settings.HOST,
            port=settings.PORT,
            log_level=settings.LOG_LEVEL.lower(),
        )
        return 0

    if not args.question:
        parser.error('a question is required (or run `python main.py serve` to start the HTTP service)')

    if args.provider:
        # The factory reads settings.MODEL_PROVIDER at call time, so a one-off override
        # is a plain attribute assignment rather than a re-import.
        settings.MODEL_PROVIDER = args.provider

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    agent = SourceCheckAgent(model=get_model())
    response = agent.answer_question(args.question)

    if args.json:
        print(response.model_dump_json(indent=2))
        return 0 if response.status == "success" else 1

    _print_summary(response)
    return 0 if response.status == "success" else 1


def _print_summary(response: AgentResponse) -> None:
    """Human-friendly CLI rendering of an AgentResponse."""
    status_icon = {"success": "[ok]", "insufficient_evidence": "[no evidence]", "error": "[error]"}.get(
        response.status, "[?]"
    )
    print(f"{status_icon} status: {response.status}")
    print(f"\nQ: {response.question}\n")
    print(f"A: {response.answer}")

    if response.sources:
        print("\nSources:")
        for url in response.sources:
            print(f"  - {url}")

    for index, claim in enumerate(response.claims, start=1):
        for citation in claim.citations:
            print(f"\n  {index}. \"{claim.claim}\"")
            print(f"     {citation.url}")
            if citation.quote:
                print(f"     evidence: {citation.quote[:200]}")

    if response.errors:
        print("\nDiagnostics:")
        for err in response.errors:
            print(f"  ! {err}")


def main() -> int:
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
