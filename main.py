"""
Main application entry point.

Exposes two ways to drive the agent:

* **CLI**  - `python main.py "your question"` for a single ad-hoc query.
* **HTTP** - a FastAPI service for automated graders and programmatic clients.

Run the service:

    python main.py serve
    # equivalent to: uvicorn main:app --host 0.0.0.0 --port 8000

Endpoints:

    GET  /health      liveness + configured model provider + bridge reachability
    GET  /            service metadata
    POST /answer      run the full retrieval -> evidence -> reasoning pipeline

Model selection is driven by `MODEL_PROVIDER` in the environment / `.env`:
`mock` (offline default), `opencode` (free models via bridge/opencode_bridge.py),
`local_rocm`, or `local_server`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.agent import SourceCheckAgent
from app.config import settings
from app.model import ModelError, ModelTimeoutError, get_model
from app.schemas import AgentResponse

logger = logging.getLogger("sourcecheck")

# Reserved CLI words that mean "start the HTTP service" rather than "answer this".
# Keeping them in one place documents the ambiguity instead of hiding it.
_SERVE_WORDS = frozenset({"serve", "server", "api"})


# ==============================================================================
# HTTP request schemas
# ==============================================================================


class AnswerRequest(BaseModel):
    """Payload accepted by POST /answer."""

    question: str = Field(..., min_length=1, description="Natural-language question to answer")
    max_search_results: Optional[int] = Field(
        default=None, ge=1, le=50, description="Override MAX_SEARCH_RESULTS for this request"
    )
    max_pages_to_fetch: Optional[int] = Field(
        default=None, ge=1, le=20, description="Override MAX_PAGES_TO_FETCH for this request"
    )
    max_evidence_passages: Optional[int] = Field(
        default=None, ge=1, le=50, description="Override MAX_EVIDENCE_FOR_MODEL for this request"
    )


# ==============================================================================
# Application factory
# ==============================================================================


def create_app() -> FastAPI:
    """Build the FastAPI application. Kept as a factory so tests can mount it."""
    app = FastAPI(
        title="SourceCheck AI",
        description="Autonomous web information retrieval agent with grounded, cited answers.",
        version="1.0.0",
    )

    @app.get("/health")
    def health() -> Dict[str, Any]:
        """Liveness probe plus a cheap check that the configured model is reachable."""
        provider = settings.MODEL_PROVIDER.lower().strip()
        payload: Dict[str, Any] = {
            "status": "ok",
            "provider": provider,
            "search_provider": settings.SEARCH_PROVIDER,
        }

        if provider == "opencode":
            payload["bridge"] = _bridge_health()

        return payload

    @app.get("/")
    def root() -> Dict[str, Any]:
        return {
            "service": "SourceCheck AI",
            "endpoints": {"health": "/health", "answer": "POST /answer"},
            "model_provider": settings.MODEL_PROVIDER,
        }

    @app.post("/answer", response_model=AgentResponse)
    def answer(req: AnswerRequest) -> AgentResponse:
        """Run the full pipeline. Declared `def` so FastAPI runs it in its threadpool."""
        try:
            agent = SourceCheckAgent(model=get_model())
            return agent.answer_question(
                question=req.question,
                max_search_results=req.max_search_results,
                max_pages_to_fetch=req.max_pages_to_fetch,
                max_evidence_passages=req.max_evidence_passages,
            )
        except (ModelError, ModelTimeoutError) as exc:
            # The model layer already produced a precise message - surface it as a 503
            # rather than a bare 500, since it almost always means "backend is down".
            logger.error("Model backend failure: %s", exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unhandled failure in /answer")
            raise HTTPException(status_code=500, detail="Internal agent error") from exc

    @app.exception_handler(HTTPException)
    def _json_errors(_request, exc: HTTPException) -> JSONResponse:
        """Keep error bodies JSON-shaped so clients never have to parse text."""
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app


def _bridge_health_url() -> str:
    """Map the OpenAI base URL (.../v1) to the bridge's root /health URL.

    Split out so the (non-obvious) path arithmetic is unit-testable: probing
    /v1/health 404s and makes a perfectly healthy bridge look down.
    """
    base = settings.BRIDGE_ENDPOINT.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")].rstrip("/")
    return f"{base}/health"


def _bridge_health() -> Dict[str, Any]:
    """Probe the bridge's liveness endpoint. Never raises - health must not fail."""
    url = _bridge_health_url()

    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get(url)
            response.raise_for_status()
            body = response.json()
        return {
            "endpoint": settings.BRIDGE_ENDPOINT,
            "reachable": True,
            "model": body.get("model", settings.BRIDGE_MODEL),
            "opencode_version": body.get("opencode_version", "unknown"),
        }
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return {
            "endpoint": settings.BRIDGE_ENDPOINT,
            "reachable": False,
            "hint": "Start it with: python bridge/opencode_bridge.py",
            "error": str(exc),
        }


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
