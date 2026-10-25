#!/usr/bin/env python3
"""opencode_bridge.py — OpenAI-compatible bridge from Hermes to the OpenCode CLI.

Fulfils each request by invoking an already-authenticated ``opencode run`` process and
exposes it over an OpenAI-compatible chat-completions HTTP surface, so Hermes (or any
chat-completions client) can point a *custom endpoint* provider at it and drive
OpenCode's **free** models (opencode zen: MiMo, DeepSeek, Nemotron, …).

Sibling of ``../../mac/claude_code_bridge.py`` — same wire protocol, different backend.

Highlights:
  * Free-by-default — the advertised catalogue is discovered from
    ``opencode models --verbose`` and filtered to models whose input *and* output
    cost is 0, so a mis-typed model can't silently start billing a paid provider.
    Escape hatch: ``--no-free-only`` / ``--models``.
  * Real usage — every ``step_finish`` event carries per-step tokens/cost; the bridge
    sums them across the whole agentic run and reports them in ``usage``.
  * Real streaming — ``--format json`` events become incremental
    ``chat.completion.chunk`` SSE deltas, de-duplicated by part id so re-emitted
    (growing) text parts only ever send their new suffix.
  * Prompt on stdin — never argv, so a 200k-char prompt can't hit ARG_MAX.
  * Concurrency control — a bounded semaphore caps concurrent ``opencode``
    subprocesses (free tiers rate-limit), with a fast 429 when saturated.
  * Housekeeping — ``opencode run`` always persists a session; the bridge deletes
    the sessions it created (``--keep-sessions`` to opt out).
  * Operability — process-group kill on timeout, graceful SIGTERM/SIGINT drain,
    constant-time auth, and ``/health`` / ``/config`` / ``/v1/models`` / ``/metrics``.

Stdlib only; no third-party deps, so it stays trivially portable onto a developer host
or into a slim container.

PATCH NOTES (v2.1.4 compatibility)
---------------------------------
Upstream targets a newer OpenCode CLI than v2.1.4. Applied here so the bridge runs
against the CLI actually installed on this machine:

  1. ``discover_models`` — v2.1.4 accepts neither a positional provider nor
     ``--verbose``/``--refresh``. It now tries the verbose form first and falls back
     to a plain ``opencode models`` id listing (see ``_parse_plain_models``).
  2. Free-only gating — the plain listing carries no cost metadata, so free/paid
     admission goes through ``_admit_as_free`` instead of trusting an empty
     ``cost`` object (which would have read as "free" and admitted paid models).
  3. ``build_command`` — ``--dir`` / ``--variant`` / ``--pure`` do not exist on
     v2.1.4 and yargs rejects the entire command on an unknown flag, so each is
     gated on a live ``run --help`` probe (``_detect_run_flags``).
  4. Default + fallback catalogue refreshed to the current free tier.

Upstream: https://github.com/ch-muhammad-asim/OpenDevOps/tree/main/opencode
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import hmac
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator, Optional

BRIDGE_VERSION = "1.0.0"
_IS_WINDOWS = os.name == "nt"


def default_opencode_bin() -> str:
    """Best-effort cross-platform default for the OpenCode CLI."""
    env = os.getenv("OPENCODE_BIN")
    if env:
        return env
    found = shutil.which("opencode")
    if found:
        return found
    return os.path.expanduser("~/.opencode/bin/opencode")


def resolve_opencode_bin(value: str) -> Optional[str]:
    """Return an existing path for ``value`` (direct path or a PATH lookup)."""
    if value and os.path.exists(value):
        return value
    return shutil.which(value) if value else None


#: Windows npm/pnpm shims. Spawning one makes the *real* CLI a grandchild of a
#: cmd.exe wrapper, so terminating the direct child leaves opencode.exe running
#: with inherited stdio handles - which wedges run_events' stdout read forever
#: and leaks the concurrency slot. Resolve to the actual exe where possible.
_SHIM_SUFFIXES = (".cmd", ".bat")


def resolve_real_binary(path: str) -> str:
    """Map a ``.CMD``/``.bat`` shim to the executable it wraps, when discoverable.

    npm's global shim sits in ``<npm root>/opencode.CMD`` while the real binary
    lives at ``<npm root>/node_modules/@opencode/cli/bin/opencode.exe``. If the
    candidate is not there (different package manager, non-npm layout), the
    original shim path is returned unchanged - never a path we made up.
    """
    if not path or os.path.splitext(path)[1].lower() not in _SHIM_SUFFIXES:
        return path
    # Only a shim that claims to BE opencode (npm root also holds npm.cmd,
    # npx.cmd, ...) may be rewritten - never rewrite a different tool's shim
    # to the opencode binary.
    if os.path.splitext(os.path.basename(path))[0].lower() != "opencode":
        return path
    root = os.path.dirname(path)
    candidate = os.path.join(root, "node_modules", "@opencode", "cli", "bin", "opencode.exe")
    if os.path.isfile(candidate):
        return candidate
    return path


DEFAULT_OPENCODE_BIN = default_opencode_bin()
# opencode zen's free tier — what the OpenCode desktop model picker labels "… Free".
# NOTE: this bridge ships with a v2.1.4 compatibility patch (see PATCH NOTES below);
# `opencode/mimo-v2.5-free` has since been retired from the free tier, so the default
# tracks the current MiMo free model. Discovery always re-orders the catalogue anyway.
DEFAULT_MODEL = "opencode/mimo-v2.6-flash-free"
# Providers scanned for the advertised catalogue. `opencode` is opencode zen, which
# hosts the free models; add more (comma-separated) to surface your own keys' models.
DEFAULT_MODEL_PROVIDERS = "opencode"
# Last-resort catalogue. The live list always comes from `opencode models --verbose`
# (see discover_models), and every successful discovery is persisted to --model-cache,
# which is preferred over this list when discovery fails. These IDs are therefore only
# reached on a cold start with no network AND no cache — they are a seed, not the source
# of truth, and are expected to drift as opencode zen's free tier changes.
FALLBACK_FREE_MODELS = [
    "opencode/mimo-v2.6-flash-free",
    "opencode/deepseek-v4.1-flash-free",
    "opencode/nemotron-3-ultra-free",
    "opencode/nemotron-3.5-lightning-free",
    "opencode/glm-5.3-flash-free",
    "opencode/beam-free",
    "opencode/flint-40b-free",
    "opencode/big-pickle",
]
# Free models whose id does NOT carry the "-free" suffix. Used only when the CLI
# cannot report cost metadata (see `_admit_as_free`), so a name-based heuristic
# still admits them instead of silently shrinking the catalogue.
KNOWN_FREE_IDS = {"opencode/big-pickle"}
# How often to re-run discovery so a long-lived process picks up models opencode zen
# adds or retires without a restart. 0 disables it.
DEFAULT_MODEL_REFRESH_INTERVAL = 3600
DEFAULT_MAX_PROMPT_CHARS = 200_000
# Images: chat-completions ``image_url`` parts are extracted to temp files and attached
# via ``opencode run --file``, which forwards them to the model. 0 disables image
# handling entirely (image parts are then silently dropped, as before).
# 32 MiB: full-screen Retina/5K screenshots are routinely 10-25 MiB, and Hermes
# sends screenshots uncompressed, so a 10 MiB cap rejected them with HTTP 400.
DEFAULT_MAX_IMAGES = 4
DEFAULT_MAX_IMAGE_BYTES = 32 * 1024 * 1024
DEFAULT_TIMEOUT = 300
# Free tiers rate-limit aggressively, so default lower than the Claude bridge.
DEFAULT_MAX_CONCURRENCY = 2
DEFAULT_QUEUE_WAIT = 30  # seconds a request waits for a concurrency slot before 429

_REDACT_MARKERS = ("sk-ant-", "sk-", "Bearer ", "Authorization:", "api_key", "apiKey")
# Substrings that mark an upstream error as "retry later" rather than "bad request".
_RATE_LIMIT_MARKERS = ("rate limit", "rate_limit", "429", "too many requests", "quota",
                       "overloaded", "free usage limit", "usage limit reached")


# ── helpers ──────────────────────────────────────────────────────────────────
def _configure_console_encoding() -> None:
    """Never let a log line take the server down on a non-UTF-8 console.

    A Windows console on cp1252 raises UnicodeEncodeError on any codepoint above
    U+00FF — box-drawing, arrows, em-dashes. That crashed this bridge *after* it had
    already bound its port, so the listener died looking like a startup failure.
    Degrade to '?' instead: mangled log text beats a dead bridge.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # Python < 3.7 or a replaced/wrapped stream
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError, AttributeError):
            pass


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def _text_from_content(content: Any) -> str:
    """Flatten a chat-completions message ``content`` (str or list of parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return "" if content is None else str(content)


# ── images ───────────────────────────────────────────────────────────────────
_DATA_URL_PREFIX = "data:"
_EXT_BY_MIME = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg",
    "image/webp": ".webp", "image/gif": ".gif",
}


class ImageExtractionError(ValueError):
    """A client-supplied image could not be decoded/attached (HTTP 400)."""


def _decode_image_url(url: str, index: int) -> tuple[bytes, Optional[str]]:
    """Return (bytes, mime) for a chat-completions ``image_url`` value."""
    url = (url or "").strip()
    if not url:
        raise ImageExtractionError(f"message contains an empty image_url (image #{index + 1})")
    if url.startswith(_DATA_URL_PREFIX):
        header, _, payload = url.partition(",")
        mime = header[len(_DATA_URL_PREFIX):].split(";", 1)[0].strip().lower() or None
        try:
            # Some clients line-wrap base64 payloads; strip whitespace before the
            # strict decode or a perfectly good image is rejected as corrupt.
            data = base64.b64decode(re.sub(r"\s+", "", payload), validate=True)
        except Exception as exc:
            raise ImageExtractionError(f"image #{index + 1}: invalid base64 data ({exc})") from exc
        return data, mime
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        path = urllib.request.url2pathname(parsed.path)
        try:
            with open(path, "rb") as handle:
                return handle.read(), None
        except OSError as exc:
            raise ImageExtractionError(f"image #{index + 1}: cannot read {path!r} ({exc})") from exc
    if parsed.scheme in ("http", "https"):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "opencode-bridge"})
            with urllib.request.urlopen(request, timeout=30) as response:
                mime = (response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
                return response.read(), mime or None
        except OSError as exc:
            raise ImageExtractionError(f"image #{index + 1}: fetch failed ({exc})") from exc
    raise ImageExtractionError(
        f"image #{index + 1}: unsupported image_url scheme {parsed.scheme!r} "
        "(expected a data:, file: or http(s): URL)")


def extract_images(messages: list[dict[str, Any]], max_images: int,
                   max_bytes_per_image: int) -> list[tuple[bytes, Optional[str]]]:
    """Collect ``(data, mime)`` for every ``image_url`` content part in ``messages``.

    Order follows the conversation; each image is returned once even if a client
    re-sends history across turns.
    """
    # Count first, download later: an over-limit request must be rejected before
    # any data: payload is decoded or any http(s) image is fetched.
    declared = 0
    for message in messages or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                declared += 1
    if max_images <= 0:
        if declared:
            raise ImageExtractionError(
                "this bridge has image support disabled (--max-images 0); resend text-only")
        return []
    if declared > max_images:
        raise ImageExtractionError(
            f"too many images: {declared} exceeds the limit of {max_images} per request")

    found: list[tuple[bytes, Optional[str]]] = []
    for message in messages or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not (isinstance(item, dict) and item.get("type") == "image_url"):
                continue
            url = ""
            inner = item.get("image_url")
            if isinstance(inner, dict):
                url = str(inner.get("url") or "")
            elif isinstance(inner, str):
                url = inner
            found.append(_decode_image_url(url, len(found)))
    oversized = [i + 1 for i, image in enumerate(found) if len(image[0]) > max_bytes_per_image]
    if oversized:
        raise ImageExtractionError(
            f"image(s) {oversized} exceed the per-image limit of {max_bytes_per_image // (1024 * 1024)} MiB")
    return found


def stage_image_files(images: list[tuple[bytes, Optional[str]]]) -> list[str]:
    """Write extracted images to temp files for ``opencode run --file``.

    Returns absolute paths; cleanup is delegated to ``tempfile.mkdtemp`` via
    :func:`cleanup_image_files` so a crashed run can never leave them behind.
    """
    directory = tempfile.mkdtemp(prefix="opencode-bridge-img-")
    paths: list[str] = []
    for index, (data, mime) in enumerate(images):
        ext = _EXT_BY_MIME.get((mime or "").lower())
        if not ext:
            head = bytes(data[:16])
            sniffed = ("png" if head.startswith(b"\x89PNG\r\n\x1a\n")
                       else "jpg" if head.startswith(b"\xff\xd8")
                       else "gif" if head.startswith(b"GIF8")
                       else "webp" if head[0:4] == b"RIFF" and head[8:12] == b"WEBP"
                       else "")
            ext = f".{sniffed}" if sniffed else ".bin"
        path = os.path.join(directory, f"image-{index + 1}{ext}")
        with open(path, "wb") as handle:
            handle.write(data)
        paths.append(path)
    return paths


def cleanup_image_files(paths: Optional[list[str]]) -> None:
    """Remove a staged temp directory (best-effort)."""
    if not paths:
        return
    parent = os.path.dirname(paths[0])
    shutil.rmtree(parent, ignore_errors=True)


def build_prompt(messages: list[dict[str, Any]]) -> str:
    """Flatten chat-completions messages into a single OpenCode prompt.

    ``opencode run`` has no ``--append-system-prompt``, so system turns are hoisted
    into a delimited block at the top of the prompt; user/assistant turns keep their
    roles so multi-turn context survives. A turn whose content is ONLY image parts
    yields the ``(see attached image)`` placeholder — otherwise it would be skipped
    as empty and the request rejected as prompt-less despite valid attachments.
    """
    system_parts: list[str] = []
    convo: list[str] = []
    has_images = False
    for message in messages or []:
        role = str(message.get("role") or "user").lower()
        content = message.get("content")
        if isinstance(content, list) and any(
                isinstance(part, dict) and part.get("type") == "image_url" for part in content):
            has_images = True
        text = _text_from_content(content).strip()
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
        else:
            convo.append(f"{role.upper()}:\n{text}")
    blocks: list[str] = []
    if system_parts:
        blocks.append("<system-instructions>\n" + "\n\n".join(system_parts) + "\n</system-instructions>")
    blocks.append("\n\n".join(convo))
    prompt = "\n\n".join(b for b in blocks if b).strip()
    if not prompt and has_images:
        return "(see attached image)"  # attachments ride separately via --file=
    return prompt


def _safe_error_text(text: str) -> str:
    redacted = (text or "").strip()
    for marker in _REDACT_MARKERS:
        if marker in redacted:
            redacted = redacted.replace(marker, f"{marker[:3]}[redacted]")
    return redacted[-4000:]


def _usage_block(tokens: dict[str, int]) -> dict[str, int]:
    """Map summed OpenCode step tokens → chat-completions usage.

    OpenCode reports ``input``, ``output``, ``reasoning`` and ``cache.{read,write}``
    separately, and its own ``total`` is their sum. Cache traffic is prompt-side;
    reasoning is billed as output — so fold them in accordingly.
    """
    inp = int(tokens.get("input", 0)) + int(tokens.get("cache_read", 0)) + int(tokens.get("cache_write", 0))
    out = int(tokens.get("output", 0)) + int(tokens.get("reasoning", 0))
    return {"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out}


def _accumulate_tokens(total: dict[str, int], part: dict[str, Any]) -> None:
    """Add one ``step-finish`` part's token counts into ``total`` (in place)."""
    tok = part.get("tokens") or {}
    cache = tok.get("cache") or {}
    for key, value in (("input", tok.get("input")), ("output", tok.get("output")),
                       ("reasoning", tok.get("reasoning")),
                       ("cache_read", cache.get("read")), ("cache_write", cache.get("write"))):
        try:
            total[key] = total.get(key, 0) + int(value or 0)
        except (TypeError, ValueError):
            pass


def _finish_reason(reason: Optional[str]) -> str:
    """Map an OpenCode step-finish reason to an OpenAI ``finish_reason``."""
    return {"length": "length", "content-filter": "content_filter"}.get((reason or "").strip(), "stop")


def _tool_note(part: dict[str, Any]) -> Optional[str]:
    """Render a tool_use part as a one-line progress note (``--show-tools``)."""
    name = str(part.get("tool") or "tool")
    state = part.get("state") or {}
    args = state.get("input") if isinstance(state.get("input"), dict) else {}
    hint = ""
    # "name" covers the skill/agent tools — without it a failing `skill(name=…)` rendered
    # as a bare `skill()`, which hides *which* skill could not be loaded.
    for key in ("filePath", "path", "pattern", "command", "url", "query", "name", "description"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            hint = value.strip().replace("\n", " ")[:120]
            break
    status = str(state.get("status") or "")
    mark = "✗" if status == "error" else "›"
    return f"\n{mark} {name}({hint})\n" if status in ("completed", "error") else None


# ── model discovery ───────────────────────────────────────────────────────────
def _parse_verbose_models(output: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse ``opencode models <provider> --verbose``.

    The format is an id line (``provider/model``) followed by a pretty-printed JSON
    object whose braces sit in column 0 — so accumulate between those anchors.
    """
    models: list[tuple[str, dict[str, Any]]] = []
    pending: Optional[str] = None
    buffer: list[str] = []
    for line in (output or "").splitlines():
        if buffer:
            buffer.append(line)
            if line == "}":
                try:
                    meta = json.loads("\n".join(buffer))
                except json.JSONDecodeError:
                    meta = {}
                if pending:
                    models.append((pending, meta if isinstance(meta, dict) else {}))
                pending, buffer = None, []
            continue
        if line == "{":
            buffer = [line]
            continue
        stripped = line.strip()
        if stripped and stripped == line and "/" in stripped and not stripped.startswith(("{", "}", "[")):
            pending = stripped
    return models


def _parse_plain_models(output: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse the plain ``opencode models`` listing (one ``provider/model`` id per line).

    OpenCode CLI v2.1.4 dropped the positional provider argument and the ``--verbose``
    flag, so cost metadata is simply not available from the CLI. Callers therefore get
    an empty metadata dict and must decide free/paid via :func:`_admit_as_free`.
    """
    models: list[tuple[str, dict[str, Any]]] = []
    for line in (output or "").splitlines():
        stripped = line.strip()
        # Skip usage text, blank lines and JSON — only bare `provider/model` ids qualify.
        if not stripped or "/" not in stripped or " " in stripped:
            continue
        if stripped.startswith(("{", "}", "[", "#", "-", "/")):
            continue
        models.append((stripped, {}))
    return models


def _is_free(meta: dict[str, Any]) -> bool:
    cost = meta.get("cost") or {}
    try:
        return float(cost.get("input", 0) or 0) == 0.0 and float(cost.get("output", 0) or 0) == 0.0
    except (TypeError, ValueError):
        return False


def _admit_as_free(model_id: str, meta: dict[str, Any]) -> bool:
    """Decide whether a discovered model may enter a free-only catalogue.

    Two paths, in order of reliability:

    1. **Cost metadata present** (newer CLI's ``--verbose``) — trust it exactly. Note
       an *empty* metadata dict must NOT fall through to ``_is_free`` here, because a
       missing ``cost`` object would read as "cost 0" and silently admit paid models.
    2. **No metadata** (v2.1.4's plain listing) — conservative name heuristic: the
       model id must carry the ``-free`` suffix, or sit on the explicit allowlist.
       Over-admitting is impossible; a genuinely free model with an odd name can
       still be reached through ``--models`` / ``OPENCODE_BRIDGE_MODELS``.
    """
    if meta.get("cost"):
        return _is_free(meta)
    return model_id.endswith("-free") or model_id in KNOWN_FREE_IDS


def discover_models(opencode_bin: str, providers: list[str], free_only: bool,
                    timeout: int = 60, refresh: bool = True) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Return (model_ids, metadata_by_id) from the OpenCode model catalogue.

    Tries the verbose form first (``opencode models <provider> --verbose [--refresh]``),
    which carries cost metadata so free/paid can be judged exactly. On CLIs that reject
    those flags — v2.1.4 and older only accept a bare ``opencode models`` — it falls back
    to the plain id listing and gates on :func:`_admit_as_free`.

    ``refresh`` re-pulls the catalogue from models.dev so a newly published free model is
    seen; it needs network, so on failure the same command is retried without it.
    """
    ids: list[str] = []
    meta_by_id: dict[str, dict[str, Any]] = {}
    for provider in providers or [""]:
        verbose = [opencode_bin, "models"] + ([provider] if provider else []) + ["--verbose"]
        attempts = ([verbose + ["--refresh"]] if refresh else []) + [verbose, [opencode_bin, "models"]]
        parsed: list[tuple[str, dict[str, Any]]] = []
        for cmd in attempts:
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            except (OSError, subprocess.SubprocessError):
                continue
            if proc.returncode != 0 or not proc.stdout.strip():
                # Rejected flag/arg — try the next (cheaper) invocation form.
                continue
            # Distinguish the JSON-verbose listing from the plain id listing.
            if '"cost"' in proc.stdout or '"providerID"' in proc.stdout:
                parsed = _parse_verbose_models(proc.stdout)
            else:
                parsed = _parse_plain_models(proc.stdout)
            if parsed:
                break
        for model_id, meta in parsed:
            if model_id in meta_by_id:
                continue
            if free_only and not _admit_as_free(model_id, meta):
                continue
            ids.append(model_id)
            meta_by_id[model_id] = meta
    return ids, meta_by_id


def _cache_payload(models: list[str], meta: dict[str, dict[str, Any]], free_only: bool) -> dict[str, Any]:
    """Shape the discovered catalogue for the on-disk cache."""
    entries: dict[str, Any] = {}
    for model_id in models:
        info = meta.get(model_id) or {}
        limit = info.get("limit") or {}
        entries[model_id] = {
            "name": info.get("name") or model_id,
            "context_length": limit.get("context"),
            "free": _is_free(info) if info else None,
        }
    return {"updated_at": int(time.time()), "free_only": free_only, "models": entries}


def write_model_cache(path: str, models: list[str], meta: dict[str, dict[str, Any]],
                      free_only: bool) -> None:
    """Persist a successful discovery so a later cold start isn't stuck on the seed list.

    Also read by the deployment's ``init-hermes-config`` to build the dashboard's model
    picker, which is why the file carries context lengths and not just IDs.
    """
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as handle:
            # NOT sort_keys: discovery order is meaningful — the first entry is what a
            # retired default model falls back to, and it drives the picker's ordering.
            json.dump(_cache_payload(models, meta, free_only), handle, indent=2)
        os.replace(tmp, path)  # atomic: a reader never sees a half-written file
    except OSError as exc:
        print(f"[opencode-bridge] warning: could not write model cache {path}: {exc}", flush=True)


def read_model_cache(path: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Load a previously discovered catalogue. Returns ([], {}) when unusable."""
    if not path:
        return [], {}
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        entries = payload.get("models")
        if not isinstance(entries, dict):
            return [], {}
        ids = [str(k) for k in entries]
        # Re-shape into the same metadata form discovery produces, so /v1/models keeps
        # reporting context_length and the free flag from cached data.
        meta = {}
        for model_id, info in entries.items():
            info = info if isinstance(info, dict) else {}
            meta[str(model_id)] = {
                "name": info.get("name"),
                "providerID": str(model_id).split("/", 1)[0],
                "limit": {"context": info.get("context_length")},
                "cost": {"input": 0, "output": 0} if info.get("free") else {},
            }
        return ids, meta
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return [], {}


# ── the OpenCode invocation ───────────────────────────────────────────────────
class OpenCodeError(RuntimeError):
    def __init__(self, message: str, exit_code: int = 502) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def classify_error(message: str) -> int:
    """502 for a generic upstream failure, 429 when the free tier is rate-limiting."""
    lowered = (message or "").lower()
    return 429 if any(marker in lowered for marker in _RATE_LIMIT_MARKERS) else 502


def _status_or_classify(info: dict[str, Any], message: str) -> int:
    """Map an OpenCode error event to an HTTP status.

    Prefers the event's own ``status`` when it is one of the codes we already treat as
    transient/throttlable; otherwise falls back to keyword classification. Anything
    outside that set (400, 404, 422 ...) would be misleading as an upstream failure
    code, so it degrades to the keyword path rather than being passed through.
    """
    status = info.get("status")
    if isinstance(status, int) and status in (408, 429, 500, 502, 503, 504):
        return status
    return classify_error(message)


def _with_retry_hint(message: str, info: dict[str, Any]) -> str:
    """Append a human-readable wait time when the error carries ``retryAfterMs``.

    ``Free usage limit reached`` alone is unactionable; telling the caller *when* it
    clears is the difference between "retry in a loop" and "come back later".
    """
    retry = info.get("retryAfterMs")
    try:
        # Key is `retryAfterMs` — milliseconds. Reading it as seconds would have
        # advertised a 17h wait as "17185h".
        seconds = int(retry) // 1000
    except (TypeError, ValueError):
        return message
    if seconds <= 0:
        return message
    if seconds >= 3600:
        wait = f"{seconds // 3600}h"      # coarse-grained: hours matter, minutes don't
    elif seconds >= 60:
        wait = f"{seconds // 60}m"
    else:
        wait = f"{seconds}s"
    return f"{message} (retry in {wait})"


def _detect_run_flags(opencode_bin: str) -> set[str]:
    """Return the long flags ``opencode run`` actually understands on this CLI.

    Probes ``opencode run --help`` once at startup. OpenCode v2.1.4 accepts only a
    short flag set (no ``--dir`` / ``--variant`` / ``--pure``), while newer releases
    add them — and yargs rejects the *whole* command on an unknown flag, so guessing
    wrong turns every request into a usage error. An empty set means the probe itself
    failed, in which case callers stay permissive (the modern spelling) rather than
    dropping flags that are probably supported.
    """
    try:
        proc = subprocess.run([opencode_bin, "run", "--help"],
                              capture_output=True, text=True, timeout=15)
        text = (proc.stdout or "") + (proc.stderr or "")
    except Exception:  # noqa: BLE001 — best-effort probe
        return set()
    return set(re.findall(r"--[a-z][a-z0-9-]+", text))


def build_command(cfg: "BridgeConfig", model: str, image_files: Optional[list[str]] = None) -> list[str]:
    cmd = [cfg.opencode_bin, "run", "--format", "json", "--model", model]
    if cfg.auto_approve_flag:
        cmd.append(cfg.auto_approve_flag)
    if cfg.show_reasoning:
        cmd.append("--thinking")
    if cfg.agent:
        cmd += ["--agent", cfg.agent]
    # v2.1.4 has neither flag; emitting them anyway fails the entire invocation.
    if cfg.variant and cfg.supports("--variant"):
        cmd += ["--variant", cfg.variant]
    if cfg.pure and cfg.supports("--pure"):
        cmd.append("--pure")
    # Attachments must be ``--file=<path>`` (not ``--file <path>``): yargs would
    # otherwise eat the following positional as a second file name.
    for path in image_files or []:
        cmd.append(f"--file={path}")
    # ``--dir`` scopes the agent's tools; the subprocess cwd is set separately in
    # _popen, so omitting it on CLIs that lack it still honours cfg.cwd.
    if cfg.supports("--dir"):
        cmd += ["--dir", cfg.cwd]
    return cmd


def _popen(cmd: list[str], cfg: "BridgeConfig") -> subprocess.Popen:
    # Put the child in its own group/session so a timeout can kill the whole tree
    # (opencode spawns tool subprocesses), portably across POSIX and Windows.
    kwargs: dict[str, Any] = {}
    if _IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        cmd, cwd=cfg.cwd, text=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        **kwargs,
    )
    if _IS_WINDOWS:
        _assign_kill_on_close_job(proc)
    return proc


# Handles of Windows Job Objects created for children. Kept alive for the life of
# the process on purpose: closing one (explicitly, or by process death) kills the
# whole tree under it - JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. That is what reaps
# opencode.exe when the bridge itself is force-killed or crashes, since plain
# Stop-Process on the bridge would otherwise orphan every in-flight run.
_JOB_HANDLES: list[int] = []
_JOB_LIMIT_KILL_ON_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _kernel32() -> Any:
    """kernel32 with prototypes pinned.

    Without argtypes/restype ctypes passes HANDLEs as 32-bit ints, truncating
    64-bit values - the job would silently fail to be created or assigned and
    the safety net would never fire.
    """
    from ctypes import wintypes

    k32 = ctypes.windll.kernel32
    k32.CreateJobObjectW.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int,
                                            ctypes.c_void_p, ctypes.c_uint32)
    k32.SetInformationJobObject.restype = wintypes.BOOL
    k32.OpenProcess.argtypes = (ctypes.c_uint32, wintypes.BOOL, wintypes.DWORD)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    k32.AssignProcessToJobObject.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    k32.CloseHandle.restype = wintypes.BOOL
    return k32


def _assign_kill_on_close_job(proc: subprocess.Popen) -> bool:
    """Windows best-effort: tie the child's fate to this process's lifetime."""
    try:
        k32 = _kernel32()
        handle = k32.CreateJobObjectW(None, None)
        if not handle:
            return False
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_LIMIT_KILL_ON_CLOSE
        if not k32.SetInformationJobObject(
                handle, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info), ctypes.sizeof(info)):
            k32.CloseHandle(handle)
            return False
        # Assign by PID rather than proc._handle to avoid a private-attr dependency.
        # PROCESS_SET_QUOTA|PROCESS_TERMINATE is the documented minimum for
        # AssignProcessToJobObject; anything less fails with access denied.
        process = k32.OpenProcess(0x0101, False, proc.pid)
        if not process or not k32.AssignProcessToJobObject(handle, process):
            if process:
                k32.CloseHandle(process)
            k32.CloseHandle(handle)
            return False
        k32.CloseHandle(process)
        _JOB_HANDLES.append(handle)
        return True
    except (AttributeError, OSError, ValueError):
        return False  # job objects are a safety net; the explicit kill path still works


def _kill(proc: subprocess.Popen) -> None:
    """Terminate the child AND its descendants, escalating to kill.

    The child here is often a ``cmd.exe`` shim (``opencode.CMD``): terminating
    only that layer leaves the real opencode.exe running with inherited stdio
    handles, so run_events keeps blocking on a stdout pipe that will never EOF
    and the concurrency slot leaks. On Windows ``taskkill /T`` walks the tree.
    """
    try:
        if _IS_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=15)
            try:
                proc.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
                pass  # fall through to the direct terminate/kill escalation
        elif hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError, subprocess.SubprocessError):
        pass


def delete_session(cfg: "BridgeConfig", session_id: str) -> None:
    """Best-effort cleanup of a session ``opencode run`` persisted for us."""
    if not session_id or not cfg.delete_sessions:
        return
    try:
        subprocess.run([cfg.opencode_bin, "session", "delete", session_id],
                       cwd=cfg.cwd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pass


def run_events(cfg: "BridgeConfig", model: str, prompt: str,
               image_files: Optional[list[str]] = None) -> Iterator[dict[str, Any]]:
    """Run ``opencode run`` and yield normalized events.

    Yields ``{'delta': str}`` for assistant text (and tool/reasoning notes when
    enabled), then a final ``{'done': True, 'tokens': {...}, 'cost_usd': float,
    'finish_reason': str}``. The prompt goes in on stdin so it is never argv-bound.
    ``image_files`` are staged temp files attached via ``--file=<path>`` and the
    whole staging directory is removed when the run ends.
    """
    cmd = build_command(cfg, model, image_files)
    proc = _popen(cmd, cfg)
    deadline = time.time() + cfg.timeout_seconds
    tokens: dict[str, int] = {}
    cost = 0.0
    finish = "stop"
    session_id = ""
    # Text parts may be re-emitted as they grow; remember how much of each part id
    # we have already forwarded so a delta is only ever the new suffix.
    emitted: dict[str, int] = {}
    error: Optional[OpenCodeError] = None
    # Drain stderr CONTINUOUSLY: reading it only after exit deadlocks whenever
    # opencode writes more than the OS pipe buffer (~64 KiB) holds.
    stderr_chunks: list[str] = []

    def _writer() -> None:
        try:
            assert proc.stdin is not None
            proc.stdin.write(prompt)
            proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass

    def _drain_stderr() -> None:
        try:
            assert proc.stderr is not None
            for raw in iter(proc.stderr.readline, ""):
                stderr_chunks.append(raw)
        except (OSError, ValueError):
            pass

    def _watchdog() -> None:
        # The loop below only checks the deadline when a line ARRIVES, so an
        # opencode that wedges mid-run would hang the request forever. Enforce
        # the deadline here: killing the process EOFs the pipe and unblocks it.
        while time.time() < deadline and proc.poll() is None:
            time.sleep(0.5)
        if proc.poll() is None:
            _kill(proc)

    threading.Thread(target=_writer, daemon=True).start()
    threading.Thread(target=_drain_stderr, daemon=True).start()
    threading.Thread(target=_watchdog, daemon=True).start()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if time.time() > deadline:
                raise OpenCodeError("opencode command timed out", exit_code=504)
            line = line.strip()
            if not line or not line.startswith("{"):
                continue  # CLI banners / progress chrome — not events
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            session_id = event.get("sessionID") or session_id
            etype = event.get("type")
            part = event.get("part") or {}
            if etype in ("text", "reasoning"):
                text = part.get("text") or ""
                if etype == "reasoning" and not cfg.show_reasoning:
                    continue
                seen = emitted.get(part.get("id") or "", 0)
                if len(text) > seen:
                    emitted[part.get("id") or ""] = len(text)
                    chunk = text[seen:]
                    yield {"delta": f"\n_thinking:_ {chunk}\n" if etype == "reasoning" else chunk}
            elif etype == "tool_use" and cfg.show_tools:
                note = _tool_note(part)
                if note:
                    yield {"delta": note}
            elif etype == "step_finish":
                _accumulate_tokens(tokens, part)
                try:
                    cost += float(part.get("cost") or 0)
                except (TypeError, ValueError):
                    pass
                finish = _finish_reason(part.get("reason"))
            elif etype == "error":
                info = event.get("error") or {}
                data = info.get("data") if isinstance(info.get("data"), dict) else {}
                # v2.1.4 puts `message` directly on the error object (and `status`,
                # `retryAfterMs` beside it); older shapes nest it under `data.message`
                # or expose only `name`. Reading only the nested forms is what turned
                # every upstream failure into the useless "opencode error".
                message = str(
                    info.get("message")
                    or (data or {}).get("message")
                    or info.get("name")
                    or info.get("type")
                    or "opencode error"
                )
                # The upstream HTTP status is authoritative for classification: a 429
                # quota/rate-limit must surface as 429 even when its prose contains no
                # rate-limit keyword (e.g. "Free usage limit reached").
                error = OpenCodeError(
                    _safe_error_text(_with_retry_hint(message, info)),
                    _status_or_classify(info, message),
                )
        try:
            rc = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _kill(proc)  # outlived its own exit grace: SIGTERM/KILL the tree, then reap
            rc = proc.wait(timeout=10)
        if error is not None:
            raise error
        if rc != 0:
            # The watchdog kills a wedged run once the deadline passes; the child then
            # dies by signal (rc < 0) with an empty stderr. Report that as the 504 it
            # is, not a generic 502 "exited with code -15".
            if time.time() > deadline:
                raise OpenCodeError("opencode command timed out", exit_code=504)
            err = "".join(stderr_chunks).strip()
            message = _safe_error_text(err or f"opencode exited with code {rc}")
            raise OpenCodeError(message, classify_error(message))
    finally:
        if proc.poll() is None:
            _kill(proc)
        if session_id:
            threading.Thread(target=delete_session, args=(cfg, session_id), daemon=True).start()
        cleanup_image_files(image_files)
    yield {"done": True, "tokens": tokens, "cost_usd": cost, "finish_reason": finish}


def run_blocking(cfg: "BridgeConfig", model: str, prompt: str,
                 image_files: Optional[list[str]] = None) -> dict[str, Any]:
    """Drain ``run_events`` and return {'text','tokens','cost_usd','finish_reason'}."""
    chunks: list[str] = []
    tokens: dict[str, int] = {}
    cost = 0.0
    finish = "stop"
    for event in run_events(cfg, model, prompt, image_files):
        if "delta" in event:
            chunks.append(event["delta"])
        elif event.get("done"):
            tokens, cost, finish = event["tokens"], event["cost_usd"], event["finish_reason"]
    return {"text": "".join(chunks).strip(), "tokens": tokens, "cost_usd": cost, "finish_reason": finish}


# ── config ────────────────────────────────────────────────────────────────────
class BridgeConfig:
    def __init__(self, args: argparse.Namespace) -> None:
        self.opencode_bin = args.opencode_bin
        self.cwd = args.cwd
        self.default_model = args.model
        self.model_providers = _split_csv(args.model_providers)
        self.free_only = args.free_only
        self.agent = args.agent
        self.variant = args.variant
        self.auto_approve = args.auto_approve
        # The auto-approve flag was renamed across OpenCode releases (`--auto` →
        # `--dangerously-skip-permissions`), and yargs rejects the WHOLE command on an
        # unknown flag — Hermes then shows the CLI usage text instead of a reply. main()
        # re-resolves this against `opencode run --help`; the modern spelling is the
        # offline default so selfcheck needs no opencode binary.
        self.auto_approve_flag = "--dangerously-skip-permissions" if args.auto_approve else ""
        self.show_tools = args.show_tools
        self.show_reasoning = args.show_reasoning
        self.pure = args.pure
        self.delete_sessions = args.delete_sessions
        self.max_prompt_chars = args.max_prompt_chars
        self.max_images = args.max_images
        self.max_image_bytes = args.max_image_bytes
        self.timeout_seconds = args.timeout
        self.pass_model = args.pass_model
        self.api_key = args.api_key
        # Long flags `opencode run` supports on this CLI. main() overwrites this with
        # the live probe result; the permissive default keeps offline use (and the
        # selfcheck, which never launches a subprocess) behaving as before.
        self.run_flags: set[str] = {"--dir", "--variant", "--pure", "--thinking", "--agent"}
        # Advertised catalogue: explicit --models wins, else discovery, else fallback.
        # The default model is always first so a client can never pick something the
        # bridge would refuse to run.
        self.model_cache = args.model_cache
        self.model_refresh_interval = args.model_refresh_interval
        # Catalogue mutates from the refresh thread while requests read it.
        self._lock = threading.Lock()
        configured = _split_csv(args.models)
        self.discovered_from = "flag" if configured else "pending"
        self.models = self._order(configured)
        self.model_meta: dict[str, dict[str, Any]] = {}
        # Whether the default model came from the catalogue itself rather than
        # only from being force-prepended (drives a startup warning).
        self.default_in_catalogue = args.model in configured

    def _order(self, ids: list[str]) -> list[str]:
        ordered = [self.default_model] + [m for m in ids if m != self.default_model]
        seen: set[str] = set()
        return [m for m in ordered if not (m in seen or seen.add(m))]

    def supports(self, flag: str) -> bool:
        """True when this CLI's ``run --help`` advertises ``flag``.

        A failed probe yields an empty set, in which case we stay permissive: an
        unnecessary flag is a better default than silently dropping a useful one.
        """
        return flag in self.run_flags if self.run_flags else True

    def refresh_catalogue(self, *, refresh: bool = True) -> dict[str, Any]:
        """Repoint the advertised catalogue at whatever the CLI currently offers.

        Precedence: live discovery → on-disk cache → the seed list. Called at startup and
        again on a timer, so free models opencode zen adds appear and retired ones drop
        out without a restart. Returns a small summary for logging/the refresh endpoint.
        """
        if self.discovered_from == "flag":
            return {"source": "flag", "models": len(self.models), "changed": False}

        before = list(self.models)
        ids, meta = discover_models(self.opencode_bin, self.model_providers, self.free_only,
                                    refresh=refresh)
        if ids:
            source = "opencode-cli"
            write_model_cache(self.model_cache, ids, meta, self.free_only)
        else:
            ids, meta = read_model_cache(self.model_cache)
            source = "cache" if ids else "fallback"
            if not ids:
                ids, meta = list(FALLBACK_FREE_MODELS), {}

        with self._lock:
            self.model_meta = meta
            self.discovered_from = source
            self.default_in_catalogue = self.default_model in ids
            # If the configured default is gone from the catalogue (retired from the free
            # tier, renamed, typo'd), adopt the first discovered model rather than keep
            # advertising something the CLI can no longer run.
            if not self.default_in_catalogue and self.free_only and ids:
                previous = self.default_model
                self.default_model = ids[0]
                print(f"[opencode-bridge] default model {previous!r} is no longer in the "
                      f"{source} catalogue - falling back to {self.default_model!r}", flush=True)
            self.models = self._order(ids)

        added = [m for m in self.models if m not in before]
        removed = [m for m in before if m not in self.models]
        if before and (added or removed):
            print(f"[opencode-bridge] catalogue changed via {source}: "
                  f"+{added or '[]'} -{removed or '[]'}", flush=True)
        return {"source": source, "models": len(self.models), "default": self.default_model,
                "added": added, "removed": removed, "changed": bool(added or removed)}

    def start_refresh_loop(self) -> None:
        """Re-discover on a timer so the catalogue tracks the free tier by itself."""
        if self.model_refresh_interval <= 0 or self.discovered_from == "flag":
            return

        def _loop() -> None:
            while True:
                time.sleep(self.model_refresh_interval)
                try:
                    self.refresh_catalogue()
                except Exception as exc:  # noqa: BLE001 — a refresh must never kill serving
                    print(f"[opencode-bridge] model refresh failed: {exc}", flush=True)

        threading.Thread(target=_loop, daemon=True, name="model-refresh").start()

    def resolve_model(self, requested: str) -> tuple[Optional[str], Optional[str]]:
        """Return (model_id, error). Accepts bare ids and ``provider/model`` ids."""
        if not self.pass_model:
            return self.default_model, None
        wanted = (requested or "").strip()
        if not wanted:
            return self.default_model, None
        if wanted in self.models:
            return wanted, None
        # Convenience aliases: bare "mimo-v2.5-free" → "opencode/mimo-v2.5-free",
        # and a case-insensitive match, so Hermes' free-text model box is forgiving.
        if "/" not in wanted:
            for provider in self.model_providers or ["opencode"]:
                candidate = f"{provider}/{wanted}"
                if candidate in self.models:
                    return candidate, None
        for model in self.models:
            if model.lower() == wanted.lower():
                return model, None
        if not self.free_only and "/" in wanted:
            return wanted, None  # unrestricted mode: trust the client, let opencode judge
        return None, (
            f"model {wanted!r} is not available on this bridge. It serves "
            f"{'free ' if self.free_only else ''}OpenCode models: {', '.join(self.models[:12])}"
            f"{' …' if len(self.models) > 12 else ''}. See GET /v1/models."
        )


# ── metrics ────────────────────────────────────────────────────────────────────
class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.rejected_busy = 0
        self.total_cost_usd = 0.0
        self.total_output_tokens = 0
        self.started = time.time()

    def record(self, *, error: bool = False, cost: Optional[float] = None, out_tokens: int = 0) -> None:
        with self._lock:
            self.requests += 1
            if error:
                self.errors += 1
            if cost:
                self.total_cost_usd += float(cost)
            self.total_output_tokens += int(out_tokens or 0)

    def snapshot(self, in_flight: int) -> dict[str, Any]:
        with self._lock:
            return {
                "uptime_s": int(time.time() - self.started),
                "requests": self.requests,
                "errors": self.errors,
                "rejected_busy": self.rejected_busy,
                "in_flight": in_flight,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "total_output_tokens": self.total_output_tokens,
            }


# ── HTTP handler ────────────────────────────────────────────────────────────────
class BridgeHandler(BaseHTTPRequestHandler):
    server_version = f"OpenCodeBridge/{BRIDGE_VERSION}"

    @property
    def cfg(self) -> BridgeConfig:
        return self.server.cfg  # type: ignore[attr-defined]

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _authorized(self) -> bool:
        expected = str(self.cfg.api_key or "")
        if not expected:
            return True
        bearer = (self.headers.get("authorization") or "").removeprefix("Bearer ").strip()
        x_api_key = (self.headers.get("x-api-key") or "").strip()
        return hmac.compare_digest(bearer, expected) or hmac.compare_digest(x_api_key, expected)

    def log_message(self, fmt: str, *args: Any) -> None:  # structured-ish access log
        print(f"[opencode-bridge] {self.address_string()} - {fmt % args}", flush=True)

    def _model_obj(self, model_id: str) -> dict[str, Any]:
        meta = self.cfg.model_meta.get(model_id) or {}
        obj: dict[str, Any] = {"id": model_id, "object": "model", "created": 0,
                               "owned_by": (meta.get("providerID") or "opencode-cli")}
        limit = meta.get("limit") or {}
        if limit.get("context"):
            obj["context_length"] = limit["context"]
        if meta.get("name"):
            obj["display_name"] = meta["name"]
        if meta:
            obj["free"] = _is_free(meta)
        return obj

    # ── GET ──
    def do_GET(self) -> None:
        if not self._authorized():
            self._send_json(401, {"error": {"message": "unauthorized"}})
            return
        path = self.path.rstrip("/")
        if path == "/health":
            self._send_json(200, {"status": "ok", "version": BRIDGE_VERSION,
                                  "opencode_version": self.server.opencode_version,  # type: ignore[attr-defined]
                                  "in_flight": self.server.in_flight(),  # type: ignore[attr-defined]
                                  "model": self.cfg.default_model})
            return
        if path == "/metrics":
            self._send_json(200, self.server.metrics.snapshot(self.server.in_flight()))  # type: ignore[attr-defined]
            return
        if path == "/config":
            self._send_json(200, {
                "status": "ok", "version": BRIDGE_VERSION, "model": self.cfg.default_model,
                "models": self.cfg.models, "models_source": self.cfg.discovered_from,
                "model_refresh_interval_s": self.cfg.model_refresh_interval,
                "model_cache": self.cfg.model_cache or "(disabled)",
                "model_providers": self.cfg.model_providers, "free_only": self.cfg.free_only,
                "opencode_bin": self.cfg.opencode_bin, "cwd": self.cfg.cwd,
                "agent": self.cfg.agent or "(opencode default)", "variant": self.cfg.variant or "(default)",
                "auto_approve": self.cfg.auto_approve, "show_tools": self.cfg.show_tools,
                "show_reasoning": self.cfg.show_reasoning, "delete_sessions": self.cfg.delete_sessions,
                "max_prompt_chars": self.cfg.max_prompt_chars, "timeout_s": self.cfg.timeout_seconds,
                "max_images": self.cfg.max_images,
                "max_image_bytes": self.cfg.max_image_bytes,
                "max_concurrency": self.server.max_concurrency,  # type: ignore[attr-defined]
                "pass_model": self.cfg.pass_model, "api_key_required": bool(self.cfg.api_key),
            })
            return
        if path == "/v1/models":
            self._send_json(200, {"object": "list",
                                  "data": [self._model_obj(m) for m in self.cfg.models]})
            return
        if path.startswith("/v1/models/"):
            model_id = path[len("/v1/models/"):]  # may contain the provider slash
            if model_id in self.cfg.models:
                self._send_json(200, self._model_obj(model_id))
            else:
                self._send_json(404, {"error": {"message": f"model {model_id!r} not found"}})
            return
        self._send_json(404, {"error": {"message": "not found"}})

    # ── POST ──
    def do_POST(self) -> None:
        if not self._authorized():
            self._send_json(401, {"error": {"message": "unauthorized"}})
            return
        # Force an immediate catalogue re-discovery. Useful right after opencode zen
        # changes its free tier, instead of waiting for the refresh interval.
        if self.path.rstrip("/") == "/v1/models/refresh":
            summary = self.cfg.refresh_catalogue()
            self._send_json(200, {"status": "ok", **summary, "models_list": self.cfg.models})
            return
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("content-length") or "0")
            request = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"error": {"message": f"invalid json: {exc}"}})
            return

        model, model_error = self.cfg.resolve_model(str(request.get("model") or ""))
        if model_error or not model:
            self._send_json(400, {"error": {"message": model_error or "no model"}})
            return

        prompt = build_prompt(request.get("messages") or [])
        if not prompt:
            self._send_json(400, {"error": {"message": "request has no prompt text"}})
            return
        if len(prompt) > self.cfg.max_prompt_chars:
            self._send_json(413, {"error": {"message":
                f"prompt too large: {len(prompt)} chars exceeds {self.cfg.max_prompt_chars}"}})
            return
        try:
            images = extract_images(request.get("messages") or [],
                                    self.cfg.max_images, self.cfg.max_image_bytes)
        except ImageExtractionError as exc:
            self._send_json(400, {"error": {"message": str(exc)}})
            return
        image_files = stage_image_files(images) if images else []

        # Concurrency slot — bound the number of live opencode subprocesses.
        if not self.server.slot.acquire(timeout=self.server.queue_wait):  # type: ignore[attr-defined]
            self.server.metrics.rejected_busy += 1  # type: ignore[attr-defined]
            cleanup_image_files(image_files)  # staged but never handed to run_events
            self._send_json(429, {"error": {"message": "bridge busy: too many concurrent requests"}})
            return
        try:
            if request.get("stream") is True:
                self._handle_stream(model, prompt, image_files)
            else:
                self._handle_blocking(model, prompt, image_files)
        finally:
            self.server.slot.release()  # type: ignore[attr-defined]

    def _handle_blocking(self, model: str, prompt: str, image_files: Optional[list[str]] = None) -> None:
        started = time.time()
        try:
            res = run_blocking(self.cfg, model, prompt, image_files)
        except OpenCodeError as exc:
            self.server.metrics.record(error=True)  # type: ignore[attr-defined]
            self._send_json(exc.exit_code, {"error": {"message": str(exc)}})
            return
        usage = _usage_block(res["tokens"])
        self.server.metrics.record(cost=res["cost_usd"], out_tokens=usage["completion_tokens"])  # type: ignore[attr-defined]
        print(f"[opencode-bridge] model={model} chars={len(prompt)} "
              f"out_tokens={usage['completion_tokens']} cost_usd={res['cost_usd']} "
              f"latency_ms={int((time.time()-started)*1000)}", flush=True)
        self._send_json(200, {
            "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": res["text"]},
                         "finish_reason": res["finish_reason"]}],
            "usage": usage,
        })

    def _handle_stream(self, model: str, prompt: str, image_files: Optional[list[str]] = None) -> None:
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        started = time.time()

        def frame(delta: dict[str, Any], finish: Optional[str] = None) -> bytes:
            chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                     "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            return f"data: {json.dumps(chunk)}\n\n".encode("utf-8")

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(frame({"role": "assistant"}))
            tokens: dict[str, int] = {}
            cost = 0.0
            finish = "stop"
            for event in run_events(self.cfg, model, prompt, image_files):
                if "delta" in event:
                    self.wfile.write(frame({"content": event["delta"]}))
                    self.wfile.flush()
                elif event.get("done"):
                    tokens, cost, finish = event["tokens"], event["cost_usd"], event["finish_reason"]
            self.wfile.write(frame({}, finish=finish))
            self.wfile.write(b"data: [DONE]\n\n")
            ub = _usage_block(tokens)
            self.server.metrics.record(cost=cost, out_tokens=ub["completion_tokens"])  # type: ignore[attr-defined]
            print(f"[opencode-bridge] stream model={model} chars={len(prompt)} "
                  f"out_tokens={ub['completion_tokens']} cost_usd={cost} "
                  f"latency_ms={int((time.time()-started)*1000)}", flush=True)
        except OpenCodeError as exc:
            self.server.metrics.record(error=True)  # type: ignore[attr-defined]
            # Headers are already sent, so surface the failure in-band, then close.
            try:
                self.wfile.write(frame({"content": f"\n[bridge error: {exc}]"}, finish="stop"))
                self.wfile.write(b"data: [DONE]\n\n")
            except BrokenPipeError:
                pass
        except BrokenPipeError:
            pass  # client hung up mid-stream


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, cfg: BridgeConfig, max_concurrency: int, queue_wait: int,
                 opencode_version: str) -> None:
        super().__init__(addr, handler)
        self.cfg = cfg
        self.metrics = Metrics()
        self.max_concurrency = max_concurrency
        self.queue_wait = queue_wait
        self.opencode_version = opencode_version
        self.slot = threading.BoundedSemaphore(max_concurrency)

    def in_flight(self) -> int:
        # available permits on a bounded semaphore -> derive active count
        return max(0, self.max_concurrency - self.slot._value)  # type: ignore[attr-defined]


# ── entrypoint ────────────────────────────────────────────────────────────────
def _detect_auto_approve_flag(opencode_bin: str) -> Optional[str]:
    """Return the auto-approve flag this opencode actually understands.

    Probes ``opencode run --help`` once at startup: newer CLIs spell it
    ``--dangerously-skip-permissions``, older ones ``--auto``. Returns None when the
    help lists neither (the flag is then omitted — a permission prompt may stall a
    headless run, but every request breaking on an unknown flag is strictly worse).
    """
    try:
        proc = subprocess.run([opencode_bin, "run", "--help"],
                              capture_output=True, text=True, timeout=15)
        text = (proc.stdout or "") + (proc.stderr or "")
    except Exception:  # noqa: BLE001 — best-effort probe; fall back to the modern spelling
        return "--dangerously-skip-permissions"
    for flag in ("--dangerously-skip-permissions", "--auto"):
        if flag in text:
            return flag
    return None


def _detect_opencode_version(opencode_bin: str) -> str:
    try:
        out = subprocess.run([opencode_bin, "--version"], capture_output=True, text=True, timeout=15)
        return (out.stdout or out.stderr or "").strip().splitlines()[0] if (out.stdout or out.stderr) else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OpenAI-compatible bridge to the OpenCode CLI (Hermes custom endpoint; free models by default).")
    p.add_argument("--host", default=os.getenv("BRIDGE_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.getenv("BRIDGE_PORT", "18383")))
    p.add_argument("--opencode-bin", default=os.getenv("OPENCODE_BIN", DEFAULT_OPENCODE_BIN))
    p.add_argument("--cwd", default=os.getenv("OPENCODE_BRIDGE_CWD", os.getcwd()),
                   help="directory opencode runs in (its tools read/write here)")
    p.add_argument("--model", default=os.getenv("OPENCODE_BRIDGE_MODEL", DEFAULT_MODEL),
                   help="default model, provider/model (e.g. opencode/mimo-v2.5-free)")
    p.add_argument("--models", default=os.getenv("OPENCODE_BRIDGE_MODELS", ""),
                   help="comma-separated model IDs to advertise; empty = discover from the opencode CLI")
    p.add_argument("--model-providers", default=os.getenv("OPENCODE_BRIDGE_MODEL_PROVIDERS", DEFAULT_MODEL_PROVIDERS),
                   help="comma-separated provider IDs scanned during discovery (default: opencode = opencode zen)")
    p.add_argument("--model-refresh-interval", type=int,
                   default=int(os.getenv("OPENCODE_BRIDGE_MODEL_REFRESH_INTERVAL",
                                         str(DEFAULT_MODEL_REFRESH_INTERVAL))),
                   help="seconds between catalogue re-discoveries so new/retired free models "
                        "are picked up without a restart (0 disables)")
    p.add_argument("--model-cache", default=os.getenv("OPENCODE_BRIDGE_MODEL_CACHE", ""),
                   help="path to persist the discovered catalogue; used instead of the built-in "
                        "seed list when discovery fails, and read by the k8s deployment to build "
                        "the dashboard's model picker")
    # Free-only by default: a Hermes typo must not start spending on a paid provider.
    p.add_argument("--free-only", dest="free_only", action="store_true", default=True)
    p.add_argument("--no-free-only", dest="free_only", action="store_false",
                   help="also allow paid models your opencode credentials can reach")
    p.add_argument("--agent", default=os.getenv("OPENCODE_BRIDGE_AGENT", ""),
                   help="opencode agent to run (e.g. build, plan); empty = opencode default")
    p.add_argument("--variant", default=os.getenv("OPENCODE_BRIDGE_VARIANT", ""),
                   help="provider reasoning variant (e.g. high, max, minimal)")
    p.add_argument("--auto-approve", dest="auto_approve", action="store_true", default=True,
                   help="pass the CLI's auto-approve flag (--dangerously-skip-permissions, or "
                        "--auto on older CLIs) so tool permissions never block a headless run (default)")
    p.add_argument("--no-auto-approve", dest="auto_approve", action="store_false")
    p.add_argument("--show-tools", dest="show_tools", action="store_true",
                   default=os.getenv("OPENCODE_BRIDGE_SHOW_TOOLS", "") not in ("", "0", "false"),
                   help="emit one-line tool-call notes into the response text")
    p.add_argument("--show-reasoning", dest="show_reasoning", action="store_true",
                   default=os.getenv("OPENCODE_BRIDGE_SHOW_REASONING", "") not in ("", "0", "false"),
                   help="pass --thinking and include reasoning blocks in the response text")
    p.add_argument("--pure", action="store_true", help="run opencode without external plugins")
    p.add_argument("--delete-sessions", dest="delete_sessions", action="store_true", default=True,
                   help="delete each run's persisted opencode session afterwards (default)")
    p.add_argument("--keep-sessions", dest="delete_sessions", action="store_false")
    p.add_argument("--max-prompt-chars", type=int,
                   default=int(os.getenv("OPENCODE_BRIDGE_MAX_PROMPT_CHARS", str(DEFAULT_MAX_PROMPT_CHARS))))
    p.add_argument("--max-images", type=int,
                   default=int(os.getenv("OPENCODE_BRIDGE_MAX_IMAGES", str(DEFAULT_MAX_IMAGES))),
                   help="max image_url attachments accepted per request (0 disables image support)")
    p.add_argument("--max-image-bytes", type=int,
                   default=int(os.getenv("OPENCODE_BRIDGE_MAX_IMAGE_BYTES", str(DEFAULT_MAX_IMAGE_BYTES))),
                   help="per-image size limit in bytes (data: URLs and fetched http(s) images)")
    p.add_argument("--api-key", default=os.getenv("OPENCODE_BRIDGE_API_KEY", ""))
    p.add_argument("--timeout", type=int, default=int(os.getenv("OPENCODE_BRIDGE_TIMEOUT", str(DEFAULT_TIMEOUT))))
    p.add_argument("--max-concurrency", type=int,
                   default=int(os.getenv("OPENCODE_BRIDGE_MAX_CONCURRENCY", str(DEFAULT_MAX_CONCURRENCY))))
    p.add_argument("--queue-wait", type=int,
                   default=int(os.getenv("OPENCODE_BRIDGE_QUEUE_WAIT", str(DEFAULT_QUEUE_WAIT))))
    # Honor the client-requested model by default. Use --no-pass-model to pin --model.
    p.add_argument("--pass-model", dest="pass_model", action="store_true", default=True)
    p.add_argument("--no-pass-model", dest="pass_model", action="store_false")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    _configure_console_encoding()
    args = build_parser().parse_args(argv)
    resolved = resolve_opencode_bin(args.opencode_bin)
    if not resolved:
        raise SystemExit(
            f"opencode binary not found: {args.opencode_bin!r}. Install OpenCode "
            "(https://opencode.ai) and ensure `opencode` is on PATH, or pass --opencode-bin /full/path.")
    args.opencode_bin = resolved
    # Prefer the real exe over the npm .CMD shim: a cmd.exe layer between us and
    # opencode.exe means terminate()/poll() act on the wrong process, and the
    # grandchild's inherited pipes keep run_events blocked after the shim dies.
    args.opencode_bin = resolve_real_binary(args.opencode_bin)
    if not os.path.isdir(args.cwd):
        raise SystemExit(f"working directory not found: {args.cwd}")

    cfg = BridgeConfig(args)
    opencode_version = _detect_opencode_version(args.opencode_bin)
    # Discover which optional `run` flags this CLI actually accepts BEFORE building
    # any command — v2.1.4 rejects --dir/--variant/--pure, newer builds accept them.
    cfg.run_flags = _detect_run_flags(args.opencode_bin)
    if cfg.auto_approve:
        flag = _detect_auto_approve_flag(args.opencode_bin)
        cfg.auto_approve_flag = flag or ""
        if not flag:
            print("[opencode-bridge] warning: this opencode advertises no auto-approve flag "
                  "(`run --help` lists neither --dangerously-skip-permissions nor --auto) - "
                  "tool permission prompts may stall headless runs", flush=True)
    cfg.refresh_catalogue()
    cfg.start_refresh_loop()
    if not cfg.default_in_catalogue:
        # The default is always advertised (force-prepended); warn when the catalogue
        # itself didn't list it, since that usually means a typo or a paid model.
        print(f"[opencode-bridge] warning: default model {cfg.default_model!r} is not in the "
              f"{'free ' if cfg.free_only else ''}catalogue from `opencode models "
              f"{' '.join(cfg.model_providers)}` - requests may fail", flush=True)
    server = BridgeServer((args.host, args.port), BridgeHandler, cfg,
                          max(1, args.max_concurrency), args.queue_wait, opencode_version)

    def _shutdown(signum, _frame):  # graceful drain
        print(f"[opencode-bridge] signal {signum} - shutting down", flush=True)
        threading.Thread(target=server.shutdown, daemon=True).start()

    # SIGINT is available everywhere; SIGTERM/SIGBREAK vary by platform — register
    # each defensively so the bridge stays cross-platform (Linux/macOS/Windows).
    for signame in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, signame, None)
        if sig is not None:
            try:
                signal.signal(sig, _shutdown)
            except (ValueError, OSError, RuntimeError):
                pass

    print(f"[opencode-bridge] v{BRIDGE_VERSION} listening on http://{args.host}:{args.port}/v1 "
          f"model={cfg.default_model} opencode={opencode_version} cwd={args.cwd} "
          f"free_only={cfg.free_only} models={len(cfg.models)} ({cfg.discovered_from}) "
          f"max_concurrency={args.max_concurrency} api_key_required={bool(args.api_key)}", flush=True)
    connect_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    # NOTE: ASCII-only on purpose. The upstream box-drawing/arrow glyphs crash on a
    # Windows console whose codepage (cp1252) cannot encode them — UnicodeEncodeError
    # took the whole bridge down *after* it had already bound the port.
    print(
        "\n"
        "+-- Connect SourceCheck AI ------------------------------------+\n"
        "| Endpoint (OpenAI-compatible):                                |\n"
        f"|   Base URL: {f'http://{connect_host}:{args.port}/v1':<49}|\n"
        f"|   API key:  {'(required - use your bridge api-key)' if args.api_key else '(none required)':<49}|\n"
        f"|   Model:    {cfg.default_model:<49}|\n"
        "+--------------------------------------------------------------+",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _selfcheck() -> None:
    # Offline checks for the parts with real logic: prompt assembly, the OpenCode
    # token → chat-completions usage mapping, `models --verbose` parsing, model
    # resolution, error classification and command construction.
    prompt = build_prompt([
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ])
    assert prompt == "<system-instructions>\nbe terse\n</system-instructions>\n\nUSER:\nhi", repr(prompt)
    assert build_prompt([{"role": "user", "content": "hi"}]) == "USER:\nhi"

    tokens: dict[str, int] = {}
    _accumulate_tokens(tokens, {"tokens": {"input": 10, "output": 4, "reasoning": 2,
                                           "cache": {"read": 5, "write": 1}}})
    _accumulate_tokens(tokens, {"tokens": {"input": 3, "output": 1}})
    assert _usage_block(tokens) == {"prompt_tokens": 19, "completion_tokens": 7, "total_tokens": 26}, tokens

    parsed = _parse_verbose_models(
        'opencode/mimo-v2.5-free\n{\n  "id": "mimo-v2.5-free",\n  "providerID": "opencode",\n'
        '  "cost": {"input": 0, "output": 0}\n}\n'
        'paid/some-model\n{\n  "id": "some-model",\n  "cost": {"input": 3, "output": 15}\n}\n')
    assert [m for m, _ in parsed] == ["opencode/mimo-v2.5-free", "paid/some-model"], parsed
    assert _is_free(parsed[0][1]) and not _is_free(parsed[1][1])

    # v2.1.4 path: plain `opencode models` — one bare id per line, no cost metadata.
    # Usage chrome, blanks and JSON must not be mistaken for model ids.
    plain = _parse_plain_models(
        "opencode/mimo-v2.6-flash-free\nopencode/big-pickle\n\n"
        "USAGE\n  opencode models [flags]\n  --verbose\n")
    assert [m for m, _ in plain] == ["opencode/mimo-v2.6-flash-free", "opencode/big-pickle"], plain
    # Cost metadata wins when present; without it the name heuristic applies, and a
    # missing `cost` object must never be read as "free" (that would admit paid models).
    assert _admit_as_free("opencode/mimo-v2.6-flash-free", {})
    assert _admit_as_free("opencode/big-pickle", {})
    assert not _admit_as_free("openai/gpt-5", {})
    assert not _admit_as_free("opencode/some-pro", {"cost": {"input": 1, "output": 2}})
    assert _admit_as_free("opencode/some-pro", {"cost": {"input": 0, "output": 0}})

    cfg = BridgeConfig(build_parser().parse_args(
        ["--models", "opencode/mimo-v2.6-flash-free,opencode/big-pickle"]))
    # The default is always force-prepended (and de-duplicated) ahead of --models.
    assert cfg.models == ["opencode/mimo-v2.6-flash-free", "opencode/big-pickle"], cfg.models
    assert cfg.resolve_model("big-pickle")[0] == "opencode/big-pickle"        # bare-id alias
    assert cfg.resolve_model("OpenCode/Big-Pickle")[0] == "opencode/big-pickle"  # case-insensitive
    assert cfg.resolve_model("")[0] == cfg.default_model                      # empty -> default
    resolved, err = cfg.resolve_model("gpt-4o")                              # paid/unknown -> refused
    assert resolved is None and err and "not available" in err, (resolved, err)
    assert BridgeConfig(build_parser().parse_args(
        ["--models", "opencode/big-pickle", "--no-free-only"])).resolve_model("openai/gpt-4o")[0] == "openai/gpt-4o"
    assert BridgeConfig(build_parser().parse_args(
        ["--no-pass-model"])).resolve_model("anything")[0] == DEFAULT_MODEL

    cmd = build_command(BridgeConfig(build_parser().parse_args(
        ["--show-reasoning", "--agent", "build", "--variant", "high", "--cwd", os.getcwd()])), "opencode/x-free")
    for expected in ("run", "--format", "json", "--model", "opencode/x-free",
                     "--dangerously-skip-permissions",
                     "--thinking", "--agent", "build", "--variant", "high", "--dir"):
        assert expected in cmd, (expected, cmd)
    assert "-p" not in cmd and "--prompt" not in cmd, cmd  # prompt goes over stdin
    # --no-auto-approve must drop the flag entirely (either spelling).
    no_auto = build_command(BridgeConfig(build_parser().parse_args(["--no-auto-approve"])), "opencode/x-free")
    assert "--dangerously-skip-permissions" not in no_auto and "--auto" not in no_auto, no_auto

    # v2.1.4 advertises neither --dir nor --variant: emitting them would make yargs
    # reject the whole command, so the probed flag set must suppress them.
    legacy = BridgeConfig(build_parser().parse_args(
        ["--show-reasoning", "--agent", "build", "--variant", "high", "--cwd", os.getcwd()]))
    legacy.run_flags = {"--model", "--agent", "--format", "--file", "--thinking", "--auto"}
    legacy_cmd = build_command(legacy, "opencode/x-free")
    assert "--dir" not in legacy_cmd and "--variant" not in legacy_cmd, legacy_cmd
    assert "--thinking" in legacy_cmd and "--agent" in legacy_cmd, legacy_cmd  # supported flags kept
    # A failed probe (empty set) must stay permissive rather than drop everything.
    unprobed = BridgeConfig(build_parser().parse_args(["--variant", "high"]))
    unprobed.run_flags = set()
    assert unprobed.supports("--dir") and unprobed.supports("--variant")

    # Model cache round-trip: what discovery produces must survive a write/read cycle
    # with context lengths and the free flag intact, since the deployment builds the
    # dashboard's model picker from this file.
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "sub", "models.json")
        discovered = ["opencode/new-shiny-free", "opencode/mimo-v2.5-free"]
        discovered_meta = {
            "opencode/new-shiny-free": {"name": "New Shiny Free", "providerID": "opencode",
                                        "limit": {"context": 512000}, "cost": {"input": 0, "output": 0}},
            "opencode/mimo-v2.5-free": {"name": "MiMo V2.5 Free", "providerID": "opencode",
                                        "limit": {"context": 200000}, "cost": {"input": 0, "output": 0}},
        }
        write_model_cache(cache_path, discovered, discovered_meta, True)
        ids, meta = read_model_cache(cache_path)
        assert ids == discovered, ids
        assert meta["opencode/new-shiny-free"]["limit"]["context"] == 512000, meta
        assert _is_free(meta["opencode/new-shiny-free"]), meta
        assert read_model_cache(os.path.join(tmpdir, "absent.json")) == ([], {})
        with open(os.path.join(tmpdir, "junk.json"), "w", encoding="utf-8") as handle:
            handle.write("{not json")
        assert read_model_cache(os.path.join(tmpdir, "junk.json")) == ([], {})  # corrupt -> ignored
        assert read_model_cache("") == ([], {})
        write_model_cache("", discovered, discovered_meta, True)  # no path -> no-op, no raise

        # A retired default model must not stay advertised: the catalogue wins.
        stale = BridgeConfig(build_parser().parse_args(
            ["--model", "opencode/retired-free", "--model-cache", cache_path,
             "--opencode-bin", "/nonexistent-so-discovery-fails"]))
        stale.refresh_catalogue(refresh=False)
        assert stale.discovered_from == "cache", stale.discovered_from
        assert stale.default_model == "opencode/new-shiny-free", stale.default_model
        assert stale.models == discovered, stale.models          # cache order preserved
        assert stale.resolve_model("opencode/new-shiny-free")[0] == "opencode/new-shiny-free"

        # No cache and no CLI -> the seed list, and the default still resolves.
        seeded = BridgeConfig(build_parser().parse_args(
            ["--opencode-bin", "/nonexistent-so-discovery-fails"]))
        seeded.refresh_catalogue(refresh=False)
        assert seeded.discovered_from == "fallback", seeded.discovered_from
        assert seeded.models == FALLBACK_FREE_MODELS[:1] + [m for m in FALLBACK_FREE_MODELS[1:]], seeded.models

    assert classify_error("429 Too Many Requests") == 429
    assert classify_error("model not found") == 502
    assert classify_error("Free usage limit reached") == 429  # added marker
    # Upstream `status` wins over keyword heuristics, but only for codes we map;
    # a 404 from upstream must not be forwarded as an upstream-failure code.
    assert _status_or_classify({"status": 429}, "whatever") == 429
    assert _status_or_classify({"status": 503}, "whatever") == 503
    assert _status_or_classify({"status": 404}, "model not found") == 502
    assert _status_or_classify({}, "boom") == 502
    # v2.1.4 nests retryAfterMs; the hint must render and never raise on junk.
    assert _with_retry_hint("Free usage limit reached",
                            {"retryAfterMs": 61502000}) == "Free usage limit reached (retry in 17h)"
    assert _with_retry_hint("x", {"retryAfterMs": 90000}) == "x (retry in 1m)"
    assert _with_retry_hint("x", {"retryAfterMs": 5000}) == "x (retry in 5s)"
    assert _with_retry_hint("x", {"retryAfterMs": "nonsense"}) == "x"
    assert _with_retry_hint("x", {}) == "x"
    assert _finish_reason("tool-calls") == "stop" and _finish_reason("length") == "length"
    assert "sk-[redacted]" in _safe_error_text("bad key sk-abc123")
    assert _tool_note({"tool": "read", "state": {"status": "completed",
                                                 "input": {"filePath": "/tmp/a.txt"}}}) == "\n› read(/tmp/a.txt)\n"
    assert _tool_note({"tool": "read", "state": {"status": "running"}}) is None
    # A failed skill load must name the skill, not render as a bare `skill()`.
    assert _tool_note({"tool": "skill", "state": {"status": "error",
                                                  "input": {"name": "lead-devops-sre"}}}) == "\n✗ skill(lead-devops-sre)\n"

    # ── vision path (the reason most users deploy this bridge) ──────────────
    raw_png = b"\x89PNG\r\n\x1a\n12345678"
    png_b64 = base64.b64encode(raw_png).decode()
    # Line-wrapped base64 (some clients fold it) must still decode.
    folded = "\n".join(png_b64[i:i + 16] for i in range(0, len(png_b64), 16))
    data, mime = _decode_image_url(f"data:image/png;base64,{folded}", 0)
    assert data == raw_png and mime == "image/png", (data, mime)
    # An image-only turn must yield the attachment placeholder, not "" (which
    # would 400 the request despite perfectly valid attachments).
    image_only = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}}]}]
    assert build_prompt(image_only) == "(see attached image)", build_prompt(image_only)
    mixed = [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}}]}]
    assert build_prompt(mixed) == "USER:\nwhat is this?", build_prompt(mixed)  # text wins, image rides via --file=
    staged = stage_image_files([(raw_png, None)])  # mime-less: magic-byte sniffing picks .png
    try:
        assert staged[0].endswith(".png"), staged
    finally:
        cleanup_image_files(staged)
    assert not os.path.exists(os.path.dirname(staged[0])), "staging dir must be cleaned up"
    # Over-limit image COUNT is refused before any payload is decoded/fetched.
    try:
        extract_images(image_only * 5, DEFAULT_MAX_IMAGES, DEFAULT_MAX_IMAGE_BYTES)
        raise SystemExit("selfcheck: over-limit image count was accepted")
    except ImageExtractionError as exc:
        assert "too many images" in str(exc), exc
    assert extract_images([], 0, DEFAULT_MAX_IMAGE_BYTES) == []          # disabled + none present
    try:
        extract_images(image_only, 0, DEFAULT_MAX_IMAGE_BYTES)           # disabled + image -> clear error
        raise SystemExit("selfcheck: images accepted while support is disabled")
    except ImageExtractionError as exc:
        assert "--max-images 0" in str(exc), exc

    # ── shim resolution + process-tree reaping (the orphan-slot leak fix) ───
    # opencode.CMD spawns opencode.exe as a grandchild; killing only the shim
    # left the grandchild holding the stdio pipes, wedging run_events forever
    # and leaking the concurrency slot (observed as `in_flight` stuck at 2 and
    # every later request answered 429 "bridge busy").
    with tempfile.TemporaryDirectory() as td:
        exe = os.path.join(td, "node_modules", "@opencode", "cli", "bin", "opencode.exe")
        os.makedirs(os.path.dirname(exe))
        with open(exe, "w", encoding="utf-8"):
            pass
        shim = os.path.join(td, "opencode.CMD")
        with open(shim, "w", encoding="utf-8"):
            pass
        assert resolve_real_binary(shim) == exe        # npm layout -> real binary
        lonely = os.path.join(td, "other.CMD")
        with open(lonely, "w", encoding="utf-8"):
            pass
        assert resolve_real_binary(lonely) == lonely   # different tool's shim -> unchanged
        # opencode-named shim without the npm layout beside it -> unchanged.
        bare = os.path.join(td, "empty", "opencode.CMD")
        os.makedirs(os.path.dirname(bare), exist_ok=True)
        with open(bare, "w", encoding="utf-8"):
            pass
        assert resolve_real_binary(bare) == bare
        assert resolve_real_binary(exe) == exe         # already a real binary
        assert resolve_real_binary("") == ""

    if _IS_WINDOWS:
        def _pid_alive(pid: int) -> bool:
            from ctypes import wintypes
            k32 = _kernel32()
            handle = k32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
            if handle:
                k32.CloseHandle(handle)
                return True
            return False

        def _wait_until(check, timeout: float = 8.0) -> bool:
            end = time.time() + timeout
            while time.time() < end:
                if check():
                    return True
                time.sleep(0.2)
            return check()

        # 1) _kill must reap the GRANDCHILD, not just the .cmd wrapper.
        with tempfile.TemporaryDirectory() as td:
            child_py = os.path.join(td, "child.py")
            pid_file = os.path.join(td, "pid.txt")
            with open(child_py, "w", encoding="utf-8") as handle:
                handle.write("import os, sys, time\n"
                             f"open({pid_file!r}, 'w').write(str(os.getpid()))\n"
                             "time.sleep(60)\n")
            shim = os.path.join(td, "tree.CMD")
            with open(shim, "w", encoding="utf-8") as handle:
                handle.write(f'@echo off\r\n"{sys.executable}" "{child_py}"\r\n')
            proc = subprocess.Popen(
                [shim], cwd=td, text=True,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)  # type: ignore[attr-defined]
            try:
                assert _wait_until(lambda: os.path.exists(pid_file)), "grandchild never started"
                with open(pid_file, encoding="utf-8") as handle:
                    grandchild_pid = int(handle.read().strip())
                _kill(proc)
                assert proc.poll() is not None, "_kill must terminate the direct child"
                assert _wait_until(lambda: not _pid_alive(grandchild_pid)), \
                    "_kill leaked the grandchild (the in-flight slot leak)"
            finally:
                if proc.poll() is None:
                    _kill(proc)
                proc.stdin and proc.stdin.close()
                proc.stdout and proc.stdout.close()
                proc.stderr and proc.stderr.close()

        # 2) Job object: closing it (what happens when the bridge dies) must
        #    kill the child - covers force-killed/crashed bridge processes.
        k32 = _kernel32()
        keeper = len(_JOB_HANDLES)
        job_child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            assert _assign_kill_on_close_job(job_child), "job assignment failed"
            assert len(_JOB_HANDLES) == keeper + 1
            k32.CloseHandle(_JOB_HANDLES.pop())          # simulate bridge death
            assert _wait_until(lambda: job_child.poll() is not None), \
                "kill-on-close job did not kill the child"
        finally:
            if job_child.poll() is None:
                job_child.kill()
                job_child.wait(timeout=5)

    print("selfcheck ok")


if __name__ == "__main__":
    if os.getenv("BRIDGE_SELFCHECK"):
        _selfcheck()
    else:
        main()
