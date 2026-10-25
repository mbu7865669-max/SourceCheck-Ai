# OpenCode Bridge

Puts the **OpenCode CLI** behind an OpenAI-compatible HTTP endpoint so SourceCheck AI can
reason with OpenCode's **free** models (opencode zen) at **$0 inference cost**.

```
SourceCheck agent ──HTTP (OpenAI chat-completions)──▶ bridge :18383 ──▶ `opencode run --format json` ──▶ free models
```

Adapted from [`ch-muhammad-asim/OpenDevOps/opencode`](https://github.com/ch-muhammad-asim/OpenDevOps/tree/main/opencode)
— the upstream script targeted a newer CLI than the v2.1.4 installed here, so it carries a
compatibility patch (see the `PATCH NOTES` block at the top of `opencode_bridge.py`).

---

## Quick start

```powershell
# start (foreground, Ctrl+C to stop)
.\bridge\run-bridge.ps1

# or detached
.\bridge\run-bridge.ps1 background

# health check + a live completion
.\bridge\run-bridge.ps1 test

# what free models are available
.\bridge\run-bridge.ps1 models

# stop / status / offline logic checks
.\bridge\run-bridge.ps1 stop
.\bridge\run-bridge.ps1 status
.\bridge\run-bridge.ps1 selfcheck
```

No arguments works too: `python bridge\opencode_bridge.py`

> Requires the `opencode` CLI on `PATH` (`opencode --version` → `v2.1.4`). It inherits
> your existing OpenCode login — the bridge itself needs no credentials.

---

## Hook up the agent

```powershell
$env:MODEL_PROVIDER = "opencode"
python main.py "What year did the first human walk on the Moon?"
```

Or put it in `.env` (copy from `.env.example`):

```ini
MODEL_PROVIDER=opencode
BRIDGE_ENDPOINT=http://127.0.0.1:18383/v1
BRIDGE_MODEL=opencode/mimo-v2.6-flash-free
BRIDGE_TIMEOUT_SECONDS=120.0
```

The wiring lives in `app/model.py` (`OpenCodeBridgeLLM`) and is selected by
`get_model()` when `MODEL_PROVIDER=opencode`.

---

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | Blocking + `"stream": true` (SSE) |
| `GET /v1/models` | Advertised free-model catalogue |
| `GET /health` | Liveness + OpenCode version + in-flight count |
| `GET /config` | Effective configuration |
| `GET /metrics` | Requests, errors, cost, uptime |

```bash
curl http://127.0.0.1:18383/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{"model":"opencode/mimo-v2.6-flash-free","messages":[{"role":"user","content":"hello"}]}'
```

Binds **loopback only**. Set `BRIDGE_HOST=0.0.0.0` to expose it — only sensible behind
something you trust, since it has no auth by default.

---

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `BRIDGE_PORT` | `18383` | Listen port |
| `BRIDGE_HOST` | `127.0.0.1` | Bind address |
| `OPENCODE_BRIDGE_MODEL` | `opencode/mimo-v2.6-flash-free` | Default model |
| `OPENCODE_BRIDGE_MODELS` | *(auto-discovered)* | Comma-separated catalogue override |
| `OPENCODE_BRIDGE_API_KEY` | *(empty)* | If set, clients must send `Authorization: Bearer <key>` |
| `OPENCODE_BRIDGE_TIMEOUT` | `300` | Per-request timeout (seconds) |
| `OPENCODE_BRIDGE_MAX_CONCURRENCY` | `2` | Concurrent `opencode` subprocesses |
| `OPENCODE_BRIDGE_FREE_ONLY` | `1` | `0` also allows paid models |
| `OPENCODE_BIN` | *(auto)* | Explicit path to the `opencode` binary |

Every flag has a CLI equivalent — `python bridge\opencode_bridge.py --help`.

---

## Free-model guarantee

The catalogue is discovered from `opencode models` and filtered so a typo can never quietly
start billing a paid provider. On v2.1.4 the CLI reports no cost metadata, so admission uses
a conservative name heuristic (`-free` suffix, plus a small allowlist) — see `_admit_as_free`.
Set `OPENCODE_BRIDGE_FREE_ONLY=0` to lift it.

---

## v2.1.4 compatibility patch

| Upstream assumed | v2.1.4 reality | Fix |
| --- | --- | --- |
| `opencode models <provider> --verbose` | no positional arg, no `--verbose` | falls back to plain `opencode models` id listing |
| cost metadata present | not exposed | `_admit_as_free()` name heuristic |
| `run --dir` / `--variant` / `--pure` | flags don't exist; yargs rejects the *whole* command | gated on a live `run --help` probe |
| `--dangerously-skip-permissions` | spelled `--auto` | probed (upstream behaviour, kept) |
| default `opencode/mimo-v2.5-free` | retired from the free tier | now `opencode/mimo-v2.6-flash-free` |
| UTF-8 console | cp1252 crashes the banner | ASCII banner + `errors="replace"` |
| `error.message` nested under `data` | v2.1.4 puts it on the error object | reads both → real messages surface |

That last row is why an exhausted quota used to appear as the useless `opencode error`;
it now reads `Free usage limit reached ... (retry in 17h)` and returns **429**, not 502.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `OpenCode bridge is not reachable` | `.\bridge\run-bridge.ps1 start` |
| `model ... is not available on this bridge` | `.\bridge\run-bridge.ps1 models` for valid ids |
| `Free usage limit reached (retry in Nh)` | OpenCode free-tier quota exhausted — wait, or set `OPENCODE_BRIDGE_FREE_ONLY=0` with your own provider key |
| `429 bridge busy` | raise `OPENCODE_BRIDGE_MAX_CONCURRENCY` (gently — free tiers rate-limit) |
| `504 opencode command timed out` | raise `OPENCODE_BRIDGE_TIMEOUT` / `BRIDGE_TIMEOUT_SECONDS` |
| Catalogue looks stale | `POST /v1/models/refresh`, or restart the bridge |
