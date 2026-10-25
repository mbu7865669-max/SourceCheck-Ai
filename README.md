# SourceCheck AI

**Autonomous Web Information Retrieval Agent**  
*Built for the AMD AI League — Match 4: Intelligent Web Information Retrieval*

---

## 1. Project Overview

**SourceCheck AI** is an autonomous web information retrieval and question-answering agent. Given a natural-language question, it queries the live web, extracts and verifies primary source evidence, and returns a concise, grounded factual answer alongside the exact supporting citation. If reliable evidence cannot be found or verified, it explicitly abstains rather than generating an unsupported answer.

### Key Goals
- **Factual Precision:** Target exact answers (names, dates, numbers, specifications, versions) rather than generic, open-ended conversational summaries.
- **Zero Hallucination & Explicit Abstention:** Strictly verify that the answer is supported by the retrieved source excerpt. If unverified, return `status: "insufficient_evidence"`.
- **Low Latency:** Optimize retrieval and inference to complete under strict evaluation ceilings (sub-30-second target).
- **Open-Source & AMD ROCm Ready:** Powered by open-source models suitable for execution on AMD GPU hardware using the AMD ROCm environment, without dependencies on paid proprietary APIs.

---

## 2. Core Workflow Pipeline

1. **Question Analysis:** Parse the user question and determine the expected answer type (e.g., date, number, person, specification).
2. **Search & Retrieval:** Generate targeted search queries and fetch candidate web pages with bounded retries and timeouts.
3. **HTML Cleaning & Extraction:** Strip boilerplate and extract clean text and structured data tables.
4. **Passage Scoring:** Rank and select top relevant evidence passages to minimize prompt token length.
5. **Model Inference:** Utilize an open-source language model to extract the factual candidate answer and exact supporting excerpt.
6. **Evidence Verification:** Confirm that the answer verbatim exists in the retrieved context; abstain if unsupported.
7. **Structured Output:** Return standardized, machine-readable JSON containing `answer`, `source_url`, `evidence`, and `status`.

---

## 3. Project Structure

```text
SourceCheck-Ai/
├── app/
│   ├── __init__.py           # Application package marker
│   ├── config.py             # Configuration and environment variable management
│   ├── schemas.py            # Pydantic data models for requests, responses, and evidence
│   ├── retrieval.py          # Search provider integration and HTTP fetching/parsing
│   ├── extractor.py          # Relevant passage ranking and text extraction
│   ├── model.py              # Model interface: mock / local ROCm / local server / OpenCode bridge
│   └── agent.py              # Full pipeline orchestration coordinator
├── bridge/
│   ├── opencode_bridge.py    # OpenAI-compatible bridge backed by the OpenCode CLI (stdlib only)
│   ├── run-bridge.ps1        # Windows launcher: start / test / models / stop / status
│   └── README.md             # Bridge setup, configuration, troubleshooting
├── tests/
│   ├── __init__.py           # Test package marker
│   ├── test_agent.py         # Agent orchestration and grounded-reasoning tests
│   ├── test_extractor.py     # Passage splitting and relevance ranking tests
│   ├── test_retrieval.py     # Search, fetching, HTML cleaning, SSRF guard tests
│   ├── test_model.py         # Model factory + OpenCode bridge contract tests
│   └── test_main.py          # HTTP routes and CLI argument tests
├── main.py                   # Application entrypoint (FastAPI server & CLI interface)
├── requirements.txt          # Python dependencies
├── .env.example              # Template for environment configuration
├── .gitignore                # Git exclusions (caches, venvs, weights, secrets)
└── README.md                 # Project documentation and guide
```

---

## 4. Running with OpenCode's Free Models

The agent can reason with OpenCode's **free** models by routing requests through a local
bridge that exposes an OpenAI-compatible endpoint:

```
SourceCheck agent ──HTTP──▶ bridge :18383 ──▶ `opencode run` ──▶ opencode zen (free, $0)
```

```powershell
# 1. start the bridge (keeps running in its own window)
.\bridge\run-bridge.ps1 background

# 2. confirm it is healthy and list the free models
.\bridge\run-bridge.ps1 test
.\bridge\run-bridge.ps1 models

# 3. ask a question through the agent
$env:MODEL_PROVIDER = "opencode"
python main.py "What year did the first human walk on the Moon?"

# ...or serve it over HTTP
python main.py serve
curl -X POST http://127.0.0.1:8000/answer -H "content-type: application/json" `
     -d '{"question":"What year did the first human walk on the Moon?"}'
```

Set `MODEL_PROVIDER=opencode` in `.env` to make it the default. Other providers remain
available: `mock` (offline, the test-suite default), `local_rocm`, `local_server`.
Full bridge documentation: [`bridge/README.md`](bridge/README.md).

---

## 5. Planned Milestones

- [x] **Milestone 1 — Requirements and Architecture:** Skeleton setup, architecture design, and requirement breakdown.
- [x] **Milestone 2 — Basic Retrieval:** Search query execution, HTTP fetching, HTML cleaning, and test cases.
- [x] **Milestone 3 — Model Integration:** Model abstraction (`app/model.py`) with mock, CPU/ROCm, local-server and OpenCode-bridge backends.
- [x] **Milestone 4 — Evidence Verification:** Exact answer extraction, source citation matching, and abstention handling.
- [ ] **Milestone 5 — Evaluation and Optimization:** Benchmark testing against question sets for latency, accuracy, and refusal correctness.
- [ ] **Milestone 6 — Docker and Submission:** Containerization, deployment documentation, and submission deliverables.
