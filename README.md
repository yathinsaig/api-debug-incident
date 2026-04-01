---
title: API Incident Debugger
emoji: 🔧
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# API Integration Incident Debugger — OpenEnv Environment

A real-world OpenEnv-compliant benchmark where an AI agent debugs broken API
integrations step-by-step. The agent reads error logs, makes test calls to a
live mock server, and patches the request configuration to resolve faults —
the exact workflow a senior developer follows at 2am when production is down.

---

## Problem Description

Third-party API integrations break constantly — expired tokens, deprecated
endpoints, malformed payloads, wrong base URLs. Unlike code-generation
benchmarks (SWE-bench) or API-selection benchmarks (ToolBench), this
environment models the **diagnostic loop**: the agent must form hypotheses,
make targeted test calls, and converge on root cause efficiently.

**What makes this novel:**
- The mock server returns realistic structured HTTP errors (401, 422, 404...)
- The agent is penalised for wasted test calls — efficiency is rewarded
- Hard mode injects ~40% irrelevant noise into logs, forcing signal/noise discrimination
- The grader measures *how* the agent debugs, not just *whether* it succeeds

---

## Architecture

```
Agent (inference.py / baseline.py)
      |
      v  Action (JSON)
  env.py  ───────────────────────────► mock_server.py (FastAPI :8765)
      |   ◄──────────────────────────── HTTP response (real network call)
      |
      v  Observation (Pydantic)
  Agent receives: request state, response, logs, budget
      |
      v
  grader.py  →  score 0.0–1.0
```

---

## Project Structure

```
api-incident-debugger/
├── server/
│   ├── __init__.py
│   └── app.py             Web API server (reset / step / state endpoints)
├── models.py              Pydantic schemas (Action, Observation, APIRequest, ...)
├── env.py                 Core OpenEnv environment (reset / step / state)
├── scenarios.py           6 scenario definitions (easy / medium / hard)
├── mock_server.py         FastAPI mock server with injectable faults
├── grader.py              Programmatic grader — score 0.0–1.0
├── inference.py           Inference script (Groq LLM agent loop)
├── baseline.py            Baseline agent (Groq llama3-8b)
├── app.py                 Root-level FastAPI app for HF Spaces
├── openenv.yaml           OpenEnv metadata
├── pyproject.toml         Project config with dependencies and scripts
├── uv.lock                Locked dependencies
├── Dockerfile             Docker deployment (mock server + web API)
├── requirements.txt
└── README.md
```

---

## Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.1-8b-instant
HF_TOKEN=your_hf_token
```

Get a free Groq key at https://console.groq.com

---

## API Endpoints

The server exposes the following OpenEnv-compliant endpoints:

| Method | Endpoint      | Description                              |
|--------|---------------|------------------------------------------|
| GET    | `/`           | Health check — returns 200               |
| POST   | `/reset`      | Start a new episode (optional `scenario_id`) |
| POST   | `/step`       | Execute an action (`action_type` + `parameters`) |
| GET    | `/state`      | Return full environment state            |
| GET    | `/scenarios`  | List all available scenarios             |

---

## Observation Space

Each step the agent receives an `Observation` with these fields:

| Field             | Type           | Description                                          |
|-------------------|----------------|------------------------------------------------------|
| `request`         | APIRequest     | Current request: endpoint, method, headers, body, base_url |
| `response`        | APIResponse    | Last HTTP response: status_code, body, error_message, latency_ms |
| `logs`            | list[str]      | Last 20 log lines (hard mode includes noise)        |
| `step_count`      | int            | Steps taken so far this episode                     |
| `action_history`  | list[str]      | Ordered list of action_type strings taken           |
| `budget_remaining`| int            | Steps left before forced termination                |

---

## Action Space

| Action           | Parameters                                         | Effect                                           |
|------------------|----------------------------------------------------|--------------------------------------------------|
| `inspect_logs`   | none                                               | Surfaces log lines in next observation           |
| `analyze_response` | none                                             | Highlights status code and error message         |
| `make_test_call` | endpoint, method, headers, body, base_url         | Sends a live HTTP call to mock server            |
| `patch_config`   | key (str), value (str)                            | Changes a header key or base_url                 |
| `patch_request`  | field (str), value (any)                          | Modifies endpoint, method, or body field         |
| `submit_fix`     | none                                               | Final validation — triggers grader               |

---

## Tasks (6 Scenarios)

### Scenario 0 — Easy: Missing auth header
- **Fault:** `Authorization` header completely absent
- **Fix:** Add `Authorization: Bearer <valid-token>`
- **Budget:** 10 steps

### Scenario 1 — Medium: Malformed payload
- **Faults:** `user_id` sent as string instead of int + `email` field missing
- **Fix:** Cast `user_id` to integer + add `email` field
- **Budget:** 15 steps

### Scenario 2 — Hard: Multi-fault + noisy logs
- **Faults:** Expired token + deprecated `/v1/users` → `/v2/users` + missing `role` field
- **Extra:** ~40% noise in logs
- **Budget:** 20 steps

### Scenario 3 — Easy: Rate limited
- **Fault:** 429 Too Many Requests
- **Budget:** 8 steps

### Scenario 4 — Medium: Wrong base URL + wrong auth scheme
- **Faults:** Wrong port (9999 → 8765) + Basic auth instead of Bearer
- **Budget:** 12 steps

### Scenario 5 — Hard: Kitchen sink
- **Faults:** Expired token + deprecated endpoint + wrong field type + missing email + missing role
- **Budget:** 25 steps

---

## Reward Function

| Event                                    | Reward        |
|------------------------------------------|---------------|
| Every step taken                         | -0.05         |
| `inspect_logs` (first 2 times)           | +0.10         |
| `analyze_response` (first 2 times)       | +0.10         |
| `make_test_call` (base cost)             | -0.05         |
| `make_test_call` discovers new error type | +0.15        |
| Each fault resolved via patch            | +0.30         |
| `submit_fix` succeeds (status 200)       | +1.50         |
| Efficiency bonus on success              | +0.50 x (1 - steps/budget) |
| Premature `submit_fix` (wrong)           | -0.20         |
| Repeated identical action                | -0.10         |

**Reward range:** -5.0 to +3.0

---

## Grader

Scores are computed by `APIDebugGrader` in `grader.py`:

| Component         | Weight | Criterion                                              |
|-------------------|--------|--------------------------------------------------------|
| Success           | 0.50   | Did the final request return 200?                     |
| Efficiency        | 0.25   | `1 - steps_used / max_steps` (higher = fewer steps)  |
| Reasoning         | 0.25   | Fraction of pre-patch steps that were diagnostic      |

**Final score = weighted sum, clipped to [0.0, 1.0]**

---

## Setup & Usage

### Local

```bash
git clone https://github.com/yathinsaig/api-debug-incident.git
cd api-debug-incident
pip install -r requirements.txt

# Run inference (all scenarios)
python inference.py

# Run specific scenario
python inference.py --scenario 0

# Start the web API server
uvicorn mock_server:app --host 0.0.0.0 --port 8765 &
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Swagger docs at http://127.0.0.1:7860/docs
```

### Docker

```bash
docker build -t api-incident-debugger .
docker run -e GROQ_API_KEY=your_key api-incident-debugger
```

### HF Space

Deployed at: https://huggingface.co/spaces/yathingrandhi2003/api-incident-debugger

---

## Baseline Results (llama3-8b-instant via Groq)

| Scenario                    | Difficulty | Steps | Success | Efficiency | Reasoning | Score  |
|-----------------------------|------------|-------|---------|------------|-----------|--------|
| Missing auth header         | easy       | 4     | YES     | 0.600      | 1.000     | 0.7750 |
| Malformed payload           | medium     | 9     | YES     | 0.400      | 0.667     | 0.6167 |
| Multi-fault + noisy logs    | hard       | 16    | YES     | 0.200      | 0.500     | 0.3875 |
| **Average**                 |            |       |         |            |           | **0.5930** |

---

## License

MIT
