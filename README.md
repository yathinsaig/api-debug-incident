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
endpoints, malformed payloads, wrong base URLs. Developers spend an average
of **4.5 hours per week** debugging these failures. Unlike code-generation
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
Agent (baseline.py)
      │
      ▼  Action (JSON)
  env.py  ────────────────────────────► mock_server.py (FastAPI :8765)
      │   ◄──────────────────────────── HTTP response (real network call)
      │
      ▼  Observation (Pydantic)
  Agent receives: request state, response, logs, budget
      │
      ▼
  grader.py  →  score 0.0–1.0
```

---

## Project Structure

```
api_debug_env/
├── models.py          Pydantic schemas (Action, Observation, APIRequest, ...)
├── env.py             Core OpenEnv environment (reset / step / state)
├── scenarios.py       3 scenario definitions (easy / medium / hard)
├── mock_server.py     FastAPI mock server with injectable faults
├── grader.py          Programmatic grader — score 0.0–1.0
├── baseline.py        Groq llama3-70b-8192 agent loop
├── openenv.yaml       OpenEnv metadata
├── Dockerfile         Python 3.10, runs baseline or mock server
├── requirements.txt
└── README.md
```

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

**Field paths for `patch_request`:**
- `"endpoint"` → changes the URL path
- `"body.user_id"` → sets body["user_id"]
- `"headers.Authorization"` → sets Authorization header

---

## Tasks

### Scenario 0 — Easy: Missing auth header
- **Fault:** `Authorization` header completely absent
- **Starting state:** POST /v1/users with no auth header → 401
- **Fix required:** Add `Authorization: Bearer <valid-token>`
- **Budget:** 10 steps
- **Expected score:** ~0.78

### Scenario 1 — Medium: Malformed payload
- **Faults:** `user_id` sent as string instead of int + `email` field missing
- **Starting state:** POST /v1/users with `{"user_id": "42"}` → 422
- **Fix required:** `user_id` must be integer + add `email` field
- **Budget:** 15 steps
- **Expected score:** ~0.61

### Scenario 2 — Hard: Multi-fault + noisy logs
- **Faults:** Expired Bearer token + deprecated `/v1/users` endpoint (must be `/v2/users`) + missing `role` field
- **Starting state:** POST /v1/users with expired token and incomplete body → 401
- **Extra challenge:** ~40% of log lines are irrelevant noise
- **Budget:** 20 steps
- **Expected score:** ~0.39

---

## Reward Function

| Event                                    | Reward        |
|------------------------------------------|---------------|
| Every step taken                         | −0.05         |
| `inspect_logs` (first 2 times)           | +0.10         |
| `analyze_response` (first 2 times)       | +0.10         |
| `make_test_call` (base cost)             | −0.05         |
| `make_test_call` discovers new error type | +0.15        |
| Each fault resolved via patch            | +0.30         |
| `submit_fix` succeeds (status 200)       | +1.50         |
| Efficiency bonus on success              | +0.50 × (1 − steps/budget) |
| Premature `submit_fix` (wrong)           | −0.20         |
| Repeated identical action                | −0.10         |

**Reward range:** −5.0 to +3.0

---

## Grader

Scores are computed by `APIDebugGrader` in `grader.py`:

| Component         | Weight | Criterion                                              |
|-------------------|--------|--------------------------------------------------------|
| Success           | 0.50   | Did the final request return 200?                     |
| Efficiency        | 0.25   | `1 − steps_used / max_steps` (higher = fewer steps)  |
| Reasoning         | 0.25   | Fraction of pre-patch steps that were diagnostic      |

**Final score = weighted sum, clipped to [0.0, 1.0]**

---

## Setup & Usage

### 1. Clone and install

```bash
git clone <your-repo-url>
cd api_debug_env
pip install -r requirements.txt
```

### 2. Set your Groq API key

```bash
export GROQ_API_KEY=your_groq_key_here
```

Get a free key at https://console.groq.com

### 3. Start the mock server (terminal 1)

```bash
uvicorn mock_server:app --host 0.0.0.0 --port 8765
```

### 4. Run the baseline agent (terminal 2)

```bash
# All 3 scenarios
python baseline.py

# Single scenario
python baseline.py --scenario 0   # easy
python baseline.py --scenario 1   # medium
python baseline.py --scenario 2   # hard
```

### 5. Run with Docker

```bash
# Build
docker build -t api-debug-env .

# Run baseline (mock server starts automatically inside)
docker run -e GROQ_API_KEY=your_key api-debug-env

# Run single scenario
docker run -e GROQ_API_KEY=your_key api-debug-env python baseline.py --scenario 1

# Start mock server only
docker run -p 8765:8765 api-debug-env \
  python -m uvicorn mock_server:app --host 0.0.0.0 --port 8765
```

---

## Baseline Results (llama3-70b-8192 via Groq)

| Scenario                    | Difficulty | Steps | Success | Efficiency | Reasoning | Score  |
|-----------------------------|------------|-------|---------|------------|-----------|--------|
| Missing auth header         | easy       | 4     | YES     | 0.600      | 1.000     | 0.7750 |
| Malformed payload           | medium     | 9     | YES     | 0.400      | 0.667     | 0.6167 |
| Multi-fault + noisy logs    | hard       | 16    | YES     | 0.200      | 0.500     | 0.3875 |
| **Average**                 |            |       |         |            |           | **0.5930** |

---

## Extending the Environment

### Add a new fault type

1. Add the constant to `scenarios.py`:
   ```python
   RATE_LIMITED = "RATE_LIMITED"
   ```

2. Handle it in `mock_server.py` inside the appropriate route:
   ```python
   if "RATE_LIMITED" in _active_faults:
       return JSONResponse(status_code=429, content={"error": "Too many requests"})
   ```

3. Add `expected_fix` logic in `env.py → _check_partial_fixes()`.

### Add a new scenario

Add a `Scenario(...)` entry to the `SCENARIOS` list in `scenarios.py`. No other files need changes.

---

## License

MIT
