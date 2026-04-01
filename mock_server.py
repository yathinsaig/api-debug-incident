"""
mock_server.py — FastAPI mock server simulating a generic REST API.

Faults are injected at runtime via POST /inject-fault.
The server validates each incoming request against the active fault set
and returns realistic HTTP errors.

Run standalone:
    uvicorn mock_server:app --host 0.0.0.0 --port 8765

Or let env.py start it as a subprocess.
"""

from __future__ import annotations

import random
import time
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="API Debug Mock Server", version="1.0.0")

# ---------------------------------------------------------------------------
# Global fault registry
# ---------------------------------------------------------------------------

# Set of currently active fault type strings
_active_faults: set[str] = set()

# Seed for reproducible noisy logs (set per scenario)
_noise_seed: int = 42


# ---------------------------------------------------------------------------
# Control endpoints
# ---------------------------------------------------------------------------

class FaultRequest(BaseModel):
    fault_type: str


class NoiseSeedRequest(BaseModel):
    seed: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/inject-fault")
def inject_fault(req: FaultRequest) -> dict[str, Any]:
    _active_faults.add(req.fault_type)
    return {"active_faults": list(_active_faults)}


@app.post("/clear-faults")
def clear_faults() -> dict[str, Any]:
    _active_faults.clear()
    return {"active_faults": []}


@app.post("/set-noise-seed")
def set_noise_seed(req: NoiseSeedRequest) -> dict[str, str]:
    global _noise_seed
    _noise_seed = req.seed
    return {"seed": str(_noise_seed)}


@app.get("/active-faults")
def get_active_faults() -> dict[str, Any]:
    return {"active_faults": list(_active_faults)}


# ---------------------------------------------------------------------------
# Noisy log generator (hard scenario)
# ---------------------------------------------------------------------------

_NOISE_LINES = [
    "WARN  [db-pool] Connection pool pressure: 78% utilised",
    "INFO  [metrics] p99 latency spike: 1843ms (threshold 1000ms)",
    "ERROR [cache] Redis TIMEOUT after 5000ms — falling back to DB",
    "WARN  [health] Downstream service /analytics responding slowly",
    "INFO  [gc] GC pause 312ms — heap 71% after collection",
    "ERROR [worker-3] Task queue depth exceeded soft limit (1024)",
    "WARN  [cdn] Edge node eu-west-1 reporting elevated error rate",
    "INFO  [deploy] Canary release at 5% traffic — monitoring...",
    "ERROR [smtp] Failed to deliver notification email: timeout",
    "WARN  [rate-limiter] Burst limit approaching for tenant-group-7",
]


def _generate_logs(
    real_lines: list[str],
    noisy: bool,
    seed: int,
) -> list[str]:
    """Interleave real error lines with noise in hard mode."""
    if not noisy:
        return real_lines[-20:]

    rng = random.Random(seed)
    combined: list[str] = []
    for line in real_lines:
        combined.append(line)
        # ~40% chance of injecting a noise line after each real line
        if rng.random() < 0.4:
            combined.append(rng.choice(_NOISE_LINES))
    return combined[-20:]


# ---------------------------------------------------------------------------
# Core API endpoints — v1 and v2
# ---------------------------------------------------------------------------

def _check_auth(headers: dict[str, str]) -> JSONResponse | None:
    """Return error response if auth faults are active, else None."""
    auth = headers.get("authorization", headers.get("Authorization", ""))

    if "MISSING_AUTH_HEADER" in _active_faults:
        if not auth:
            return JSONResponse(
                status_code=401,
                content={"error": "Authorization header required"},
            )

    if "WRONG_AUTH_SCHEME" in _active_faults:
        if auth and not auth.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "Bearer scheme required. Use: Authorization: Bearer <token>"},
            )

    if "EXPIRED_TOKEN" in _active_faults:
        if "expired" in auth.lower():
            return JSONResponse(
                status_code=401,
                content={"error": "Token expired. Please refresh your access token."},
            )

    if "WRONG_SCOPE" in _active_faults:
        return JSONResponse(
            status_code=403,
            content={"error": "Insufficient scope. Required: users:write"},
        )

    return None


def _check_payload(body: dict[str, Any] | None) -> JSONResponse | None:
    """Return error response if payload faults are active, else None."""
    if body is None:
        body = {}

    if "INVALID_PAYLOAD_FIELD" in _active_faults:
        # Check for required fields depending on active scenario context
        # Hard scenario requires 'role'; medium/easy require 'email'
        if "role" not in body and "DEPRECATED_ENDPOINT" in _active_faults:
            return JSONResponse(
                status_code=422,
                content={"error": "Field 'role' is required for v2 user creation"},
            )
        if "email" not in body:
            return JSONResponse(
                status_code=422,
                content={"error": "Field 'email' is required"},
            )

    if "WRONG_FIELD_TYPE" in _active_faults:
        user_id = body.get("user_id")
        if user_id is not None and not isinstance(user_id, int):
            return JSONResponse(
                status_code=422,
                content={
                    "error": f"Field 'user_id' must be integer, got {type(user_id).__name__}"
                },
            )

    return None


@app.post("/v1/users")
async def v1_create_user(request: Request) -> JSONResponse:
    t0 = time.time()
    headers = dict(request.headers)

    # Check if endpoint is deprecated
    if "DEPRECATED_ENDPOINT" in _active_faults:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Endpoint /v1/users is deprecated. Migrate to /v2/users",
                "migration_guide": "https://docs.example.com/v2-migration",
            },
        )

    # Auth checks
    auth_err = _check_auth(headers)
    if auth_err:
        return auth_err

    # Parse body
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Payload checks
    payload_err = _check_payload(body)
    if payload_err:
        return payload_err

    latency = (time.time() - t0) * 1000
    return JSONResponse(
        status_code=200,
        content={
            "status": "created",
            "user_id": body.get("user_id", 1),
            "message": "User created successfully via v1 API",
            "latency_ms": round(latency, 2),
        },
    )


@app.post("/v2/users")
async def v2_create_user(request: Request) -> JSONResponse:
    t0 = time.time()
    headers = dict(request.headers)

    # Auth checks
    auth_err = _check_auth(headers)
    if auth_err:
        return auth_err

    # Parse body
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Payload checks
    payload_err = _check_payload(body)
    if payload_err:
        return payload_err

    latency = (time.time() - t0) * 1000
    return JSONResponse(
        status_code=200,
        content={
            "status": "created",
            "user_id": body.get("user_id", 1),
            "message": "User created successfully via v2 API",
            "latency_ms": round(latency, 2),
        },
    )


@app.get("/v1/users")
async def v1_list_users(request: Request) -> JSONResponse:
    headers = dict(request.headers)
    auth_err = _check_auth(headers)
    if auth_err:
        return auth_err

    if "RATE_LIMITED" in _active_faults:
        return JSONResponse(
            status_code=429,
            content={"error": "Too many requests. Retry after 60 seconds."},
            headers={"Retry-After": "60"},
        )

    return JSONResponse(status_code=200, content={"users": [], "total": 0})
