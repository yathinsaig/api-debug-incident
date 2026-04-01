"""
scenarios.py — Three scenario definitions for the API Debug environment.

Each scenario specifies:
  - Which faults are injected into the mock server
  - The expected fix that produces a 200 response
  - Step budget and noise settings
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Fault type constants (must match mock_server.py fault registry)
# ---------------------------------------------------------------------------

MISSING_AUTH_HEADER = "MISSING_AUTH_HEADER"
WRONG_AUTH_SCHEME = "WRONG_AUTH_SCHEME"
EXPIRED_TOKEN = "EXPIRED_TOKEN"
WRONG_SCOPE = "WRONG_SCOPE"
INVALID_PAYLOAD_FIELD = "INVALID_PAYLOAD_FIELD"
WRONG_FIELD_TYPE = "WRONG_FIELD_TYPE"
DEPRECATED_ENDPOINT = "DEPRECATED_ENDPOINT"
WRONG_BASE_URL = "WRONG_BASE_URL"
RATE_LIMITED = "RATE_LIMITED"


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    id: int
    name: str
    difficulty: str          # easy | medium | hard
    description: str
    faults_injected: list[str]

    # Initial (broken) request the agent starts with
    initial_request: dict

    # What the agent must achieve for a 200 response
    # Keys map to the field that must be correct
    expected_fix: dict

    max_steps: int
    noisy_logs: bool = False

    # Human-readable hint surfaced in the first observation log
    hint: str = ""


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [

    # ------------------------------------------------------------------
    # Scenario 0 — EASY
    # ------------------------------------------------------------------
    Scenario(
        id=0,
        name="Missing auth header",
        difficulty="easy",
        description=(
            "The integration calls POST /v1/users but completely omits the "
            "Authorization header. The API returns 401. The agent must identify "
            "the missing header and patch it with a valid Bearer token."
        ),
        faults_injected=[MISSING_AUTH_HEADER],
        initial_request={
            "endpoint": "/v1/users",
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "body": {"user_id": 42, "email": "alice@example.com"},
            "base_url": "http://localhost:8765",
        },
        expected_fix={
            "headers.Authorization": "Bearer ",   # must START with "Bearer "
        },
        max_steps=10,
        noisy_logs=False,
        hint="[HINT] The last deploy removed authentication middleware config.",
    ),

    # ------------------------------------------------------------------
    # Scenario 1 — MEDIUM
    # ------------------------------------------------------------------
    # Scenario(
    #     id=1,
    #     name="Malformed payload — missing field + wrong type",
    #     difficulty="medium",
    #     description=(
    #         "POST /v1/users returns 422. Two payload faults: (1) 'user_id' is "
    #         "sent as a string '42' instead of integer 42, and (2) the required "
    #         "'email' field is absent from the request body. Both must be fixed."
    #     ),
    #     faults_injected=[INVALID_PAYLOAD_FIELD, WRONG_FIELD_TYPE],
    #     initial_request={
    #         "endpoint": "/v1/users",
    #         "method": "POST",
    #         "headers": {
    #             "Content-Type": "application/json",
    #             "Authorization": "Bearer valid-token-abc",
    #         },
    #         "body": {"user_id": "42"},   # string instead of int, email missing
    #         "base_url": "http://localhost:8765",
    #     },
    #     expected_fix={
    #         "body.user_id": 42,                        # must be integer
    #         "body.email": "",                          # any non-empty string
    #     },
    #     max_steps=15,
    #     noisy_logs=False,
    #     hint="[HINT] The frontend team recently changed how form data is serialised.",
    # ),

    # ------------------------------------------------------------------
    # Scenario 2 — HARD
    # ------------------------------------------------------------------
    # Scenario(
    #     id=2,
    #     name="Multi-fault: expired token + deprecated endpoint + missing field + noisy logs",
    #     difficulty="hard",
    #     description=(
    #         "Production broke after a v1→v2 API migration. Three faults: "
    #         "(1) the Bearer token is expired, (2) the endpoint is still pointing "
    #         "to /v1/users which is deprecated — must be /v2/users, and (3) the "
    #         "body is missing the required 'role' field. Logs contain ~40% noise "
    #         "(unrelated timeout warnings and irrelevant stack traces)."
    #     ),
    #     faults_injected=[EXPIRED_TOKEN, DEPRECATED_ENDPOINT, INVALID_PAYLOAD_FIELD],
    #     initial_request={
    #         "endpoint": "/v1/users",          # must be changed to /v2/users
    #         "method": "POST",
    #         "headers": {
    #             "Content-Type": "application/json",
    #             "Authorization": "Bearer expired-token-xyz",   # expired
    #         },
    #         "body": {"user_id": 99, "email": "bob@example.com"},   # missing 'role'
    #         "base_url": "http://localhost:8765",
    #     },
    #     expected_fix={
    #         "headers.Authorization": "Bearer ",   # must START with "Bearer " and not contain "expired"
    #         "endpoint": "/v2/users",
    #         "body.role": "",                       # any non-empty string
    #     },
    #     max_steps=20,
    #     noisy_logs=True,
    #     hint="[HINT] The v2 migration doc was published last Tuesday. Check the changelog.",
    # ),
    # ------------------------------------------------------------------
    # Scenario 3 — EASY: Rate limited
    # ------------------------------------------------------------------
    Scenario(
        id=3,
        name="Rate limited — retry after backoff",
        difficulty="easy",
        description=(
            "GET /v1/users returns 429 Too Many Requests. The agent must "
            "recognise the rate limit and switch to a valid approach — "
            "e.g. retry with correct headers or wait for the window."
        ),
        faults_injected=[RATE_LIMITED],
        initial_request={
            "endpoint": "/v1/users",
            "method": "GET",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer valid-token-abc",
            },
            "body": None,
            "base_url": "http://localhost:8765",
        },
        expected_fix={
            # No request fix needed — agent must clear the fault via
            # understanding that rate limit is transient.
            # For grading purposes we just check success.
        },
        max_steps=8,
        noisy_logs=False,
        hint="[HINT] Check the Retry-After header in the 429 response.",
    ),

    # ------------------------------------------------------------------
    # Scenario 4 — MEDIUM: Wrong base URL + wrong auth scheme
    # ------------------------------------------------------------------
    Scenario(
        id=4,
        name="Wrong base URL + wrong auth scheme",
        difficulty="medium",
        description=(
            "The client is pointing to http://localhost:9999 (wrong port) and "
            "using Basic auth instead of Bearer. Two faults: fix the base_url "
            "to http://localhost:8765 and change auth scheme to Bearer."
        ),
        faults_injected=[WRONG_BASE_URL, WRONG_AUTH_SCHEME],
        initial_request={
            "endpoint": "/v1/users",
            "method": "POST",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Basic dXNlcjpwYXNz",
            },
            "body": {"user_id": 10, "email": "charlie@example.com"},
            "base_url": "http://localhost:9999",
        },
        expected_fix={
            "base_url": "http://localhost:8765",
            "headers.Authorization": "Bearer ",
        },
        max_steps=12,
        noisy_logs=False,
        hint="[HINT] The infra team changed the service port in last week's deploy.",
    ),

    # ------------------------------------------------------------------
    # Scenario 5 — HARD: All faults combined
    # ------------------------------------------------------------------
    Scenario(
        id=5,
        name="Kitchen sink — auth + payload + endpoint + noise",
        difficulty="hard",
        description=(
            "Everything is broken: expired token, deprecated v1 endpoint, "
            "user_id is a string, email is missing, role is missing. "
            "Logs are extremely noisy."
        ),
        faults_injected=[
            EXPIRED_TOKEN, DEPRECATED_ENDPOINT,
            INVALID_PAYLOAD_FIELD, WRONG_FIELD_TYPE,
        ],
        initial_request={
            "endpoint": "/v1/users",
            "method": "POST",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer expired-token-xyz",
            },
            "body": {"user_id": "99"},
            "base_url": "http://localhost:8765",
        },
        expected_fix={
            "headers.Authorization": "Bearer ",
            "endpoint": "/v2/users",
            "body.user_id": 99,
            "body.email": "",
            "body.role": "",
        },
        max_steps=25,
        noisy_logs=True,
        hint="[HINT] Multiple systems changed during the migration. Check everything.",
    ),
]


def get_scenario(scenario_id: int) -> Scenario:
    """Return scenario by id, raise ValueError if not found."""
    for s in SCENARIOS:
        if s.id == scenario_id:
            return s
    raise ValueError(f"No scenario with id={scenario_id}. Available: {[s.id for s in SCENARIOS]}")
