"""
env.py — Core OpenEnv environment for API Integration Incident Debugger.

Implements the full OpenEnv interface:
    reset(scenario_id)  → Observation
    step(action)        → (Observation, Reward, bool, dict)
    state()             → dict

Also manages:
  - Spawning / tearing down the mock server subprocess
  - Injecting faults into the mock server at episode start
  - Computing rewards per step
  - Tracking which faults have been resolved
"""

from __future__ import annotations

import copy
import json
import subprocess
import time
from typing import Any

import httpx

from models import (
    Action,
    APIRequest,
    APIResponse,
    Observation,
    Reward,
    ScenarioState,
)
from scenarios import SCENARIOS, Scenario, get_scenario

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOCK_SERVER_PORT = 8765
MOCK_BASE_URL = f"http://localhost:{MOCK_SERVER_PORT}"
SERVER_STARTUP_TIMEOUT = 15   # seconds to wait for mock server to be ready

# Log templates for different fault responses
_LOG_TEMPLATES: dict[str, list[str]] = {
    "401": [
        "ERROR [auth] Request rejected — authentication failure",
        "INFO  [middleware] Auth check failed at entry point",
        "DEBUG [request] Headers received: {headers}",
    ],
    "403": [
        "ERROR [auth] Token valid but scope insufficient",
        "INFO  [authz] Required scope: users:write — provided: users:read",
    ],
    "404": [
        "ERROR [router] No route matched for {method} {endpoint}",
        "INFO  [router] Available routes: POST /v2/users, GET /v2/users",
        "WARN  [deprecation] /v1/ prefix was retired in release 2.4.0",
    ],
    "422": [
        "ERROR [validation] Request body failed schema validation",
        "DEBUG [validator] Received body: {body}",
        "INFO  [schema] Required fields for this endpoint: user_id (int), email (str)",
    ],
    "429": [
        "WARN  [rate-limit] Client exceeded request quota",
        "INFO  [rate-limit] Current window: 60s — limit: 100 req/window",
    ],
    "200": [
        "INFO  [request] {method} {endpoint} → 200 OK ({latency_ms}ms)",
        "INFO  [auth] Token validated successfully",
        "INFO  [db] User record written to database",
    ],
}


class APIDebugEnv:
    """
    OpenEnv-compliant environment for API Integration Incident Debugger.

    Usage:
        env = APIDebugEnv()
        obs = env.reset(scenario_id=0)
        action = Action(action_type="inspect_logs", parameters={})
        obs, reward, done, info = env.step(action)
        state = env.state()
        env.close()
    """

    def __init__(self, mock_server_port: int = MOCK_SERVER_PORT) -> None:
        self.port = mock_server_port
        self.base_url = f"http://localhost:{self.port}"
        self._server_proc: subprocess.Popen | None = None
        self._scenario: Scenario | None = None
        self._state: ScenarioState | None = None
        self._current_request: APIRequest | None = None
        self._last_response: APIResponse | None = None
        self._logs: list[str] = []
        self._inspect_count: int = 0
        self._analyze_count: int = 0
        self._seen_error_types: set[int] = set()

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, scenario_id: int = 0, use_mock: bool = True) -> Observation:
        """
        Start a new episode:
        1. Ensure mock server is running (skipped if use_mock=False)
        2. Clear existing faults, inject scenario faults
        3. Return initial (broken) Observation

        Set use_mock=False when testing against a real/external API.
        """
        self._scenario = get_scenario(scenario_id)
        if use_mock:
            self._ensure_server_running()
            self._inject_faults(self._scenario)

        # Build initial broken request from scenario definition
        req_data = self._scenario.initial_request
        self._current_request = APIRequest(**req_data)

        # Reset counters
        self._state = ScenarioState(
            scenario_id=scenario_id,
            faults_injected=list(self._scenario.faults_injected),
        )
        self._inspect_count = 0
        self._analyze_count = 0
        self._seen_error_types = set()
        self._logs = []

        # Make the initial broken call to populate first response
        self._last_response = self._call_mock(self._current_request)
        self._logs = self._build_logs(self._current_request, self._last_response)

        # Prepend the scenario hint
        if self._scenario.hint:
            self._logs.insert(0, self._scenario.hint)

        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """
        Execute one action and return (Observation, Reward, done, info).
        """
        if self._state is None or self._scenario is None:
            raise RuntimeError("Call reset() before step()")

        self._state.step_count += 1
        reward = Reward()
        reward.step_reward = -0.05   # always pay step cost
        info: dict[str, Any] = {"action": action.action_type, "faults_remaining": []}

        # ---- Dispatch action ----------------------------------------
        at = action.action_type
        params = action.parameters

        action_key = f"{at}:{str(sorted(params.items()))}"
        repeated = action_key in [
            f"{h['action_type']}:{str(sorted(h['parameters'].items()))}"
            for h in self._state.action_history
        ]

        if at == "inspect_logs":
            reward, info = self._act_inspect_logs(reward, info, repeated)

        elif at == "analyze_response":
            reward, info = self._act_analyze_response(reward, info, repeated)

        elif at == "make_test_call":
            reward, info = self._act_make_test_call(reward, info, params)

        elif at == "patch_config":
            reward, info = self._act_patch_config(reward, info, params)

        elif at == "patch_request":
            reward, info = self._act_patch_request(reward, info, params)

        elif at == "submit_fix":
            reward, info = self._act_submit_fix(reward, info)

        # Repeated-action penalty (same action + same params twice)
        if repeated and at not in ("submit_fix",):
            reward.penalty += -0.10
            info["repeated_action"] = True

        # Record action
        self._state.action_history.append(
            {"action_type": at, "parameters": params, "step": self._state.step_count}
        )

        reward.compute_total()
        self._state.cumulative_reward += reward.total
        info["faults_remaining"] = [
            f for f in self._state.faults_injected
            if f not in self._state.faults_resolved
        ]
        info["cumulative_reward"] = round(self._state.cumulative_reward, 4)

        # Check budget exhaustion
        budget_used = self._state.step_count >= self._scenario.max_steps
        done = self._state.done or budget_used
        if budget_used and not self._state.done:
            info["termination"] = "budget_exhausted"

        obs = self._build_observation()
        return obs, reward, done, info

    def state(self) -> dict[str, Any]:
        """Return full internal state as a plain dict."""
        if self._state is None:
            return {}
        return {
            "scenario_id": self._state.scenario_id,
            "scenario_name": self._scenario.name if self._scenario else "",
            "difficulty": self._scenario.difficulty if self._scenario else "",
            "faults_injected": self._state.faults_injected,
            "faults_resolved": self._state.faults_resolved,
            "step_count": self._state.step_count,
            "max_steps": self._scenario.max_steps if self._scenario else 0,
            "cumulative_reward": round(self._state.cumulative_reward, 4),
            "action_history": self._state.action_history,
            "done": self._state.done,
            "current_request": self._current_request.model_dump() if self._current_request else {},
            "last_response": self._last_response.model_dump() if self._last_response else {},
        }

    def close(self) -> None:
        """Shut down the mock server subprocess."""
        if self._server_proc and self._server_proc.poll() is None:
            self._server_proc.terminate()
            self._server_proc.wait(timeout=5)
            self._server_proc = None

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _act_inspect_logs(
        self, reward: Reward, info: dict, repeated: bool
    ) -> tuple[Reward, dict]:
        self._inspect_count += 1
        if self._inspect_count <= 2 and not repeated:
            reward.action_quality_reward += 0.10
        info["logs_shown"] = len(self._logs)
        return reward, info

    def _act_analyze_response(
        self, reward: Reward, info: dict, repeated: bool
    ) -> tuple[Reward, dict]:
        self._analyze_count += 1
        if self._analyze_count <= 2 and not repeated:
            reward.action_quality_reward += 0.10
        info["response_status"] = self._last_response.status_code if self._last_response else None
        return reward, info

    def _act_make_test_call(
        self, reward: Reward, info: dict, params: dict
    ) -> tuple[Reward, dict]:
        # Build a test request from params, falling back to current request fields
        cur = self._current_request

        raw_body = params.get("body", copy.deepcopy(cur.body))
        if isinstance(raw_body, str):
            try:
                parsed_body = json.loads(raw_body)
                if isinstance(parsed_body, dict):
                    raw_body = parsed_body
                else:
                    info["body_parse_warning"] = "body string parsed but is not a JSON object"
            except json.JSONDecodeError:
                # allow APIRequest model validator to handle invalid body types
                info["body_parse_warning"] = "body must be JSON object; kept as raw string"

        test_req = APIRequest(
            endpoint=params.get("endpoint", cur.endpoint),
            method=params.get("method", cur.method),
            headers=params.get("headers", dict(cur.headers)),
            body=raw_body,
            base_url=params.get("base_url", cur.base_url),
        )
        resp = self._call_mock(test_req)
        reward.step_reward += -0.05   # extra wasted-call cost

        new_error = resp.status_code not in self._seen_error_types
        if new_error and resp.status_code != 200:
            reward.action_quality_reward += 0.15
            self._seen_error_types.add(resp.status_code)

        self._last_response = resp
        self._logs = self._build_logs(test_req, resp)
        info["test_call_status"] = resp.status_code
        info["new_error_type_discovered"] = new_error
        return reward, info

    def _act_patch_config(
        self, reward: Reward, info: dict, params: dict
    ) -> tuple[Reward, dict]:
        key = params.get("key", "")
        value = params.get("value", "")
        if not key:
            info["patch_error"] = "key param required for patch_config"
            return reward, info

        cur = self._current_request
        if key == "base_url":
            self._current_request = cur.model_copy(update={"base_url": value})
        else:
            # Treat as a header key
            new_headers = dict(cur.headers)
            new_headers[key] = value
            self._current_request = cur.model_copy(update={"headers": new_headers})

        # Check if this patch resolved any faults
        partial = self._check_partial_fixes()
        reward.partial_fix_reward += partial * 0.30
        info["patch_applied"] = {key: value}
        info["faults_newly_resolved"] = self._state.faults_resolved[:]
        return reward, info

    def _act_patch_request(
        self, reward: Reward, info: dict, params: dict
    ) -> tuple[Reward, dict]:
        field = params.get("field", "")
        value = params.get("value")
        if not field:
            info["patch_error"] = "field param required for patch_request"
            return reward, info

        cur = self._current_request
        if field == "endpoint":
            self._current_request = cur.model_copy(update={"endpoint": value})
        elif field == "method":
            self._current_request = cur.model_copy(update={"method": value})
        elif field.startswith("body."):
            body_key = field[5:]
            new_body = dict(cur.body or {})
            new_body[body_key] = value
            self._current_request = cur.model_copy(update={"body": new_body})
        elif field.startswith("headers."):
            header_key = field[8:]
            new_headers = dict(cur.headers)
            new_headers[header_key] = value
            self._current_request = cur.model_copy(update={"headers": new_headers})
        else:
            info["patch_error"] = f"Unknown field path: {field}"
            return reward, info

        # Check if this patch resolved any faults
        partial = self._check_partial_fixes()
        reward.partial_fix_reward += partial * 0.30
        info["patch_applied"] = {field: value}
        return reward, info

    def _act_submit_fix(
        self, reward: Reward, info: dict
    ) -> tuple[Reward, dict]:
        # Make the actual call with the current patched request
        resp = self._call_mock(self._current_request)
        self._last_response = resp
        self._logs = self._build_logs(self._current_request, resp)

        if resp.status_code == 200:
            reward.success_reward = 1.50
            steps_used = self._state.step_count
            max_steps = self._scenario.max_steps
            reward.efficiency_bonus = max(0.0, 0.50 * (1 - steps_used / max_steps))
            self._state.done = True
            info["result"] = "success"
        else:
            reward.penalty += -0.20   # premature / wrong submission
            info["result"] = "failure"
            info["submit_response"] = resp.model_dump()

        return reward, info

    # ------------------------------------------------------------------
    # Fault resolution checker
    # ------------------------------------------------------------------

    def _check_partial_fixes(self) -> int:
        """
        Compare current request against scenario expected_fix.
        Returns the number of *newly* resolved faults this step.
        """
        scenario = self._scenario
        req = self._current_request
        newly_resolved = 0

        for fault_key, expected_value in scenario.expected_fix.items():
            fault_name = self._fault_key_to_name(fault_key)
            if fault_name in self._state.faults_resolved:
                continue  # already counted

            actual = self._extract_field(req, fault_key)
            resolved = False

            if isinstance(expected_value, str) and expected_value == "Bearer ":
                # Header must start with "Bearer " and not contain "expired"
                resolved = (
                    isinstance(actual, str)
                    and actual.startswith("Bearer ")
                    and "expired" not in actual.lower()
                )
            elif isinstance(expected_value, str) and expected_value == "":
                # Any non-empty string satisfies this
                resolved = isinstance(actual, str) and len(actual) > 0
            else:
                resolved = actual == expected_value

            if resolved:
                self._state.faults_resolved.append(fault_name)
                newly_resolved += 1

        return newly_resolved

    def _fault_key_to_name(self, fault_key: str) -> str:
        """Map expected_fix key to a fault name string for tracking."""
        mapping = {
            "headers.Authorization": "AUTH_FIXED",
            "endpoint": "ENDPOINT_FIXED",
            "body.user_id": "TYPE_FIXED",
            "body.email": "EMAIL_FIXED",
            "body.role": "ROLE_FIXED",
        }
        return mapping.get(fault_key, fault_key)

    def _extract_field(self, req: APIRequest, field_path: str) -> Any:
        """Extract a nested field from the request by dot-path."""
        if field_path.startswith("headers."):
            key = field_path[8:]
            return req.headers.get(key, req.headers.get(key.lower(), ""))
        elif field_path.startswith("body."):
            key = field_path[5:]
            return (req.body or {}).get(key)
        elif field_path == "endpoint":
            return req.endpoint
        elif field_path == "base_url":
            return req.base_url
        return None

    # ------------------------------------------------------------------
    # Mock server helpers
    # ------------------------------------------------------------------

    def _call_mock(self, req: APIRequest) -> APIResponse:
        """Send a real HTTP request to the mock server."""
        url = f"{req.base_url}{req.endpoint}"
        try:
            with httpx.Client(timeout=5.0) as client:
                t0 = time.time()
                resp = client.request(
                    method=req.method,
                    url=url,
                    headers=req.headers,
                    json=req.body,
                )
                latency = (time.time() - t0) * 1000
                try:
                    body = resp.json()
                except Exception:
                    body = {"raw": resp.text}
                return APIResponse(
                    status_code=resp.status_code,
                    body=body,
                    error_message=body.get("error"),
                    latency_ms=round(latency, 2),
                )
        except httpx.ConnectError:
            return APIResponse(
                status_code=503,
                body={"error": "Connection refused — check base_url"},
                error_message="Connection refused — check base_url",
                latency_ms=0.0,
            )
        except Exception as exc:
            return APIResponse(
                status_code=500,
                body={"error": str(exc)},
                error_message=str(exc),
                latency_ms=0.0,
            )

    def _inject_faults(self, scenario: Scenario) -> None:
        """Clear existing faults and inject this scenario's faults."""
        with httpx.Client(timeout=5.0) as client:
            client.post(f"{self.base_url}/clear-faults")
            for fault in scenario.faults_injected:
                client.post(
                    f"{self.base_url}/inject-fault",
                    json={"fault_type": fault},
                )
            # Set noise seed for reproducibility
            client.post(
                f"{self.base_url}/set-noise-seed",
                json={"seed": scenario.id},
            )

    def _ensure_server_running(self) -> None:
        """Start mock server subprocess if not already running."""
        # Check if already alive
        try:
            with httpx.Client(timeout=1.0) as client:
                r = client.get(f"{self.base_url}/health")
                if r.status_code == 200:
                    return
        except Exception:
            pass

        # Start it
        self._server_proc = subprocess.Popen(
            [
                "uvicorn",
                "mock_server:app",
                "--host", "0.0.0.0",
                "--port", str(self.port),
                "--log-level", "error",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Poll until healthy
        deadline = time.time() + SERVER_STARTUP_TIMEOUT
        while time.time() < deadline:
            time.sleep(0.3)
            try:
                with httpx.Client(timeout=1.0) as client:
                    r = client.get(f"{self.base_url}/health")
                    if r.status_code == 200:
                        return
            except Exception:
                continue

        raise RuntimeError(
            f"Mock server did not start within {SERVER_STARTUP_TIMEOUT}s. "
            "Make sure uvicorn is installed and port 8765 is free."
        )

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_logs(self, req: APIRequest, resp: APIResponse) -> list[str]:
        """Generate structured log lines for the current request/response."""
        status_key = str(resp.status_code)
        templates = _LOG_TEMPLATES.get(status_key, [
            f"INFO  [request] {req.method} {req.endpoint} → {resp.status_code}",
        ])
        lines = []
        for tpl in templates:
            lines.append(
                tpl.format(
                    method=req.method,
                    endpoint=req.endpoint,
                    headers=dict(req.headers),
                    body=req.body,
                    latency_ms=resp.latency_ms,
                )
            )
        if resp.error_message:
            lines.append(f"ERROR [api] {resp.error_message}")

        # Interleave noise if hard scenario
        if self._scenario and self._scenario.noisy_logs:
            import random
            rng = random.Random(self._state.step_count if self._state else 0)
            noisy: list[str] = []
            noise_pool = [
                "WARN  [db-pool] Connection pool pressure: 78% utilised",
                "INFO  [metrics] p99 latency spike detected",
                "ERROR [cache] Redis TIMEOUT — falling back to DB",
                "WARN  [health] Downstream /analytics responding slowly",
                "INFO  [gc] GC pause 312ms",
                "ERROR [worker] Task queue depth exceeded soft limit",
            ]
            for line in lines:
                noisy.append(line)
                if rng.random() < 0.4:
                    noisy.append(rng.choice(noise_pool))
            lines = noisy

        return lines[-20:]

    def _build_observation(self) -> Observation:
        return Observation(
            request=self._current_request,
            response=self._last_response,
            logs=list(self._logs),
            step_count=self._state.step_count,
            action_history=[
                h["action_type"] for h in self._state.action_history
            ],
            budget_remaining=self._scenario.max_steps - self._state.step_count,
        )
