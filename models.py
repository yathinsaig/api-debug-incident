"""
models.py — Pydantic schemas for API Integration Incident Debugger.
All models use strict extra="forbid" to catch accidental field mismatches.
"""

from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
import json


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------

class APIRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: str = Field(..., description="Path component e.g. /v1/users")
    method: str = Field(..., description="HTTP verb: GET | POST | PUT | DELETE")
    headers: dict[str, str] = Field(default_factory=dict)
    body: Optional[dict[str, Any]] = Field(default=None)
    base_url: str = Field(default="http://localhost:8765", description="Supports staging vs prod fault")

    @field_validator("body", mode="before")
    def parse_body(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                # keep original value for the final validation error path
                pass
        return v


class APIResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status_code: int
    body: dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: APIRequest
    response: APIResponse
    logs: list[str] = Field(default_factory=list, description="Last 20 log lines")
    step_count: int = 0
    action_history: list[str] = Field(default_factory=list)
    budget_remaining: int = Field(..., description="Max steps remaining")


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

ActionType = Literal[
    "inspect_logs",
    "analyze_response",
    "make_test_call",
    "patch_config",
    "patch_request",
    "submit_fix",
]


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType = Field(..., description="One of the six allowed action types")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Action-specific kwargs.\n"
            "  make_test_call  → {endpoint, method, headers, body, base_url}\n"
            "  patch_config    → {key: str, value: str}  (base_url or header key)\n"
            "  patch_request   → {field: str, value: Any} (endpoint/method/body field)\n"
            "  others          → {} (no params required)"
        ),
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_reward: float = 0.0
    action_quality_reward: float = 0.0
    partial_fix_reward: float = 0.0
    success_reward: float = 0.0
    efficiency_bonus: float = 0.0
    penalty: float = 0.0
    total: float = 0.0

    def compute_total(self) -> "Reward":
        self.total = (
            self.step_reward
            + self.action_quality_reward
            + self.partial_fix_reward
            + self.success_reward
            + self.efficiency_bonus
            + self.penalty
        )
        return self


# ---------------------------------------------------------------------------
# Internal state helpers
# ---------------------------------------------------------------------------

class ScenarioState(BaseModel):
    """Tracks which faults are still active during an episode."""
    model_config = ConfigDict(extra="forbid")

    scenario_id: int
    faults_injected: list[str]
    faults_resolved: list[str] = Field(default_factory=list)
    step_count: int = 0
    cumulative_reward: float = 0.0
    action_history: list[dict[str, Any]] = Field(default_factory=list)
    done: bool = False
