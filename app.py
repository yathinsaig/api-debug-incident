"""
app.py — FastAPI web server exposing the OpenEnv API for HF Spaces.

Endpoints:
    GET  /           → health check (returns 200)
    POST /reset      → reset(scenario_id) → Observation
    POST /step       → step(action)       → (Observation, Reward, done, info)
    GET  /state      → state()            → dict
    GET  /scenarios   → list all available scenarios
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env import APIDebugEnv
from models import Action
from scenarios import SCENARIOS

app = FastAPI(title="API Incident Debugger — OpenEnv", version="1.0.0")
env = APIDebugEnv()


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    scenario_id: int = 0


class StepRequest(BaseModel):
    action_type: str
    parameters: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "environment": "api-incident-debugger", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest):
    obs = env.reset(scenario_id=req.scenario_id)
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    action = Action(action_type=req.action_type, parameters=req.parameters)
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state()


@app.get("/scenarios")
def list_scenarios():
    return [
        {
            "id": s.id,
            "name": s.name,
            "difficulty": s.difficulty,
            "description": s.description,
            "max_steps": s.max_steps,
        }
        for s in SCENARIOS
    ]
