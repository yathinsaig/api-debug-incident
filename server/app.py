"""
server/app.py — FastAPI web server exposing the OpenEnv API.

This is the entry point for multi-mode deployment.
"""

from __future__ import annotations

import sys
import os
from typing import Any

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel

from env import APIDebugEnv
from models import Action
from scenarios import SCENARIOS

app = FastAPI(title="API Incident Debugger — OpenEnv", version="1.0.0")
env = APIDebugEnv()


class ResetRequest(BaseModel):
    scenario_id: int = 0


class StepRequest(BaseModel):
    action_type: str
    parameters: dict[str, Any] = {}


@app.get("/")
def root():
    return {"status": "ok", "environment": "api-incident-debugger", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
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


def main():
    """Entry point for project.scripts."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
