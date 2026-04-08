"""
Inference Script — API Integration Incident Debugger
=====================================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    GROQ_API_KEY   Your Groq API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, List, Optional

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

API_KEY = os.getenv("GROQ_API_KEY") or HF_TOKEN or os.getenv("API_KEY")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")
BENCHMARK = "api-incident-debugger"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_RETRIES = 2
SLEEP_BETWEEN_STEPS = 0.3

SYSTEM_PROMPT = """You are an expert API debugging agent. Your job is to diagnose and fix a broken API integration step-by-step.

You will receive the current state of the integration: the request being made, the error response received, recent log lines, and your action history.

RULES:
1. You MUST respond with ONLY a valid JSON object. No explanation, no markdown, no prose, no code fences.
2. The JSON must match this exact schema:
   {"action_type": "<type>", "parameters": {<optional key-value pairs>}}
3. Valid action_type values:
   - "inspect_logs"      → read the log lines carefully (no parameters needed)
   - "analyze_response"  → analyse the HTTP status and error message (no parameters needed)
   - "make_test_call"    → make a test HTTP call. Parameters: endpoint, method, headers, body, base_url
   - "patch_config"      → change a config value. Parameters: key (e.g. "Authorization" or "base_url"), value
   - "patch_request"     → modify the request. Parameters: field (e.g. "endpoint", "body.user_id", "headers.Authorization"), value
   - "submit_fix"        → declare the fix complete and trigger final validation (no parameters needed)

STRATEGY:
- Start by inspecting logs and analysing the response BEFORE making changes.
- Form a hypothesis about the root cause from the error message and logs.
- Make targeted patches one at a time.
- Only submit_fix when you are confident the request is fully corrected.
- Avoid repeating the same action with the same parameters.

Output ONLY the JSON. Nothing else."""


# ---------------------------------------------------------------------------
# Stdout logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment interaction via HTTP
# ---------------------------------------------------------------------------

import httpx

HTTP_TIMEOUT = 30.0


def env_reset(scenario_id: int) -> dict:
    with httpx.Client(timeout=HTTP_TIMEOUT) as c:
        resp = c.post(f"{ENV_URL}/reset", json={"scenario_id": scenario_id})
        resp.raise_for_status()
        return resp.json()


def env_step(action_type: str, parameters: dict) -> dict:
    with httpx.Client(timeout=HTTP_TIMEOUT) as c:
        resp = c.post(
            f"{ENV_URL}/step",
            json={"action_type": action_type, "parameters": parameters},
        )
        resp.raise_for_status()
        return resp.json()


def env_scenarios() -> list[dict]:
    with httpx.Client(timeout=HTTP_TIMEOUT) as c:
        resp = c.get(f"{ENV_URL}/scenarios")
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Build user prompt from observation
# ---------------------------------------------------------------------------

def build_user_prompt(obs: dict, scenario_name: str) -> str:
    request = obs.get("request", {})
    response = obs.get("response", {})
    logs = obs.get("logs", [])
    step_count = obs.get("step_count", 0)
    budget_remaining = obs.get("budget_remaining", 0)
    action_history = obs.get("action_history", [])

    logs_text = "\n".join(logs) if logs else "(no logs yet)"
    body_text = json.dumps(request.get("body"), indent=2) if request.get("body") else "null"

    return f"""=== API DEBUGGING TASK: {scenario_name} ===

CURRENT REQUEST:
  Method:   {request.get('method', '')}
  Base URL: {request.get('base_url', '')}
  Endpoint: {request.get('endpoint', '')}
  Headers:  {json.dumps(request.get('headers', {}), indent=2)}
  Body:
{body_text}

LAST RESPONSE:
  Status:  {response.get('status_code', 0)}
  Error:   {response.get('error_message') or 'none'}
  Body:    {json.dumps(response.get('body', {}), indent=2)}

RECENT LOGS:
{logs_text}

STEP: {step_count} / budget remaining: {budget_remaining}
ACTION HISTORY: {action_history if action_history else 'none yet'}

What is your next action? Respond with ONLY the JSON action object."""


# ---------------------------------------------------------------------------
# LLM call — uses OpenAI Client
# ---------------------------------------------------------------------------

def get_action_from_llm(
    client: OpenAI,
    messages: list[dict],
    retries: int = MAX_RETRIES,
) -> dict | None:
    raw = ""
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_tokens=256,
            )
            raw = (response.choices[0].message.content or "").strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            data = json.loads(raw)
            return data

        except json.JSONDecodeError as e:
            print(f"    [warn] JSON parse error (attempt {attempt+1}): {e}", flush=True)
            if attempt < retries:
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON. "
                        "Respond with ONLY a valid JSON object matching the Action schema."
                    ),
                })
                time.sleep(0.5)
        except Exception as e:
            print(f"    [warn] LLM/parse error (attempt {attempt+1}): {e}", flush=True)
            time.sleep(0.5)

    return None


# ---------------------------------------------------------------------------
# Scoring helpers (mirrors grader.py logic)
# ---------------------------------------------------------------------------

def compute_score(trajectory: list[dict], max_steps: int, final_status: int) -> float:
    steps_taken = len(trajectory)
    success = final_status == 200

    success_score = 1.0 if success else 0.0
    efficiency_score = max(0.0, 1.0 - steps_taken / max_steps) if steps_taken > 0 else 0.0

    diagnostic_types = {"inspect_logs", "analyze_response"}
    patch_types = {"patch_config", "patch_request", "make_test_call", "submit_fix"}
    pre_patch_steps = []
    for t in trajectory:
        at = t.get("action_type", "")
        if at in patch_types:
            break
        pre_patch_steps.append(at)

    if pre_patch_steps:
        reasoning_score = sum(1 for at in pre_patch_steps if at in diagnostic_types) / len(pre_patch_steps)
    else:
        reasoning_score = 0.0

    return round(min(1.0, max(0.0,
        0.50 * success_score + 0.25 * efficiency_score + 0.25 * reasoning_score
    )), 4)


# ---------------------------------------------------------------------------
# Run a single scenario
# ---------------------------------------------------------------------------

def run_scenario(client: OpenAI, scenario: dict) -> dict:
    scenario_id = scenario["id"]
    scenario_name = scenario["name"]
    max_steps = scenario.get("max_steps", 10)
    task_name = f"scenario-{scenario_id}-{scenario_name.replace(' ', '-').lower()}"

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        obs = env_reset(scenario_id)

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        trajectory: list[dict] = []
        done = False

        while not done:
            user_prompt = build_user_prompt(obs, scenario_name)
            messages.append({"role": "user", "content": user_prompt})

            action_data = get_action_from_llm(client, messages)
            if action_data is None:
                action_data = {"action_type": "inspect_logs", "parameters": {}}

            action_type = action_data.get("action_type", "inspect_logs")
            parameters = action_data.get("parameters", {})

            error_msg = None
            try:
                result = env_step(action_type, parameters)
            except Exception as e:
                error_msg = str(e)
                log_step(
                    step=steps_taken + 1,
                    action=action_type,
                    reward=0.0,
                    done=True,
                    error=error_msg,
                )
                rewards.append(0.0)
                steps_taken += 1
                break

            obs = result.get("observation", {})
            reward_data = result.get("reward", {})
            done = result.get("done", False)
            info = result.get("info", {})

            reward_total = reward_data.get("total", 0.0)
            rewards.append(reward_total)
            steps_taken += 1

            trajectory.append({
                "action_type": action_type,
                "parameters": parameters,
                "reward_total": reward_total,
                "response_status": obs.get("response", {}).get("status_code", 0),
                "step": steps_taken,
            })

            # Check for errors from the environment
            last_error = info.get("patch_error") or None
            if info.get("result") == "failure":
                last_error = obs.get("response", {}).get("error_message")

            log_step(
                step=steps_taken,
                action=action_type,
                reward=reward_total,
                done=done,
                error=last_error,
            )

            messages.append({
                "role": "assistant",
                "content": json.dumps(action_data),
            })

            time.sleep(SLEEP_BETWEEN_STEPS)

        final_status = obs.get("response", {}).get("status_code", 0)
        success = final_status == 200
        score = compute_score(trajectory, max_steps, final_status)

    except Exception as e:
        print(f"    [error] Scenario {scenario_id} exception: {e}", flush=True)
        traceback.print_exc()

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "success": success,
        "score": score,
        "steps_taken": steps_taken,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("ERROR: No API key set. Set GROQ_API_KEY, HF_TOKEN, or API_KEY.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    try:
        all_scenarios = env_scenarios()
    except Exception as e:
        print(f"ERROR: Cannot reach env at {ENV_URL}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    results: list[dict] = []
    for scenario in all_scenarios:
        try:
            result = run_scenario(client, scenario)
            results.append(result)
        except Exception as e:
            print(f"    [error] Scenario {scenario['id']} failed: {e}", flush=True)
            traceback.print_exc()
            results.append({
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "success": False,
                "score": 0.0,
                "steps_taken": 0,
            })

    # Print summary
    if results:
        avg_score = sum(r["score"] for r in results) / len(results)
        successes = sum(1 for r in results if r["success"])
        print(f"\n--- SUMMARY: {successes}/{len(results)} scenarios passed, avg score: {avg_score:.4f} ---", flush=True)


if __name__ == "__main__":
    main()
