"""
baseline.py — Groq-powered baseline agent for API Integration Incident Debugger.

Uses llama3-70b-8192 via the Groq SDK.
Reads GROQ_API_KEY from environment variable.

Usage:
    python baseline.py                   # run all 3 scenarios
    python baseline.py --scenario 0      # run only scenario 0
    python baseline.py --scenario 2      # run only scenario 2 (hard)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any
from dotenv import load_dotenv

from groq import Groq

# load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
load_dotenv()  # Load from current directory .env if it exists

from env import APIDebugEnv
from grader import APIDebugGrader, GradeResult
from models import Action, Observation
from scenarios import SCENARIOS



# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "llama-3.1-8b-instant"
MAX_RETRIES = 2       # JSON parse retries per step
SLEEP_BETWEEN_STEPS = 0.3   # seconds (be kind to rate limits)

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
# Helper: build user prompt from observation
# ---------------------------------------------------------------------------

def build_user_prompt(obs: Observation, scenario_name: str) -> str:
    logs_text = "\n".join(obs.logs) if obs.logs else "(no logs yet)"
    body_text = json.dumps(obs.request.body, indent=2) if obs.request.body else "null"

    return f"""=== API DEBUGGING TASK: {scenario_name} ===

CURRENT REQUEST:
  Method:   {obs.request.method}
  Base URL: {obs.request.base_url}
  Endpoint: {obs.request.endpoint}
  Headers:  {json.dumps(obs.request.headers, indent=2)}
  Body:
{body_text}

LAST RESPONSE:
  Status:  {obs.response.status_code}
  Error:   {obs.response.error_message or 'none'}
  Body:    {json.dumps(obs.response.body, indent=2)}

RECENT LOGS:
{logs_text}

STEP: {obs.step_count} / budget remaining: {obs.budget_remaining}
ACTION HISTORY: {obs.action_history if obs.action_history else 'none yet'}

What is your next action? Respond with ONLY the JSON action object."""


# ---------------------------------------------------------------------------
# Helper: call Groq and parse action
# ---------------------------------------------------------------------------

def get_action_from_llm(
    client: Groq,
    messages: list[dict],
    retries: int = MAX_RETRIES,
) -> Action | None:
    """Call Groq API and parse the response as an Action. Returns None on failure."""
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=256,
            )
            raw = response.choices[0].message.content.strip()

            # Strip accidental markdown fences if the model adds them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            data = json.loads(raw)
            action = Action(**data)
            return action

        except json.JSONDecodeError as e:
            print(f"    [warn] JSON parse error (attempt {attempt+1}): {e}")
            if attempt < retries:
                # Ask model to fix its output
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
            print(f"    [warn] Action parse error (attempt {attempt+1}): {e}")
            time.sleep(0.5)

    return None


# ---------------------------------------------------------------------------
# Run a single scenario
# ---------------------------------------------------------------------------

def run_scenario(
    env: APIDebugEnv,
    client: Groq,
    grader: APIDebugGrader,
    scenario_id: int,
) -> GradeResult:
    from scenarios import get_scenario
    scenario = get_scenario(scenario_id)
    print(f"\n{'='*60}")
    print(f"  Scenario {scenario_id}: {scenario.name}")
    print(f"  Difficulty: {scenario.difficulty.upper()} | Budget: {scenario.max_steps} steps")
    print(f"{'='*60}")

    obs = env.reset(scenario_id)
    trajectory: list[dict[str, Any]] = []
    total_reward = 0.0
    final_status = obs.response.status_code
    done = False

    # Conversation history for multi-turn context
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while not done:
        user_prompt = build_user_prompt(obs, scenario.name)
        messages.append({"role": "user", "content": user_prompt})

        print(f"\n  Step {obs.step_count + 1} | Status: {obs.response.status_code} | "
              f"Budget left: {obs.budget_remaining}")

        action = get_action_from_llm(client, messages)
        if action is None:
            print("  [error] Failed to get valid action from LLM. Skipping step.")
            # Use a safe fallback action
            action = Action(action_type="inspect_logs", parameters={})

        print(f"  Action: {action.action_type} | Params: {action.parameters}")

        obs, reward, done, info = env.step(action)
        total_reward += reward.total
        final_status = obs.response.status_code

        # Record for grader
        trajectory.append({
            "action_type": action.action_type,
            "parameters": action.parameters,
            "reward_total": reward.total,
            "response_status": obs.response.status_code,
            "step": obs.step_count,
        })

        # Add assistant turn to conversation history
        messages.append({
            "role": "assistant",
            "content": json.dumps(action.model_dump()),
        })

        print(f"  Reward: {reward.total:+.3f} (cumulative: {total_reward:+.3f})")
        if info.get("result") == "success":
            print("  *** FIX ACCEPTED — Status 200 ***")
        elif info.get("result") == "failure":
            print(f"  ✗ Submit rejected — {obs.response.error_message}")

        time.sleep(SLEEP_BETWEEN_STEPS)

    # Grade the episode
    result = grader.score(
        trajectory=trajectory,
        max_steps=scenario.max_steps,
        scenario_id=scenario_id,
        scenario_name=scenario.name,
        difficulty=scenario.difficulty,
        final_status=final_status,
        total_reward=total_reward,
    )

    print(f"\n  Final score: {result.final_score:.4f} | "
          f"Success: {result.success} | Steps: {result.steps_taken}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="API Debug Env — Groq Baseline Agent")
    parser.add_argument(
        "--scenario",
        type=int,
        default=None,
        help="Run a specific scenario (0=easy, 1=medium, 2=hard). Default: all.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable is not set.")
        print("Export it with: export GROQ_API_KEY=your_key_here")
        sys.exit(1)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    env = APIDebugEnv()
    grader = APIDebugGrader()

    scenario_ids = (
        [args.scenario]
        if args.scenario is not None
        else [s.id for s in SCENARIOS]
    )

    results: list[GradeResult] = []
    try:
        for sid in scenario_ids:
            result = run_scenario(env, client, grader, sid)
            results.append(result)
    finally:
        env.close()

    grader.print_report(results)


if __name__ == "__main__":
    main()
