"""
DataPipeline-Sentry: Baseline Inference Script
================================================
Runs all three tasks against the live HF Space using an LLM for action
selection.  Follows the OpenEnv Phase 2 structured-logging protocol:

    [START] task=NAME
    [STEP]  step=N reward=R
    [END]   task=NAME score=S steps=N

All prints use flush=True for evaluator compatibility.
"""

import json
import os

import requests
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────
API_KEY = os.environ.get("API_KEY", "dummy_key")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_BASE_URL = os.environ.get("API_BASE_URL")  # LiteLLM proxy

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL, timeout=30.0)

BASE_URL = "https://hello12334-openenv-data-cleaner.hf.space"

MAX_STEPS = 15

# ── Task Descriptions (used in LLM prompt) ───────────────────────────────────
TASK_DESCRIPTIONS = {
    "schema_drift": (
        "Easy — An upstream API changed its JSON schema (trade_price→px, "
        "trade_volume→vol, added _metadata). Use inspect_schema() to find "
        "the mismatch, then apply_mapping_patch(field_mapping) with the "
        "correct mapping to fix the pipeline. Finally submit."
    ),
    "late_data": (
        "Medium — Trade data is arriving out of order, causing incorrect "
        "moving averages. Use inspect_schema() to see the data, then call "
        "watermark_reprocess(watermark_threshold_ms) with a reasonable "
        "threshold (5000–60000 ms) to re-sort and fix rolling averages. "
        "Finally submit."
    ),
    "poisoned_records": (
        "Hard — A Data Poisoning event has occurred. Use "
        "statistical_traceback() to get per-source statistics, identify "
        "the anomalous source, and call quarantine_source(source_id) "
        "with the correct source ID. Finally submit."
    ),
}


def llm_choose_action(observation: dict, task: str) -> dict | None:
    """Ask the LLM to choose the next action given the current observation."""
    desc = TASK_DESCRIPTIONS.get(task, task)
    prompt = f"""You are an autonomous AI Data Reliability Engineer working on a Financial Data Lake.

CURRENT TASK: {desc}

OBSERVATION (JSON):
{json.dumps(observation, indent=2, default=str)}

AVAILABLE ACTIONS (choose ONE):
- {{"action": "inspect_schema"}}
- {{"action": "apply_mapping_patch", "field_mapping": {{"old_name": "new_name", ...}}}}
- {{"action": "watermark_reprocess", "watermark_threshold_ms": <int>}}
- {{"action": "statistical_traceback"}}
- {{"action": "quarantine_source", "source_id": "<source_id>"}}
- {{"action": "submit"}}

Return ONLY a JSON object with key "action" matching one of the schemas above.
Think step by step: what has been done so far (step_count, data_snapshot),
what is the logical next action for this task?
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"LLM API Exception: {e}", flush=True)
        return None


def run_inference() -> None:
    """Execute all three tasks with structured logging."""
    print("--- DataPipeline-Sentry Inference ---", flush=True)

    tasks = ["schema_drift", "late_data", "poisoned_records"]

    for task in tasks:
        # ── [START] ──────────────────────────────────────────────────────
        print(f"[START] task={task}", flush=True)

        # Reset environment
        try:
            res = requests.post(f"{BASE_URL}/reset", json={"task": task}, timeout=30)
            res.raise_for_status()
            obs = res.json().get("observation", {})
        except Exception as e:
            print(f"Error resetting env for task {task}: {e}", flush=True)
            print(f"[END] task={task} score=0.0 steps=0", flush=True)
            continue

        done = False
        steps = 0
        reward = 0.0

        while not done and steps < MAX_STEPS:
            action_payload = llm_choose_action(obs, task)

            # Fallback: submit if LLM fails
            if action_payload is None:
                action_payload = {"action": {"action": "submit"}}

            # Wrap bare action dict if needed
            if "action" in action_payload and isinstance(action_payload["action"], str):
                action_payload = {"action": action_payload}

            steps += 1
            try:
                res = requests.post(
                    f"{BASE_URL}/step", json=action_payload, timeout=30
                )
                res.raise_for_status()
                data = res.json()
                obs = data.get("observation", {})
                reward = float(data.get("reward", 0.0))
                done = data.get("done", True)
            except Exception as e:
                print(f"Error executing step: {e}", flush=True)
                done = True

            # ── [STEP] ──────────────────────────────────────────────────
            print(f"[STEP] step={steps} reward={reward}", flush=True)

        # Get final score from grader
        try:
            grader_res = requests.get(f"{BASE_URL}/grader", timeout=15)
            grader_res.raise_for_status()
            score = float(grader_res.json().get("score", 0.0))
        except Exception as e:
            print(f"Error getting grader score: {e}", flush=True)
            score = 0.0

        # ── [END] ────────────────────────────────────────────────────────
        print(f"[END] task={task} score={score} steps={steps}", flush=True)


if __name__ == "__main__":
    run_inference()
