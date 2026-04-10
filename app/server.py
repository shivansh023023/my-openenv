"""
DataPipeline-Sentry: FastAPI Server
====================================
OpenEnv-compliant REST API exposing /reset, /step, /state, /tasks,
/grader, /baseline, and / (health check) endpoints.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import Body, FastAPI

from app.env import DataPipelineSentryEnv
from app.models import ActionRequest, Observation, ResetRequest, StepResponse

# ── Singleton Environment ────────────────────────────────────────────────────
env = DataPipelineSentryEnv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise default task on startup; no special teardown needed."""
    env.reset()
    yield


app = FastAPI(
    title="DataPipeline-Sentry",
    description="Financial Data Lake Reliability Environment for OpenEnv Phase 2",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    """Health-check / ping."""
    return {"status": "ok", "environment": "DataPipeline-Sentry"}


@app.post("/reset")
def reset_env(req: Optional[ResetRequest] = Body(default=None)):
    """Reset the environment to a fresh episode of the given task."""
    task_name = (
        req.task if (req is not None and req.task is not None) else "schema_drift"
    )
    obs: Observation = env.reset(task=task_name)
    return {"observation": obs}


@app.post("/step", response_model=StepResponse)
def step_env(req: ActionRequest):
    """Execute one agent action and return the new observation."""
    obs, reward, done = env.step(req.action)
    return StepResponse(observation=obs, reward=reward, done=done)


@app.get("/state")
def get_state():
    """Return serialised internal state for debugging."""
    return {"state": env.state()}


@app.get("/tasks")
def get_tasks():
    """Enumerate available tasks and their descriptions."""
    return {
        "tasks": {
            "schema_drift": (
                "Easy — An upstream API changed its JSON schema. "
                "Use inspect_schema() to find the mismatch and "
                "apply_mapping_patch() to fix the pipeline."
            ),
            "late_data": (
                "Medium — Trade data is arriving out of order, causing "
                "incorrect moving averages. Implement watermark_reprocess() "
                "to re-align the timeline."
            ),
            "poisoned_records": (
                "Hard — A Data Poisoning event has occurred. Perform "
                "statistical_traceback(), identify the corrupt source, "
                "and quarantine_source() while re-calculating the "
                "10k-row baseline."
            ),
        }
    }


@app.get("/grader")
def run_grader():
    """Return the current grading score for the active episode."""
    return {"score": env.grade()}


@app.get("/baseline")
def run_baseline():
    """Return baseline performance information."""
    return {"status": "success", "baseline_score": 0.85}


# ── Entrypoints ──────────────────────────────────────────────────────────────

def main():
    """CLI entrypoint via pyproject.toml [project.scripts]."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
