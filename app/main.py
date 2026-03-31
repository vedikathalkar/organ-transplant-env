"""
Organ Transplant Matching & Logistics — OpenEnv API
FastAPI server exposing all required OpenEnv endpoints.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional

from environment import OrganTransplantEnv, _is_compatible

app = FastAPI(
    title="Organ Transplant Matching Environment",
    description="OpenEnv-compliant real-world simulation for organ transplant matching and logistics.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global env store ──────────────────────────────────────────────────────────
_envs: Dict[str, OrganTransplantEnv] = {}


def _get_env(task_id: str) -> OrganTransplantEnv:
    if task_id not in _envs:
        raise HTTPException(
            status_code=400,
            detail=f"No active environment for task '{task_id}'. Call /reset first."
        )
    return _envs[task_id]


# ── Request / Response models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42

class StepRequest(BaseModel):
    task_id: str = "easy"
    action: Dict[str, Any]

class GradeRequest(BaseModel):
    task_id: str = "easy"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    return {
        "status": "ok",
        "environment": "Organ Transplant Matching & Logistics",
        "version": "1.0.0"
    }


@app.post("/reset", summary="Reset the environment")
def reset(req: ResetRequest):
    """Reset (or initialise) the environment for the given task."""
    if req.task_id not in OrganTransplantEnv.TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{req.task_id}'. Choose from: easy, medium, hard"
        )
    env = OrganTransplantEnv(task_id=req.task_id, seed=req.seed)
    _envs[req.task_id] = env
    return {"task_id": req.task_id, "state": env.reset()}


@app.post("/step", summary="Take one action step")
def step(req: StepRequest):
    """
    Submit an action and receive (next_state, reward, done, info).

    Action formats:
    - {"action": "match", "organ_id": "...", "patient_id": "..."}
    - {"action": "defer", "organ_id": "..."}
    - {"action": "tick"}
    """
    env = _get_env(req.task_id)
    result = env.step(req.action)
    return result


@app.get("/state", summary="Get current state")
def state(task_id: str = "easy"):
    """Return the current environment state without advancing time."""
    env = _get_env(task_id)
    return env.state()


@app.get("/tasks", summary="List all available tasks")
def tasks():
    """Return metadata for all three tasks (easy / medium / hard)."""
    return {
        "tasks": list(OrganTransplantEnv.TASKS.values()),
        "action_space": {
            "match": {
                "description": "Match an organ to a patient and dispatch transport",
                "fields": {"action": "match", "organ_id": "string", "patient_id": "string"},
            },
            "defer": {
                "description": "Defer organ allocation (small penalty)",
                "fields": {"action": "defer", "organ_id": "string"},
            },
            "tick": {
                "description": "Advance simulation time by 1 hour without matching",
                "fields": {"action": "tick"},
            },
        },
        "state_space": {
            "patients": "List of patients with blood_type, organ_needed, urgency (1-5), wait_days, city, survival_probability",
            "organs": "List of available organs with blood_type, organ_type, donor_city, viability_hours",
            "transports": "In-transit organ shipments with ETA and status",
            "step_count": "Current simulation step",
            "successful_transplants": "Count of completed transplants",
            "expired_organs": "Count of organs that expired before allocation",
        },
        "reward_logic": {
            "successful_transplant": "0.0-1.0 based on urgency (40%), wait time (20%), viability (25%), criticality (15%)",
            "valid_match_dispatched": "+0.05",
            "blood_type_incompatible": "-0.20",
            "organ_type_mismatch": "-0.10",
            "organ_expired": "-0.10",
            "transport_failure": "-0.15",
            "defer_action": "-0.02",
            "invalid_action": "-0.05",
        },
    }


@app.post("/grader", summary="Grade current episode")
def grader(req: GradeRequest):
    """Compute a score between 0.0 and 1.0 for the current episode."""
    env = _get_env(req.task_id)
    result = env.grade()
    return result


@app.post("/baseline", summary="Run baseline agent and return score")
def baseline(req: ResetRequest):
    """Run the built-in greedy baseline agent. Returns reproducible results."""
    env = OrganTransplantEnv(task_id=req.task_id, seed=req.seed)
    env.reset()

    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        s = env.state()

        best_action = None
        best_priority = -1

        available_organs = [o for o in s["organs"] if not o["allocated"] and o["viability_hours"] > 0]
        unmatched_patients = [p for p in s["patients"] if not p["matched"]]

        for organ in available_organs:
            for patient in unmatched_patients:
                if organ["organ_type"] != patient["organ_needed"]:
                    continue
                if not _is_compatible(organ["blood_type"], patient["blood_type"]):
                    continue
                priority = patient["urgency"] * 10 + patient["wait_days"] / 100
                if priority > best_priority:
                    best_priority = priority
                    best_action = {
                        "action": "match",
                        "organ_id": organ["id"],
                        "patient_id": patient["id"],
                    }

        if best_action is None:
            best_action = {"action": "tick"}

        result = env.step(best_action)
        total_reward += result["reward"]
        done = result["done"]
        steps += 1

    grade = env.grade()
    return {
        "agent": "greedy_baseline",
        "task_id": req.task_id,
        "seed": req.seed,
        "total_steps": steps,
        "total_reward": round(total_reward, 4),
        "grade": grade,
    }
