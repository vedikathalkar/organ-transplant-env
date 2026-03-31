# 🏥 Organ Transplant Matching & Logistics — OpenEnv

> A real-world AI simulation where an agent acts as a national organ transplant coordinator,
> matching donated organs to waiting patients while respecting medical constraints, blood-type
> compatibility, and organ viability windows.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0-blue)](openenv.yaml)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)

---

## 📋 Table of Contents
- [Environment Description](#environment-description)
- [State Space](#state-space)
- [Action Space](#action-space)
- [Reward Logic](#reward-logic)
- [Tasks](#tasks)
- [API Endpoints](#api-endpoints)
- [Setup Instructions](#setup-instructions)
- [Baseline Agent](#baseline-agent)
- [Example Usage](#example-usage)

---

## 🌍 Environment Description

Every hour, organs become available from donors across India's major cities.
A pool of patients waits on the national transplant registry, each with:
- A required organ type
- A blood type (compatibility must be respected)
- An urgency level (1 = stable, 5 = critical)
- A city location (affects transport time)

The **AI agent** must decide:
1. **Which organ to match to which patient** — respecting medical compatibility
2. **When to defer** — sometimes waiting for a better match is worth the risk
3. **How to manage time** — organs expire; transport takes time; urgency escalates

This is a **real-world operational research problem** used daily by transplant coordinators worldwide.

---

## 📊 State Space

```json
{
  "step_count": 5,
  "patients": [
    {
      "id": "a1b2c3d4",
      "name": "Patient-1234",
      "blood_type": "O+",
      "organ_needed": "Kidney",
      "urgency": 4,
      "wait_days": 320,
      "city": "Mumbai",
      "survival_probability": 0.55,
      "matched": false
    }
  ],
  "organs": [
    {
      "id": "e5f6g7h8",
      "organ_type": "Kidney",
      "blood_type": "O-",
      "donor_city": "Delhi",
      "viability_hours": 28.5,
      "max_viability_hours": 36.0,
      "allocated": false
    }
  ],
  "transports": [
    {
      "id": "t1t2t3t4",
      "organ_id": "e5f6g7h8",
      "patient_id": "a1b2c3d4",
      "origin_city": "Delhi",
      "destination_city": "Mumbai",
      "distance_km": 1148.0,
      "travel_hours": 1.91,
      "status": "in_transit"
    }
  ],
  "successful_transplants": 1,
  "expired_organs": 0,
  "failed_transplants": 0,
  "total_reward": 0.72,
  "done": false
}
```

| Field | Type | Description |
|---|---|---|
| `step_count` | int | Current step (1 step = 1 simulated hour) |
| `patients[].urgency` | int 1–5 | Medical urgency (5 = critical) |
| `patients[].survival_probability` | float 0–1 | Probability of survival without transplant |
| `organs[].viability_hours` | float | Hours until organ is unusable |
| `transports[].status` | enum | `in_transit`, `delivered`, or `failed` |

---

## 🎮 Action Space

Three action types:

### 1. `match` — Dispatch an organ to a patient
```json
{
  "action": "match",
  "organ_id": "e5f6g7h8",
  "patient_id": "a1b2c3d4"
}
```
**Constraints:** organ type must match patient need; blood types must be compatible; organ must have remaining viability.

### 2. `defer` — Explicitly hold an organ
```json
{
  "action": "defer",
  "organ_id": "e5f6g7h8"
}
```
Use when waiting for a better-matched patient. Incurs a small -0.02 penalty.

### 3. `tick` — Advance time by 1 hour
```json
{
  "action": "tick"
}
```
Use when no good match is currently available. Time advances, viability decreases.

---

## 🏆 Reward Logic

All rewards are **partial** — no binary 0/1 outcomes.

### Transplant Success Reward (0.0 – 1.0)
When an organ is successfully delivered and transplanted:

```
reward = 0.40 × urgency_score        # Higher urgency = more reward
       + 0.20 × wait_score            # Longer wait = more reward
       + 0.25 × viability_score       # Fresher organ = more reward
       + 0.15 × criticality_score     # Near-death patients = more reward
```

### Penalties

| Event | Reward |
|---|---|
| Valid match dispatched | +0.05 |
| Blood type incompatible | -0.20 |
| Organ type mismatch | -0.10 |
| Organ expired before match | -0.10 |
| Transport failure (hard mode) | -0.15 |
| Defer action | -0.02 |
| Invalid action | -0.05 |

---

## 📋 Tasks

### 🟢 Easy — Single City Match
- **1 organ**, **5 patients**, all in **Mumbai**
- Full organ viability (100%)
- **10 max steps**
- *Goal: Learn basic blood-type-compatible matching*

### 🟡 Medium — Multi-City Logistics
- **5 organs**, **20 patients**, across **4 cities**
- 80% initial viability
- **50 max steps**
- *Goal: Balance urgency vs. transport time*

### 🔴 Hard — Mass Casualty Cascade
- **20 organs**, **100 patients**, across **8 cities**
- 50% initial viability (organs already in degraded state)
- 3 random **transport failures** mid-simulation
- **200 max steps**
- *Goal: Triage under pressure with cascading failures*

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/reset` | POST | Reset environment for a task |
| `/step` | POST | Submit action, get next state + reward |
| `/state` | GET | Get current state |
| `/tasks` | GET | List all tasks + action/state/reward metadata |
| `/grader` | POST | Score current episode (0.0–1.0) |
| `/baseline` | POST | Run greedy baseline agent |

### Grader Score Breakdown
| Component | Weight |
|---|---|
| Match rate | 45% |
| Average match quality | 30% |
| Organ freshness (anti-expiry) | 15% |
| Transport reliability | 10% |

---

## 🚀 Setup Instructions

### Option 1: Local Development

```bash
# Clone the repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/organ-transplant-env
cd organ-transplant-env

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

# Visit the docs
open http://localhost:7860/docs
```

### Option 2: Docker

```bash
# Build
docker build -t organ-transplant-env .

# Run
docker run -p 7860:7860 organ-transplant-env

# Test
curl http://localhost:7860/
```

### Option 3: Hugging Face Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Push this repo to the Space
4. The Space will automatically build and deploy

---

## 🤖 Baseline Agent

```bash
# Run against local server
python baseline/run_baseline.py

# Run against deployed Space
python baseline/run_baseline.py --host https://YOUR_USERNAME-organ-transplant-env.hf.space

# Run single task
python baseline/run_baseline.py --task medium
```

**Expected baseline scores (seed=42):**

| Task | Expected Score |
|---|---|
| Easy | ~0.65 – 0.80 |
| Medium | ~0.45 – 0.60 |
| Hard | ~0.25 – 0.45 |

---

## 💡 Example Usage (Python)

```python
import httpx

BASE = "http://localhost:7860"

with httpx.Client() as client:
    # 1. Reset
    state = client.post(f"{BASE}/reset", json={"task_id": "easy", "seed": 42}).json()["state"]

    # 2. Find a compatible match
    organ = next(o for o in state["organs"] if not o["allocated"])
    patient = next(p for p in state["patients"] if p["organ_needed"] == organ["organ_type"])

    # 3. Match
    result = client.post(f"{BASE}/step", json={
        "task_id": "easy",
        "action": {"action": "match", "organ_id": organ["id"], "patient_id": patient["id"]}
    }).json()

    print(f"Reward: {result['reward']}")
    print(f"Done: {result['done']}")

    # 4. Grade
    score = client.post(f"{BASE}/grader", json={"task_id": "easy"}).json()
    print(f"Score: {score['score']}")
```

---

## 🗂️ Project Structure

```
organ-transplant-env/
├── app/
│   ├── main.py            # FastAPI server + all endpoints
│   └── environment.py     # Core simulation logic
├── baseline/
│   └── run_baseline.py    # Greedy baseline agent
├── openenv.yaml           # OpenEnv specification
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🏥 Real-World Relevance

This simulation is based on how national transplant registries like **NOTTO (India)**, **UNOS (USA)**, and **Eurotransplant** actually operate:
- Blood type compatibility is medically mandatory
- Organ viability windows are hard biological constraints
- Transport logistics (air ambulances) are real bottlenecks
- Urgency scoring (MELD score for liver, etc.) guides prioritization

---

*Built for OpenEnv Hackathon — Real-world AI simulation track*
