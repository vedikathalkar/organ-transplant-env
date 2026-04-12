"""
inference.py — LLM-powered agent for Organ Transplant Matching Environment
Follows exact [START]/[STEP]/[END] format required by validator.
"""

import os
import sys
import json
import httpx
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_HOST     = os.environ.get("ENV_HOST",     "http://localhost:7860")

TASKS = ["easy", "medium", "hard"]
SEED  = 42

BLOOD_COMPATIBILITY = {
    "O-":  ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"],
    "O+":  ["O+", "A+", "B+", "AB+"],
    "A-":  ["A-", "A+", "AB-", "AB+"],
    "A+":  ["A+", "AB+"],
    "B-":  ["B-", "B+", "AB-", "AB+"],
    "B+":  ["B+", "AB+"],
    "AB-": ["AB-", "AB+"],
    "AB+": ["AB+"],
}

def is_compatible(donor: str, recipient: str) -> bool:
    return recipient in BLOOD_COMPATIBILITY.get(donor, [])

def get_client() -> OpenAI:
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN if HF_TOKEN else "dummy",
    )

def greedy_action(state: dict) -> dict:
    """Greedy fallback: pick highest urgency compatible match."""
    organs   = [o for o in state["organs"]   if not o["allocated"] and o["viability_hours"] > 0]
    patients = [p for p in state["patients"] if not p["matched"]]
    best = None
    best_score = -1
    for o in organs:
        for p in patients:
            if o["organ_type"] != p["organ_needed"]:
                continue
            if not is_compatible(o["blood_type"], p["blood_type"]):
                continue
            score = p["urgency"] * 10 + p["wait_days"] / 100
            if score > best_score:
                best_score = score
                best = {"action": "match", "organ_id": o["id"], "patient_id": p["id"]}
    return best if best else {"action": "tick"}

def llm_action(llm: OpenAI, state: dict) -> dict:
    """Try LLM, fall back to greedy on any error."""
    organs   = [o for o in state["organs"]   if not o["allocated"] and o["viability_hours"] > 0]
    patients = [p for p in state["patients"] if not p["matched"]]
    valid_pairs = []
    for o in organs:
        for p in patients:
            if o["organ_type"] == p["organ_needed"] and is_compatible(o["blood_type"], p["blood_type"]):
                valid_pairs.append({
                    "organ_id":   o["id"],
                    "patient_id": p["id"],
                    "urgency":    p["urgency"],
                    "wait_days":  p["wait_days"],
                    "viability":  round(o["viability_hours"], 1),
                })
    if not valid_pairs:
        return {"action": "tick"}
    try:
        prompt = f"""Pick the best organ-patient match. Respond with JSON only.
Valid pairs: {json.dumps(valid_pairs[:5])}
Format: {{"action": "match", "organ_id": "...", "patient_id": "..."}}"""
        resp = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Medical AI. JSON only. No explanation."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=80,
            temperature=0.0,
        )
        raw    = resp.choices[0].message.content.strip()
        raw    = raw.replace("```json", "").replace("```", "").strip()
        action = json.loads(raw)
        if action.get("action") == "match":
            valid = any(
                p["organ_id"] == action.get("organ_id") and
                p["patient_id"] == action.get("patient_id")
                for p in valid_pairs
            )
            return action if valid else greedy_action(state)
        return action if action.get("action") in ["defer", "tick"] else greedy_action(state)
    except Exception:
        return greedy_action(state)


def run_task(llm: OpenAI, client: httpx.Client, task_id: str) -> dict:
    # ── Reset ──────────────────────────────────────────────────────────────
    r = client.post(f"{ENV_HOST}/reset", json={"task_id": task_id, "seed": SEED})
    r.raise_for_status()
    state = r.json()["state"]

    print(f"[START] task={task_id}", flush=True)

    done         = False
    total_reward = 0.0
    step_num     = 0

    # ── Episode loop ───────────────────────────────────────────────────────
    while not done:
        action = llm_action(llm, state)

        r = client.post(f"{ENV_HOST}/step", json={"task_id": task_id, "action": action})
        r.raise_for_status()
        result = r.json()

        state        = result["state"]
        reward       = float(result["reward"])
        done         = result["done"]
        total_reward += reward
        step_num     += 1

        print(f"[STEP] step={step_num} action={action['action']} reward={reward:.4f}", flush=True)

    # ── Grade ──────────────────────────────────────────────────────────────
    r = client.post(f"{ENV_HOST}/grader", json={"task_id": task_id})
    r.raise_for_status()
    grade = r.json()

    # Clamp score strictly between 0 and 1 exclusive
    score = float(grade["score"])
    score = round(max(0.001, min(0.999, score)), 4)

    print(f"[END] task={task_id} score={score:.4f} steps={step_num} reward={total_reward:.4f}", flush=True)

    return {
        "task_id":      task_id,
        "score":        score,
        "steps":        step_num,
        "total_reward": round(total_reward, 4),
    }


def main():
    print("[START] inference", flush=True)
    print(f"[STEP] setup API_BASE_URL={API_BASE_URL} MODEL_NAME={MODEL_NAME}", flush=True)

    llm = get_client()

    with httpx.Client(timeout=300.0) as client:
        # Health check
        try:
            r = client.get(f"{ENV_HOST}/")
            r.raise_for_status()
            print(f"[STEP] health_check status={r.json()['status']}", flush=True)
        except Exception as e:
            print(f"[STEP] health_check_failed error={e}", flush=True)
            sys.exit(1)

        results = []
        for task_id in TASKS:
            result = run_task(llm, client, task_id)
            results.append(result)

    # Save results
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"[END] inference tasks={len(results)} done=true", flush=True)


if __name__ == "__main__":
    main()