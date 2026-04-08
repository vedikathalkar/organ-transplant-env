"""
inference.py — LLM-powered agent for Organ Transplant Matching Environment
"""

import os
import sys
import json
import httpx
from openai import OpenAI

# ── Config from environment variables ────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_HOST     = os.environ.get("ENV_HOST",     "http://localhost:7860")

TASKS  = ["easy", "medium", "hard"]
SEED   = 42

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

def env_reset(client: httpx.Client, task_id: str) -> dict:
    r = client.post(f"{ENV_HOST}/reset", json={"task_id": task_id, "seed": SEED})
    r.raise_for_status()
    return r.json()["state"]

def env_step(client: httpx.Client, task_id: str, action: dict) -> dict:
    r = client.post(f"{ENV_HOST}/step", json={"task_id": task_id, "action": action})
    r.raise_for_status()
    return r.json()

def env_grade(client: httpx.Client, task_id: str) -> dict:
    r = client.post(f"{ENV_HOST}/grader", json={"task_id": task_id})
    r.raise_for_status()
    return r.json()

def build_prompt(state: dict):
    organs = [o for o in state["organs"] if not o["allocated"] and o["viability_hours"] > 0]
    patients = [p for p in state["patients"] if not p["matched"]]
    valid_pairs = []
    for o in organs:
        for p in patients:
            if o["organ_type"] == p["organ_needed"] and is_compatible(o["blood_type"], p["blood_type"]):
                valid_pairs.append({
                    "organ_id":      o["id"],
                    "patient_id":    p["id"],
                    "organ_type":    o["organ_type"],
                    "viability_h":   round(o["viability_hours"], 1),
                    "urgency":       p["urgency"],
                    "wait_days":     p["wait_days"],
                })
    prompt = f"""You are an AI organ transplant coordinator.
Match organs to patients to maximise lives saved.

VALID COMPATIBLE PAIRS:
{json.dumps(valid_pairs[:10], indent=2)}

Respond ONLY with a single valid JSON object. No explanation. No markdown.
Example: {{"action": "match", "organ_id": "abc12345", "patient_id": "xyz67890"}}
"""
    return prompt, valid_pairs

def greedy_fallback(valid_pairs: list) -> dict:
    if not valid_pairs:
        return {"action": "tick"}
    best = max(valid_pairs, key=lambda p: p["urgency"] * 10 + p["wait_days"] / 100)
    return {"action": "match", "organ_id": best["organ_id"], "patient_id": best["patient_id"]}

def llm_action(llm: OpenAI, state: dict) -> dict:
    prompt, valid_pairs = build_prompt(state)
    if not valid_pairs:
        return {"action": "tick"}
    try:
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a medical AI. Respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        action = json.loads(raw)
        if action.get("action") == "match":
            valid = any(
                p["organ_id"] == action.get("organ_id") and p["patient_id"] == action.get("patient_id")
                for p in valid_pairs
            )
            return action if valid else greedy_fallback(valid_pairs)
        elif action.get("action") in ["defer", "tick"]:
            return action
        else:
            return greedy_fallback(valid_pairs)
    except Exception as e:
        print(f"WARNING: LLM error: {e} — using greedy fallback", flush=True)
        return greedy_fallback(valid_pairs)

def run_task(llm: OpenAI, env_client: httpx.Client, task_id: str) -> dict:
    print(f"[START] task={task_id}", flush=True)

    state = env_reset(env_client, task_id)
    done         = False
    total_reward = 0.0
    steps        = 0

    while not done:
        action = llm_action(llm, state)
        result       = env_step(env_client, task_id, action)
        state        = result["state"]
        total_reward += result["reward"]
        done         = result["done"]
        steps        += 1
        print(f"[STEP] step={steps} reward={result['reward']:.4f}", flush=True)

    grade = env_grade(env_client, task_id)
    score = grade["score"]

    print(f"[END] task={task_id} score={score:.4f} steps={steps}", flush=True)

    return {
        "task_id":      task_id,
        "steps":        steps,
        "total_reward": round(total_reward, 4),
        "score":        score,
        "breakdown":    grade["breakdown"],
    }

def main():
    print("[START] inference", flush=True)
    print(f"API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"ENV_HOST={ENV_HOST}", flush=True)

    llm = get_client()
    results = []

    with httpx.Client(timeout=120.0) as env_client:
        try:
            r = env_client.get(f"{ENV_HOST}/")
            r.raise_for_status()
            print(f"Environment OK: {r.json()['status']}", flush=True)
        except Exception as e:
            print(f"ERROR: Cannot reach environment at {ENV_HOST}: {e}", flush=True)
            sys.exit(1)

        for task_id in TASKS:
            result = run_task(llm, env_client, task_id)
            results.append(result)

    print("\nFINAL SUMMARY", flush=True)
    for r in results:
        print(f"[STEP] task={r['task_id']} score={r['score']:.4f}", flush=True)

    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[END] inference done", flush=True)

if __name__ == "__main__":
    main()