"""
inference.py — LLM-powered agent for Organ Transplant Matching Environment
===========================================================================
Uses OpenAI-compatible client with:
  - API_BASE_URL : LLM API endpoint
  - MODEL_NAME   : model identifier
  - HF_TOKEN     : Hugging Face / API key

The LLM agent reads the environment state and decides the best action.
Runtime: < 20 minutes for all 3 tasks.

Usage:
    export API_BASE_URL="https://api-inference.huggingface.co/v1"
    export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
    export HF_TOKEN="hf_your_token_here"
    python inference.py
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

# Blood compatibility table (used locally to pre-filter for the LLM prompt)
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


# ── OpenAI client ─────────────────────────────────────────────────────────────
def get_client() -> OpenAI:
    if not HF_TOKEN:
        print("⚠️  HF_TOKEN not set — using unauthenticated (may fail)")
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN if HF_TOKEN else "dummy",
    )


# ── Environment API helpers ───────────────────────────────────────────────────
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


# ── Build a compact state summary for the LLM ────────────────────────────────
def build_prompt(state: dict) -> str:
    organs = [o for o in state["organs"] if not o["allocated"] and o["viability_hours"] > 0]
    patients = [p for p in state["patients"] if not p["matched"]]

    # Find valid (organ, patient) pairs
    valid_pairs = []
    for o in organs:
        for p in patients:
            if o["organ_type"] == p["organ_needed"] and is_compatible(o["blood_type"], p["blood_type"]):
                valid_pairs.append({
                    "organ_id":    o["id"],
                    "patient_id":  p["id"],
                    "organ_type":  o["organ_type"],
                    "organ_blood": o["blood_type"],
                    "donor_city":  o["donor_city"],
                    "viability_h": round(o["viability_hours"], 1),
                    "patient_blood":   p["blood_type"],
                    "patient_city":    p["city"],
                    "urgency":         p["urgency"],
                    "wait_days":       p["wait_days"],
                    "survival_prob":   p["survival_probability"],
                })

    prompt = f"""You are an AI organ transplant coordinator.
Your goal: match organs to patients to maximise lives saved.

CURRENT STATE:
- Step: {state['step_count']}
- Available organs (unallocated, viable): {len(organs)}
- Waiting patients (unmatched): {len(patients)}
- Successful transplants so far: {state['successful_transplants']}
- Expired organs so far: {state['expired_organs']}

VALID COMPATIBLE PAIRS (organ_type matches, blood compatible):
{json.dumps(valid_pairs[:10], indent=2)}

RULES:
- Organs expire if viability_hours reaches 0 — act fast on low viability organs
- Higher urgency (5=critical) patients should be prioritised
- Longer wait_days = higher priority if urgency is equal
- Lower survival_probability = more critical patient

AVAILABLE ACTIONS:
1. match: {{"action": "match", "organ_id": "...", "patient_id": "..."}}
2. defer: {{"action": "defer", "organ_id": "..."}}
3. tick:  {{"action": "tick"}}

If there are valid pairs, always choose match.
Pick the best pair: prioritise lowest viability organ + highest urgency patient.
If no valid pairs exist, choose tick.

Respond ONLY with a single valid JSON object. No explanation. No markdown. Just JSON.
Example: {{"action": "match", "organ_id": "abc12345", "patient_id": "xyz67890"}}
"""
    return prompt, valid_pairs


# ── Ask LLM for an action ─────────────────────────────────────────────────────
def llm_action(llm: OpenAI, state: dict) -> dict:
    prompt, valid_pairs = build_prompt(state)

    # If no valid pairs at all, skip LLM call and tick
    if not valid_pairs:
        return {"action": "tick"}

    try:
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical AI coordinator. Always respond with valid JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=100,
            temperature=0.1,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()

        action = json.loads(raw)

        # Validate the action
        if action.get("action") == "match":
            organ_id   = action.get("organ_id", "")
            patient_id = action.get("patient_id", "")
            # Check it's actually a valid pair
            valid = any(
                p["organ_id"] == organ_id and p["patient_id"] == patient_id
                for p in valid_pairs
            )
            if valid:
                return action
            else:
                # LLM hallucinated IDs — fall back to best greedy pair
                print("    ⚠️  LLM returned invalid IDs, using greedy fallback")
                return greedy_fallback(valid_pairs)

        elif action.get("action") in ["defer", "tick"]:
            return action

        else:
            return greedy_fallback(valid_pairs)

    except Exception as e:
        print(f"    ⚠️  LLM error: {e} — using greedy fallback")
        return greedy_fallback(valid_pairs)


def greedy_fallback(valid_pairs: list) -> dict:
    """Pick best pair by urgency + wait_days if LLM fails."""
    if not valid_pairs:
        return {"action": "tick"}
    best = max(valid_pairs, key=lambda p: p["urgency"] * 10 + p["wait_days"] / 100)
    return {"action": "match", "organ_id": best["organ_id"], "patient_id": best["patient_id"]}


# ── Run one task ──────────────────────────────────────────────────────────────
def run_task(llm: OpenAI, env_client: httpx.Client, task_id: str) -> dict:
    print(f"\n{'='*55}")
    print(f"  Task: {task_id.upper()}")
    print(f"{'='*55}")

    state = env_reset(env_client, task_id)
    print(f"  Organs   : {len(state['organs'])}")
    print(f"  Patients : {len(state['patients'])}")

    done         = False
    total_reward = 0.0
    steps        = 0

    while not done:
        action = llm_action(llm, state)
        print(f"  Step {steps+1:>3} | action={json.dumps(action)}")

        result       = env_step(env_client, task_id, action)
        state        = result["state"]
        total_reward += result["reward"]
        done         = result["done"]
        steps        += 1

        if result["reward"] > 0.1:
            print(f"           ✅ reward={result['reward']:+.4f} | transplants={state['successful_transplants']}")

    grade = env_grade(env_client, task_id)
    score = grade["score"]
    b     = grade["breakdown"]

    print(f"\n  ✅ Done in {steps} steps")
    print(f"  Total reward          : {total_reward:+.4f}")
    print(f"  Successful transplants: {b['successful_transplants']}")
    print(f"  Expired organs        : {b['expired_organs']}")
    print(f"  🏆 SCORE              : {score:.4f}")

    return {
        "task_id":      task_id,
        "steps":        steps,
        "total_reward": round(total_reward, 4),
        "score":        score,
        "breakdown":    b,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n🏥 Organ Transplant Matching — LLM Inference Agent")
    print(f"   API_BASE_URL : {API_BASE_URL}")
    print(f"   MODEL_NAME   : {MODEL_NAME}")
    print(f"   ENV_HOST     : {ENV_HOST}")
    print(f"   HF_TOKEN     : {'set ✅' if HF_TOKEN else 'NOT SET ⚠️'}")
    print(f"   Seed         : {SEED}")

    # Initialise clients
    llm = get_client()

    results = []
    with httpx.Client(timeout=120.0) as env_client:

        # Health check
        try:
            r = env_client.get(f"{ENV_HOST}/")
            r.raise_for_status()
            print(f"\n   Environment OK: {r.json()['status']}")
        except Exception as e:
            print(f"\n❌ Cannot reach environment at {ENV_HOST}: {e}")
            sys.exit(1)

        # Run all tasks
        for task_id in TASKS:
            result = run_task(llm, env_client, task_id)
            results.append(result)

    # Final summary
    print(f"\n{'='*55}")
    print("  FINAL SUMMARY")
    print(f"{'='*55}")
    for r in results:
        bar   = "█" * int(r["score"] * 20)
        print(f"  {r['task_id']:8} | {bar:<20} | {r['score']:.4f}")

    print(f"\n  Scores: { {r['task_id']: r['score'] for r in results} }")
    print()

    # Write results to file for validator
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Results saved to inference_results.json ✅")


if __name__ == "__main__":
    main()
