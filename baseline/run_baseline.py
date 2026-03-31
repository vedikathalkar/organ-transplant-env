"""
Baseline Agent — Organ Transplant Matching & Logistics
=======================================================
Greedy agent: always matches the highest-urgency compatible (organ, patient) pair.
Runs all three tasks and prints reproducible scores.

Usage:
    # Against local server
    python baseline/run_baseline.py

    # Against deployed HF Space
    python baseline/run_baseline.py --host https://your-space.hf.space
"""

import argparse
import json
import sys
import time
import httpx

DEFAULT_HOST = "http://localhost:7860"
TASKS = ["easy", "medium", "hard"]
SEED = 42


def reset(client: httpx.Client, host: str, task_id: str) -> dict:
    r = client.post(f"{host}/reset", json={"task_id": task_id, "seed": SEED})
    r.raise_for_status()
    return r.json()["state"]


def step(client: httpx.Client, host: str, task_id: str, action: dict) -> dict:
    r = client.post(f"{host}/step", json={"task_id": task_id, "action": action})
    r.raise_for_status()
    return r.json()


def grade(client: httpx.Client, host: str, task_id: str) -> dict:
    r = client.post(f"{host}/grader", json={"task_id": task_id})
    r.raise_for_status()
    return r.json()


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


def greedy_action(state: dict) -> dict:
    """Pick the best compatible match by urgency × wait_days, else tick."""
    organs = [o for o in state["organs"] if not o["allocated"] and o["viability_hours"] > 0]
    patients = [p for p in state["patients"] if not p["matched"]]

    best = None
    best_score = -1

    for organ in organs:
        for patient in patients:
            if organ["organ_type"] != patient["organ_needed"]:
                continue
            if not is_compatible(organ["blood_type"], patient["blood_type"]):
                continue
            score = patient["urgency"] * 10 + patient["wait_days"] / 100
            if score > best_score:
                best_score = score
                best = {"action": "match", "organ_id": organ["id"], "patient_id": patient["id"]}

    return best if best else {"action": "tick"}


def run_task(client: httpx.Client, host: str, task_id: str) -> dict:
    print(f"\n{'='*55}")
    print(f"  Task: {task_id.upper()}")
    print(f"{'='*55}")

    state = reset(client, host, task_id)
    print(f"  Organs   : {len(state['organs'])}")
    print(f"  Patients : {len(state['patients'])}")

    done = False
    total_reward = 0.0
    steps = 0
    start = time.time()

    while not done:
        action = greedy_action(state)
        result = step(client, host, task_id, action)
        state = result["state"]
        total_reward += result["reward"]
        done = result["done"]
        steps += 1

        if steps % 20 == 0:
            print(f"  Step {steps:>4} | reward so far: {total_reward:+.4f} | transplants: {state['successful_transplants']}")

    elapsed = time.time() - start
    graded = grade(client, host, task_id)

    print(f"\n  ✅ Done in {steps} steps ({elapsed:.1f}s)")
    print(f"  Total reward         : {total_reward:+.4f}")
    print(f"  Successful transplants: {state['successful_transplants']}")
    print(f"  Expired organs       : {state['expired_organs']}")
    print(f"  Failed transports    : {state['failed_transplants']}")
    print(f"  🏆 SCORE             : {graded['score']:.4f}")
    print(f"  Breakdown            : {json.dumps(graded['breakdown'], indent=4)}")

    return {
        "task_id": task_id,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "score": graded["score"],
        "breakdown": graded["breakdown"],
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline agent for Organ Transplant Matching Env")
    parser.add_argument("--host", default=DEFAULT_HOST, help="API base URL")
    parser.add_argument("--task", default="all", choices=["all", "easy", "medium", "hard"])
    args = parser.parse_args()

    print("\n🏥 Organ Transplant Matching — Greedy Baseline Agent")
    print(f"   Host : {args.host}")
    print(f"   Seed : {SEED}")

    tasks_to_run = TASKS if args.task == "all" else [args.task]
    results = []

    with httpx.Client(timeout=120.0) as client:
        # Health check
        try:
            r = client.get(f"{args.host}/")
            r.raise_for_status()
            print(f"\n   Server OK: {r.json()['status']}")
        except Exception as e:
            print(f"\n❌ Cannot reach server at {args.host}: {e}")
            sys.exit(1)

        for task_id in tasks_to_run:
            result = run_task(client, args.host, task_id)
            results.append(result)

    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    for r in results:
        bar = "█" * int(r["score"] * 20)
        print(f"  {r['task_id']:8} | {bar:<20} | {r['score']:.4f}")
    print()


if __name__ == "__main__":
    main()
