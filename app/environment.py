"""
Organ Transplant Matching & Logistics Environment
OpenEnv-compliant real-world simulation
"""

import random
import math
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


# ── Constants ────────────────────────────────────────────────────────────────

BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

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

CITIES = {
    "Mumbai":    (19.076, 72.877),
    "Delhi":     (28.704, 77.102),
    "Bangalore": (12.971, 77.594),
    "Chennai":   (13.082, 80.270),
    "Kolkata":   (22.572, 88.363),
    "Hyderabad": (17.385, 78.487),
    "Pune":      (18.520, 73.856),
    "Ahmedabad": (23.022, 72.571),
}

ORGAN_VIABILITY_HOURS = {
    "Heart":   6,
    "Lungs":   6,
    "Liver":   24,
    "Kidney":  36,
    "Pancreas": 24,
}

AVG_TRANSPORT_SPEED_KMH = 600  # air transport


# ── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class Patient:
    id: str
    name: str
    blood_type: str
    organ_needed: str
    urgency: int          # 1 (low) – 5 (critical)
    wait_days: int
    city: str
    survival_probability: float   # 0.0–1.0 without transplant
    matched: bool = False
    match_score: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class Organ:
    id: str
    organ_type: str
    blood_type: str
    donor_city: str
    viability_hours: float        # hours remaining
    max_viability_hours: float
    allocated: bool = False
    allocated_to: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Transport:
    id: str
    organ_id: str
    patient_id: str
    origin_city: str
    destination_city: str
    distance_km: float
    travel_hours: float
    status: str = "in_transit"    # in_transit | delivered | failed

    def to_dict(self):
        return asdict(self)


@dataclass
class EnvironmentState:
    step_count: int
    patients: List[Patient]
    organs: List[Organ]
    transports: List[Transport]
    successful_transplants: int
    failed_transplants: int
    expired_organs: int
    total_reward: float
    task_id: str
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "step_count": self.step_count,
            "patients": [p.to_dict() for p in self.patients],
            "organs": [o.to_dict() for o in self.organs],
            "transports": [t.to_dict() for t in self.transports],
            "successful_transplants": self.successful_transplants,
            "failed_transplants": self.failed_transplants,
            "expired_organs": self.expired_organs,
            "total_reward": round(self.total_reward, 4),
            "task_id": self.task_id,
            "done": self.done,
            "info": self.info,
        }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _haversine_km(city1: str, city2: str) -> float:
    lat1, lon1 = CITIES[city1]
    lat2, lon2 = CITIES[city2]
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def _is_compatible(donor_blood: str, recipient_blood: str) -> bool:
    return recipient_blood in BLOOD_COMPATIBILITY.get(donor_blood, [])


def _make_patient(organ_type: str, urgency: Optional[int] = None, city: Optional[str] = None) -> Patient:
    blood = random.choice(BLOOD_TYPES)
    urg = urgency if urgency else random.randint(1, 5)
    c = city if city else random.choice(list(CITIES.keys()))
    wait = random.randint(10, 500)
    survival = max(0.1, 0.95 - (urg * 0.1) - (wait / 2000))
    return Patient(
        id=str(uuid.uuid4())[:8],
        name=f"Patient-{random.randint(1000,9999)}",
        blood_type=blood,
        organ_needed=organ_type,
        urgency=urg,
        wait_days=wait,
        city=c,
        survival_probability=round(survival, 3),
    )


def _make_organ(organ_type: str, viability_fraction: float = 1.0, city: Optional[str] = None) -> Organ:
    blood = random.choice(BLOOD_TYPES)
    c = city if city else random.choice(list(CITIES.keys()))
    max_v = ORGAN_VIABILITY_HOURS[organ_type]
    via = max_v * viability_fraction
    return Organ(
        id=str(uuid.uuid4())[:8],
        organ_type=organ_type,
        blood_type=blood,
        donor_city=c,
        viability_hours=round(via, 2),
        max_viability_hours=max_v,
    )


# ── Main Environment ──────────────────────────────────────────────────────────

class OrganTransplantEnv:
    """
    OpenEnv-compliant Organ Transplant Matching & Logistics Environment.

    Actions:
        {
          "action": "match",
          "organ_id": "<id>",
          "patient_id": "<id>"
        }
        OR
        {
          "action": "defer",
          "organ_id": "<id>"
        }
        OR
        {
          "action": "tick"   # advance time by 1 hour
        }
    """

    TASKS = {
        "easy": {
            "id": "easy",
            "name": "Single City Match",
            "description": (
                "Match 1 organ to the best patient in a single city. "
                "All patients and donor are in Mumbai. Blood compatibility must be respected."
            ),
            "n_organs": 1,
            "n_patients": 5,
            "n_cities": 1,
            "max_steps": 10,
            "viability_fraction": 1.0,
            "difficulty": "easy",
        },
        "medium": {
            "id": "medium",
            "name": "Multi-City Logistics",
            "description": (
                "Match 5 organs to 20 patients spread across 4 cities. "
                "Transport time reduces organ viability. Prioritize urgency and compatibility."
            ),
            "n_organs": 5,
            "n_patients": 20,
            "n_cities": 4,
            "max_steps": 50,
            "viability_fraction": 0.8,
            "difficulty": "medium",
        },
        "hard": {
            "id": "hard",
            "name": "Mass Casualty Cascade",
            "description": (
                "Match 20 organs to 100 patients across all 8 cities under time pressure. "
                "Organs have reduced viability, urgency levels are critical, "
                "and 3 random transport failures will occur mid-simulation."
            ),
            "n_organs": 20,
            "n_patients": 100,
            "n_cities": 8,
            "max_steps": 200,
            "viability_fraction": 0.5,
            "difficulty": "hard",
        },
    }

    def __init__(self, task_id: str = "easy", seed: int = 42):
        self.task_id = task_id
        self.seed = seed
        self._task_cfg = self.TASKS[task_id]
        self._state: Optional[EnvironmentState] = None
        self._rng = random.Random(seed)
        self._transport_failures_remaining = 3 if task_id == "hard" else 0
        self.reset()

    # ── OpenEnv API ────────────────────────────────────────────────────────

    def reset(self) -> Dict:
        random.seed(self.seed)
        cfg = self._task_cfg
        cities_pool = list(CITIES.keys())[:cfg["n_cities"]]

        organ_types = list(ORGAN_VIABILITY_HOURS.keys())

        organs = []
        for _ in range(cfg["n_organs"]):
            organ_type = random.choice(organ_types)
            city = random.choice(cities_pool)
            organs.append(_make_organ(organ_type, cfg["viability_fraction"], city=city))

        patients = []
        # Guarantee at least one compatible patient per organ so the task is solvable
        for organ in organs:
            # Pick a blood type that is compatible with this organ's donor blood type
            compatible_recipient_types = BLOOD_COMPATIBILITY[organ.blood_type]
            recipient_blood = random.choice(compatible_recipient_types)
            city = random.choice(cities_pool)
            p = _make_patient(organ.organ_type, city=city)
            p.blood_type = recipient_blood   # override to guarantee compatibility
            patients.append(p)

        # Fill remaining patient slots randomly
        for _ in range(cfg["n_patients"] - cfg["n_organs"]):
            organ_needed = random.choice(organ_types)
            city = random.choice(cities_pool)
            patients.append(_make_patient(organ_needed, city=city))

        random.shuffle(patients)  # don't give away the guaranteed matches

        self._state = EnvironmentState(
            step_count=0,
            patients=patients,
            organs=organs,
            transports=[],
            successful_transplants=0,
            failed_transplants=0,
            expired_organs=0,
            total_reward=0.0,
            task_id=self.task_id,
            done=False,
            info={"message": "Environment reset. Ready for actions."},
        )
        self._transport_failures_remaining = 3 if self.task_id == "hard" else 0
        return self._state.to_dict()

    def step(self, action: Dict) -> Dict:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        if self._state.done:
            return {
                "state": self._state.to_dict(),
                "reward": 0.0,
                "done": True,
                "info": {"message": "Episode already done. Call reset()."},
            }

        reward = 0.0
        info = {}
        s = self._state

        action_type = action.get("action", "tick")

        # ── Advance viability on every step ───────────────────────────────
        hours_elapsed = 1.0  # each step = 1 simulated hour
        expired_this_step = []
        for organ in s.organs:
            if not organ.allocated:
                was_alive = organ.viability_hours > 0.0
                organ.viability_hours = max(0.0, organ.viability_hours - hours_elapsed)
                # Only count expiry once — the first tick it reaches 0
                if was_alive and organ.viability_hours == 0.0:
                    expired_this_step.append(organ.id)

        for oid in expired_this_step:
            s.expired_organs += 1
            reward -= 0.1
            info[f"expired_{oid}"] = "Organ expired before allocation"

        # ── Update in-transit organs ───────────────────────────────────────
        completed_transports = []
        for transport in s.transports:
            if transport.status == "in_transit":
                transport.travel_hours -= hours_elapsed
                # Random transport failure (hard mode)
                if self._transport_failures_remaining > 0 and random.random() < 0.02:
                    transport.status = "failed"
                    self._transport_failures_remaining -= 1
                    s.failed_transplants += 1
                    reward -= 0.15
                    info[f"transport_failed_{transport.id}"] = "Transport failure!"
                elif transport.travel_hours <= 0:
                    transport.status = "delivered"
                    completed_transports.append(transport)

        for transport in completed_transports:
            organ = next((o for o in s.organs if o.id == transport.organ_id), None)
            patient = next((p for p in s.patients if p.id == transport.patient_id), None)
            if organ and patient:
                viability_ratio = organ.viability_hours / organ.max_viability_hours
                transplant_reward = self._compute_transplant_reward(patient, organ, viability_ratio)
                reward += transplant_reward
                s.successful_transplants += 1
                patient.matched = True
                patient.match_score = round(transplant_reward, 4)
                info[f"transplant_success_{transport.id}"] = f"Reward: {transplant_reward:.4f}"

        # ── Handle action ─────────────────────────────────────────────────
        if action_type == "match":
            organ_id = action.get("organ_id")
            patient_id = action.get("patient_id")
            organ = next((o for o in s.organs if o.id == organ_id), None)
            patient = next((p for p in s.patients if p.id == patient_id), None)

            if organ is None:
                reward -= 0.05
                info["error"] = f"Organ {organ_id} not found"
            elif patient is None:
                reward -= 0.05
                info["error"] = f"Patient {patient_id} not found"
            elif organ.allocated:
                reward -= 0.05
                info["error"] = f"Organ {organ_id} already allocated"
            elif patient.matched:
                reward -= 0.05
                info["error"] = f"Patient {patient_id} already matched"
            elif organ.viability_hours == 0.0:
                reward -= 0.1
                info["error"] = f"Organ {organ_id} has expired"
            elif organ.organ_type != patient.organ_needed:
                reward -= 0.1
                info["error"] = "Organ type mismatch"
            elif not _is_compatible(organ.blood_type, patient.blood_type):
                reward -= 0.2
                info["error"] = "Blood type incompatible"
            else:
                # Valid match — dispatch transport
                dist = _haversine_km(organ.donor_city, patient.city)
                travel_h = max(1.0, dist / AVG_TRANSPORT_SPEED_KMH)  # min 1h even same city
                transport = Transport(
                    id=str(uuid.uuid4())[:8],
                    organ_id=organ_id,
                    patient_id=patient_id,
                    origin_city=organ.donor_city,
                    destination_city=patient.city,
                    distance_km=round(dist, 1),
                    travel_hours=round(travel_h, 2),
                )
                s.transports.append(transport)
                organ.allocated = True
                organ.allocated_to = patient_id
                reward += 0.05  # small reward for valid match dispatch
                info["match"] = f"Organ {organ_id} dispatched to {patient_id}, ETA {travel_h:.2f}h"

        elif action_type == "defer":
            organ_id = action.get("organ_id")
            organ = next((o for o in s.organs if o.id == organ_id), None)
            if organ is None:
                info["error"] = f"Organ {organ_id} not found"
            else:
                # Small penalty for deferring to encourage timely matching
                reward -= 0.02
                info["defer"] = f"Organ {organ_id} deferred. Viability remaining: {organ.viability_hours:.1f}h"

        elif action_type == "tick":
            info["tick"] = "Time advanced by 1 hour"

        else:
            reward -= 0.05
            info["error"] = f"Unknown action: {action_type}"

        s.step_count += 1
        s.total_reward += reward

        # ── Check done ────────────────────────────────────────────────────
        max_steps = self._task_cfg["max_steps"]
        all_organs_resolved = all(o.allocated or o.viability_hours == 0.0 for o in s.organs)
        no_active_transports = all(t.status != "in_transit" for t in s.transports)
        if s.step_count >= max_steps or (all_organs_resolved and no_active_transports):
            s.done = True
            info["episode_end"] = f"Steps: {s.step_count}, Total reward: {s.total_reward:.4f}"

        s.info = info

        return {
            "state": s.to_dict(),
            "reward": round(reward, 4),
            "done": s.done,
            "info": info,
        }

    def state(self) -> Dict:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state.to_dict()

    # ── Reward Computation ─────────────────────────────────────────────────

    def _compute_transplant_reward(self, patient: Patient, organ: Organ, viability_ratio: float) -> float:
        """
        Partial reward based on:
          - Patient urgency (higher urgency = more reward)
          - Patient wait time (longer wait = more reward)
          - Organ viability at delivery (fresher = more reward)
          - Survival probability improvement
        """
        urgency_score = patient.urgency / 5.0                     # 0.2 – 1.0
        wait_score = min(patient.wait_days / 500.0, 1.0)          # 0.0 – 1.0
        viability_score = viability_ratio                          # 0.0 – 1.0
        survival_score = patient.survival_probability              # 0.0 – 1.0 (higher = was healthier, still good)

        reward = (
            0.40 * urgency_score +
            0.20 * wait_score +
            0.25 * viability_score +
            0.15 * (1.0 - survival_score)   # more reward for saving critical patients
        )
        return round(reward, 4)

    # ── Grader ─────────────────────────────────────────────────────────────

    def grade(self) -> Dict:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        s = self._state
        cfg = self._task_cfg

        total_organs = len(s.organs)
        if total_organs == 0:
            return {"score": 0.001, "breakdown": {}}

        match_rate = min(s.successful_transplants / total_organs, 1.0)
        expiry_penalty = min(s.expired_organs / total_organs, 1.0)
        failure_penalty = min(s.failed_transplants / total_organs, 1.0)

        # Average match quality
        matched_patients = [p for p in s.patients if p.matched]
        avg_quality = (
            sum(p.match_score for p in matched_patients) / len(matched_patients)
            if matched_patients else 0.0
        )

        raw_score = (
            0.45 * match_rate +
            0.30 * avg_quality +
            0.15 * (1.0 - expiry_penalty) +
            0.10 * (1.0 - failure_penalty)
        )
        # Clamp strictly between 0 and 1 (exclusive) as required by validator
        score = round(min(max(raw_score, 0.001), 0.999), 4)

        return {
            "score": score,
            "breakdown": {
                "match_rate": round(match_rate, 4),
                "avg_match_quality": round(avg_quality, 4),
                "expiry_rate": round(expiry_penalty, 4),
                "transport_failure_rate": round(failure_penalty, 4),
                "successful_transplants": s.successful_transplants,
                "expired_organs": s.expired_organs,
                "failed_transports": s.failed_transplants,
                "total_steps": s.step_count,
                "total_reward": round(s.total_reward, 4),
            },
        }
