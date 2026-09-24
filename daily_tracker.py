"""
AI Gym Trainer — Daily Workout & Target Tracking Engine
=======================================================
Persists and calculates today's repetition counts, target goals (10 reps each),
and remaining exercises that still need to be completed today.
"""

import os
import json
from datetime import date

DATA_FILE = os.path.join(os.path.dirname(__file__), "daily_workout.json")
DEFAULT_TARGET = 10

_DEFAULT_EXERCISES = {
    "curl": {
        "name": "Dumbbell Curl",
        "key": "1",
        "color": (0, 200, 120),
    },
    "pec_dec": {
        "name": "Pec Dec Fly",
        "key": "2",
        "color": (0, 165, 255),
    },
    "shoulder_press": {
        "name": "Shoulder Press",
        "key": "3",
        "color": (255, 185, 30),
    },
}


def _get_today_str() -> str:
    return date.today().isoformat()


def load_daily_stats() -> dict:
    """Load today's workout stats. Resets automatically if a new day has arrived."""
    today = _get_today_str()

    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("date") == today and "exercises" in data:
                # Filter to only keep current active exercises
                data["exercises"] = {
                    ex_id: val
                    for ex_id, val in data["exercises"].items()
                    if ex_id in _DEFAULT_EXERCISES
                }
                # Ensure all current exercises exist in loaded data
                for ex_id, meta in _DEFAULT_EXERCISES.items():
                    if ex_id not in data["exercises"]:
                        data["exercises"][ex_id] = {
                            "name": meta["name"],
                            "reps": 0,
                            "completed": False,
                        }
                return data
        except Exception:
            pass

    # Initialize fresh stats for today
    fresh = {
        "date": today,
        "target_per_exercise": DEFAULT_TARGET,
        "exercises": {},
    }
    for ex_id, meta in _DEFAULT_EXERCISES.items():
        fresh["exercises"][ex_id] = {
            "name": meta["name"],
            "reps": 0,
            "completed": False,
        }
    save_daily_stats(fresh)
    return fresh


def save_daily_stats(data: dict):
    """Write stats safely to disk."""
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save daily stats: {e}")


def record_exercise_reps(exercise_id: str, reps_done: int) -> dict:
    """
    Add or update reps performed for an exercise today.
    Returns the updated stats dict.
    """
    data = load_daily_stats()
    target = data.get("target_per_exercise", DEFAULT_TARGET)

    if exercise_id in data["exercises"]:
        current_reps = data["exercises"][exercise_id].get("reps", 0)
        # Add reps to today's cumulative progress
        new_total = current_reps + reps_done
        data["exercises"][exercise_id]["reps"] = new_total
        data["exercises"][exercise_id]["completed"] = (new_total >= target)

    save_daily_stats(data)
    return data


def reset_daily_stats() -> dict:
    """Reset today's progress back to zero."""
    fresh = {
        "date": _get_today_str(),
        "target_per_exercise": DEFAULT_TARGET,
        "exercises": {},
    }
    for ex_id, meta in _DEFAULT_EXERCISES.items():
        fresh["exercises"][ex_id] = {
            "name": meta["name"],
            "reps": 0,
            "completed": False,
        }
    save_daily_stats(fresh)
    return fresh


def get_daily_summary(data: dict = None) -> dict:
    """
    Compute daily overview metrics:
      - Total reps done across all exercises
      - Total goal reps (30)
      - Progress percentage
      - Remaining exercises with reps left
      - all_completed boolean
    """
    if data is None:
        data = load_daily_stats()

    target = data.get("target_per_exercise", DEFAULT_TARGET)
    exercises = data.get("exercises", {})

    total_done   = 0
    total_target = len(exercises) * target
    remaining    = []

    for ex_id, item in exercises.items():
        reps = item.get("reps", 0)
        total_done += reps
        if reps < target:
            needed = target - reps
            remaining.append({
                "id": ex_id,
                "name": item.get("name", ex_id),
                "reps": reps,
                "needed": needed,
                "target": target,
            })

    pct = min(total_done / max(total_target, 1), 1.0)
    all_completed = (len(remaining) == 0)

    return {
        "total_done": total_done,
        "total_target": total_target,
        "percentage": pct,
        "remaining_exercises": remaining,
        "all_completed": all_completed,
    }
