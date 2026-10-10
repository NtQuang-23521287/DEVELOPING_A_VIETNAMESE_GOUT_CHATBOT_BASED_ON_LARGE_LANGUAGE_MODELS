from __future__ import annotations

from typing import Any


def compute_reward(scores: dict[str, Any], config: dict[str, Any]) -> dict[str, float | bool]:
    """Combine 6 quality metrics and 3 risk penalties.

    Missing/N/A quality metrics are excluded and remaining quality weights are renormalized.
    Risk metrics are not renormalized. Fatal safety violations apply the configured hard penalty.
    """
    q_weights = config["quality_weights"]
    r_weights = config["risk_penalties"]

    available = [(k, float(v)) for k, v in scores.items() if k in q_weights and v is not None]
    denom = sum(float(q_weights[k]) for k, _ in available)
    quality = 0.0
    if denom > 0:
        quality = sum((float(q_weights[k]) / denom) * v for k, v in available)

    risk = 0.0
    for k, beta in r_weights.items():
        value = scores.get(k)
        if value is not None:
            risk += float(beta) * float(value)

    fatal = bool(scores.get("fatal_safety_violation", False))
    fatal_penalty = float(config.get("fatal_penalty", 0.0)) if fatal else 0.0
    total = quality - risk - fatal_penalty
    return {
        "quality_component": quality,
        "risk_penalty": risk,
        "fatal_penalty": fatal_penalty,
        "fatal": fatal,
        "reward": total,
    }
