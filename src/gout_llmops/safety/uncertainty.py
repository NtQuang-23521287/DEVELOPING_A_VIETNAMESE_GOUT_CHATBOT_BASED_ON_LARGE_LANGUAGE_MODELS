from __future__ import annotations


def require_calibrated_threshold(config: dict) -> float:
    threshold = config.get("uncertainty", {}).get("threshold")
    if threshold is None:
        raise RuntimeError("Uncertainty threshold must be calibrated on development data before release.")
    return float(threshold)
