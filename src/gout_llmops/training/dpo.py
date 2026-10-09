from __future__ import annotations


def train_dpo(config: dict) -> None:
    """Phase 4 entrypoint gated until SFT and preference dataset are versioned."""
    if config.get("status") == "scaffold":
        raise RuntimeError("DPO scaffold is present, but SFT/preference data are not frozen yet.")
