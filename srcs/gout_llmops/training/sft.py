from __future__ import annotations


def train_sft(config: dict) -> None:
    """Phase 3 entrypoint.

    Intentionally gated until the baseline Best Model and SFT dataset version are frozen.
    """
    if config.get("status") == "scaffold":
        raise RuntimeError("SFT scaffold is present, but baseline selection/data approval is not finished.")
