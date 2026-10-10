from __future__ import annotations


def evaluate_ragas(*args, **kwargs):
    """Integration point for RAGAS after references are aligned and approved.

    Deliberately not auto-running on `pending_kb_alignment` references.
    """
    raise RuntimeError(
        "RAGAS is gated until test references are aligned with KB. "
        "Create an approved/reference-ready evaluation dataset first."
    )
