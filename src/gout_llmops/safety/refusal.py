from __future__ import annotations

DANGEROUS_PATTERNS = (
    "tự tăng liều",
    "tự giảm liều",
    "ngừng thuốc ngay",
)


def simple_risk_flag(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in DANGEROUS_PATTERNS)
