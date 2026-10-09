from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from gout_llmops.core.io import read_jsonl


@dataclass(frozen=True)
class Turn:
    turn_id: int
    user: str


@dataclass(frozen=True)
class Case:
    case_id: str
    scenario: str
    subset: str
    target_disease: str
    source_row_id: int
    reference_status: str
    turns: tuple[Turn, ...]


def load_cases(path: str | Path) -> list[Case]:
    raw = read_jsonl(path)
    cases: list[Case] = []
    seen = set()
    for r in raw:
        case_id = str(r["case_id"])
        if case_id in seen:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen.add(case_id)
        turns = tuple(Turn(int(t["turn_id"]), str(t["user"]).strip()) for t in r["turns"])
        scenario = str(r["scenario"])
        expected = 1 if scenario == "single" else 3 if scenario == "multi" else None
        if expected is None:
            raise ValueError(f"Unknown scenario {scenario} in {case_id}")
        if len(turns) != expected:
            raise ValueError(f"{case_id}: {scenario} must have {expected} turns")
        if any(not t.user for t in turns):
            raise ValueError(f"{case_id}: empty user turn")
        cases.append(
            Case(
                case_id=case_id,
                scenario=scenario,
                subset=str(r.get("subset", "primary")),
                target_disease=str(r.get("target_disease", "")),
                source_row_id=int(r.get("source_row_id", -1)),
                reference_status=str(r.get("reference_status", "unknown")),
                turns=turns,
            )
        )
    return cases


def summarize_cases(cases: list[Case]) -> dict[str, Any]:
    by_scenario: dict[str, int] = {}
    by_subset: dict[str, int] = {}
    turns = 0
    for c in cases:
        by_scenario[c.scenario] = by_scenario.get(c.scenario, 0) + 1
        by_subset[c.subset] = by_subset.get(c.subset, 0) + 1
        turns += len(c.turns)
    return {
        "cases": len(cases),
        "turns": turns,
        "by_scenario": by_scenario,
        "by_subset": by_subset,
    }
