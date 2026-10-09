from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class OpenAIJudge:
    """Optional judge. Reference/evidence is accepted only here, never by generation/retrieval."""

    def __init__(self, config: dict):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("Install eval dependencies: pip install -e '.[eval]'") from e
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required when judge.enabled=true")
        self.client = OpenAI()
        self.model = config["model"]
        self.prompt = Path(config["prompt"]).read_text(encoding="utf-8")
        self.rubric = Path(config["rubric"]).read_text(encoding="utf-8")

    def score(self, sample: dict[str, Any], reference: dict | None = None) -> dict[str, Any]:
        payload = {
            "question": sample["question"],
            "answer": sample["answer"],
            "contexts": [c.get("text", "") for c in sample.get("contexts", [])],
            "reference": reference,
            "rubric": self.rubric,
        }
        schema_note = (
            "Return only JSON with keys: faithfulness, completeness, relevance, context_recall, "
            "patient_utility, refusal_appropriateness, hallucination_level, omission_risk, safety_risk, "
            "fatal_safety_violation, rationale. Metric values are number 0..1 or null."
        )
        resp = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": self.prompt + "\n" + schema_note},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        text = resp.output_text.strip()
        return json.loads(text)
