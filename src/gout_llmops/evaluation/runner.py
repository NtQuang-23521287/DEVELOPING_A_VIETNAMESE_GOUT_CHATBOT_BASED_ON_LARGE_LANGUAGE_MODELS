from __future__ import annotations

import json
import random
import traceback
from pathlib import Path
from typing import Any

from gout_llmops.chat.engine import ChatEngine
from gout_llmops.core.config import load_yaml
from gout_llmops.core.hashing import sha256_file
from gout_llmops.core.io import write_json, write_jsonl
from gout_llmops.core.run_context import RunContext, git_commit
from gout_llmops.data.testset import load_cases, summarize_cases
from gout_llmops.evaluation.metrics import aggregate_predictions
from gout_llmops.models.registry import build_backend
from gout_llmops.rag.retriever import FaissRetriever
from gout_llmops.tracking.mlflow_tracker import mlflow_run


def run_baseline(
    model_config_path: str,
    eval_config_path: str,
    limit: int | None = None,
    enable_judge: bool | None = None,
    enable_ragas: bool | None = None,
) -> Path:
    model_cfg = load_yaml(model_config_path)
    eval_cfg = load_yaml(eval_config_path)
    rag_cfg = load_yaml(eval_cfg["rag"]["config"])

    random.seed(int(eval_cfg.get("seed", 2026)))
    cases = load_cases(eval_cfg["testset"]["path"])
    if limit is not None:
        cases = cases[:limit]

    # Candidate v02 intentionally remains pending; predictions can be generated,
    # but medical quality scores requiring reference must be gated later.
    run = RunContext.create(eval_cfg["output"]["root"], model_cfg["name"])
    write_json(run.output_dir / "model_config.json", model_cfg)
    write_json(run.output_dir / "eval_config.json", eval_cfg)
    write_json(run.output_dir / "rag_config.json", rag_cfg)

    manifest = {
        "run_id": run.run_id,
        "phase": eval_cfg["phase"],
        "model_name": model_cfg["name"],
        "model_id": model_cfg["model_id"],
        "model_revision": model_cfg.get("revision"),
        "git_commit": git_commit(),
        "model_config_sha256": sha256_file(model_config_path),
        "eval_config_sha256": sha256_file(eval_config_path),
        "rag_config_sha256": sha256_file(eval_cfg["rag"]["config"]),
        "testset_path": eval_cfg["testset"]["path"],
        "testset_sha256": sha256_file(eval_cfg["testset"]["path"]),
        "kb_manifest_sha256": sha256_file(Path(eval_cfg["rag"]["index_dir"]) / "manifest.json"),
        "prompt_sha256": sha256_file(eval_cfg["prompt"]["system"]),
        "case_summary": summarize_cases(cases),
    }
    write_json(run.output_dir / "run_manifest.json", manifest)

    backend = build_backend(model_cfg)
    retriever = FaissRetriever(eval_cfg["rag"]["index_dir"], rag_cfg)
    system_prompt = Path(eval_cfg["prompt"]["system"]).read_text(encoding="utf-8")
    engine = ChatEngine(
        backend,
        retriever,
        system_prompt,
        max_history_pairs=int(eval_cfg["conversation"].get("max_history_pairs", 2)),
        max_input_tokens=int(eval_cfg["input"].get("max_input_tokens", 4096)),
    )

    records: list[dict[str, Any]] = []
    retrieval_trace: list[dict[str, Any]] = []
    stop_case_ids = set()

    params = {
        "model_id": model_cfg["model_id"],
        "model_revision": model_cfg.get("revision"),
        "testset_sha256": manifest["testset_sha256"],
        "kb_manifest_sha256": manifest["kb_manifest_sha256"],
        "prompt_sha256": manifest["prompt_sha256"],
        "seed": eval_cfg.get("seed"),
    }
    with mlflow_run(eval_cfg.get("tracking", {}), run.run_id, params) as mlflow:
        for case in cases:
            history: list[dict[str, str]] = []
            for turn in case.turns:
                if case.case_id in stop_case_ids:
                    records.append(
                        {
                            "case_id": case.case_id,
                            "scenario": case.scenario,
                            "subset": case.subset,
                            "turn_id": turn.turn_id,
                            "question": turn.user,
                            "status": "skipped_dependency_failure",
                        }
                    )
                    continue
                try:
                    result, history = engine.chat(turn.user, history)
                    record = {
                        "case_id": case.case_id,
                        "scenario": case.scenario,
                        "subset": case.subset,
                        "turn_id": turn.turn_id,
                        "reference_status": case.reference_status,
                        "status": "ok",
                        **{k: v for k, v in result.items() if k != "messages"},
                    }
                    records.append(record)
                    retrieval_trace.append(
                        {
                            "case_id": case.case_id,
                            "turn_id": turn.turn_id,
                            "question": turn.user,
                            "contexts": result["contexts"],
                        }
                    )
                except Exception as e:
                    records.append(
                        {
                            "case_id": case.case_id,
                            "scenario": case.scenario,
                            "subset": case.subset,
                            "turn_id": turn.turn_id,
                            "question": turn.user,
                            "status": "error",
                            "error_type": type(e).__name__,
                            "error": str(e),
                            "traceback": traceback.format_exc(limit=3),
                        }
                    )
                    if case.scenario == "multi":
                        stop_case_ids.add(case.case_id)

        write_jsonl(run.output_dir / "predictions.jsonl", records)
        write_jsonl(run.output_dir / "retrieval_trace.jsonl", retrieval_trace)
        summary = aggregate_predictions(records)
        summary["judge_requested"] = bool(enable_judge if enable_judge is not None else eval_cfg["judge"].get("enabled"))
        summary["ragas_requested"] = bool(enable_ragas if enable_ragas is not None else eval_cfg["ragas"].get("enabled"))
        summary["scoring_note"] = (
            "Predictions are valid pipeline outputs. Medical scoring is intentionally gated while "
            "reference_status remains pending_kb_alignment."
        )
        write_json(run.output_dir / "summary.json", summary)
        if mlflow:
            for k, v in summary.items():
                if isinstance(v, (int, float)) and v is not None:
                    mlflow.log_metric(k, float(v))
            mlflow.log_artifacts(str(run.output_dir), artifact_path="run_artifacts")
    return run.output_dir
