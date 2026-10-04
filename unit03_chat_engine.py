"""C01-C07: chat_engine, history thật, giới hạn input và hội thoại ba lượt.

Đã có ở mốc này:
- C01: contract chat(question, history, contexts), không có reference/ground_truth.
- C02: một câu đi qua đúng builder/backend của khối B.
- C03: tạo history=[] mới, độc lập cho mỗi case.
- C04: chỉ thêm cặp user/assistant khi prediction có status=ok.
- C05: lượt kế tiếp nhận đúng các cặp user/assistant trước đó trong messages.
- C06: tokenizer/backend đếm input trước generation; vượt giới hạn thì reject, không truncate.
- C07: chạy trọn một case MT ba lượt; nếu một lượt lỗi, các lượt phụ thuộc sau đó là skipped.

Mốc này CHƯA làm RAG: contexts vẫn phải là [] cho tới D11.

Ví dụ gọi từ Python:
    from unit03_chat_engine import create_chat_engine
    engine = create_chat_engine()
    result = engine.chat("Gút là gì?", [], [])
    print(result["answer"])

Chạy một câu từ output A và lưu bằng chứng C02:
    python unit03_chat_engine.py --out runs/c02_first
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import unit02_model as b

ROOT = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = b.DEFAULT_QUESTIONS
DEFAULT_CONFIG = b.DEFAULT_CONFIG
DEFAULT_PROMPT = b.DEFAULT_PROMPT


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_empty_stage_input(name: str, value: list[dict], next_unit: str) -> list[dict]:
    """Giữ ranh giới unit: chưa cho chat() tiêu thụ history/context ngoài unit tương ứng."""
    if not isinstance(value, list):
        raise TypeError(f"{name} phai la list")
    if value:
        raise NotImplementedError(
            f"ChatEngine.chat hien chi cho phep {name}=[]; {name} khong rong se duoc xu ly o {next_unit}."
        )
    return value


def new_history() -> list[dict]:
    """C03 — tạo một lịch sử hoàn toàn mới cho một case/hội thoại mới.

    Mỗi lần gọi trả một list mới, không dùng mutable default hay biến toàn cục,
    nên việc append vào case A không thể làm thay đổi history của case B.
    """
    return []


def _validate_history_shape(history: list[dict]) -> None:
    """Kiểm tra history đã hoàn tất theo cặp user -> assistant."""
    if not isinstance(history, list):
        raise TypeError("history phai la list")
    if len(history) % 2 != 0:
        raise ValueError("history phai gom cac cap user/assistant da hoan tat")
    expected = ("user", "assistant")
    for idx, item in enumerate(history):
        if not isinstance(item, dict):
            raise TypeError(f"history[{idx}] phai la object")
        role = item.get("role")
        content = item.get("content")
        want = expected[idx % 2]
        if role != want:
            raise ValueError(f"history[{idx}].role phai la {want}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"history[{idx}].content khong duoc rong")


class InputLimitError(ValueError):
    """C06: lỗi preflight có kèm metadata để runner ghi bằng chứng."""

    def __init__(self, budget: dict):
        self.budget = dict(budget)
        tokens = self.budget.get("input_tokens")
        violations = ",".join(self.budget.get("violations") or []) or "unknown"
        super().__init__(
            f"Input {tokens} tokens vuot gioi han ({violations}); "
            f"policy={self.budget.get('policy', 'reject_no_truncation')}"
        )


def build_messages_with_history(
    question: str, system_prompt: str, history: list[dict]
) -> list[dict]:
    """C05 — system + toàn bộ history đã hoàn tất + câu hỏi hiện tại.

    History được copy, không sửa tại chỗ. Khi history rỗng, gọi lại đúng B04 để
    C02 vẫn đi qua cùng logic cũ. Không có reference/category/context trong messages.
    """
    _validate_history_shape(history)
    if not history:
        return b.build_messages(question, system_prompt)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question khong duoc rong")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt khong duoc rong")
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend({"role": item["role"], "content": item["content"]} for item in history)
    messages.append({"role": "user", "content": question})
    return messages


def inspect_input_budget(backend: Any, messages: list[dict]) -> dict:
    """C06 — yêu cầu backend kiểm tra token budget trước generation."""
    inspector = getattr(backend, "inspect_input", None)
    if not callable(inspector):
        raise TypeError("backend phai co ham inspect_input(messages) tu C06")
    budget = inspector(messages)
    if not isinstance(budget, dict):
        raise TypeError("backend.inspect_input phai tra dict")
    required = {"input_tokens", "max_input_tokens", "max_new_tokens", "fits", "violations", "policy"}
    missing = required - set(budget)
    if missing:
        raise ValueError(f"input budget thieu truong: {sorted(missing)}")
    if type(budget["input_tokens"]) is not int or budget["input_tokens"] < 0:
        raise ValueError("input_tokens phai la so nguyen khong am")
    if budget["policy"] != "reject_no_truncation":
        raise ValueError("C06 chi chap nhan policy=reject_no_truncation")
    if not isinstance(budget["violations"], list):
        raise TypeError("violations phai la list")
    if budget["fits"] is not True:
        raise InputLimitError(budget)
    return dict(budget)


def append_turn(
    history: list[dict],
    question: str,
    answer: str | None,
    *,
    status: str,
) -> list[dict]:
    """C04 — thêm output THẬT của một lượt thành công vào history.

    - Nếu status != "ok", giữ history nguyên trạng: lỗi không trở thành lượt thành công.
    - Không có tham số reference/ground_truth, nên helper không có đường riêng để
      đưa đáp án chuẩn vào lịch sử. Caller phải truyền answer của prediction/model.
    - Trả một list mới để tránh side effect ngoài ý muốn.
    """
    _validate_history_shape(history)
    copied = [dict(item) for item in history]
    if status != "ok":
        return copied
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question khong duoc rong khi status=ok")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer khong duoc rong khi status=ok")
    copied.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    return copied


def history_from_prediction(prediction: dict) -> list[dict]:
    """C03+C04 cho một prediction đầu tiên: [] -> [user, assistant] nếu thành công."""
    if not isinstance(prediction, dict):
        raise TypeError("prediction phai la object")
    history = new_history()
    return append_turn(
        history,
        prediction.get("question"),
        prediction.get("answer"),
        status=prediction.get("status", ""),
    )


def build_c03_c04_evidence(prediction_path: Path, out: Path) -> dict:
    """Tạo bằng chứng C03-C04 từ prediction thật đã có ở C02; không gọi model lại."""
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    lines = [line for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("C03-C04 evidence hien can dung dung mot prediction C02")
    prediction = json.loads(lines[0])
    if not isinstance(prediction, dict):
        raise TypeError("prediction JSON phai la object")

    out.mkdir(parents=True, exist_ok=False)
    initial = new_history()
    final_history = append_turn(
        initial,
        prediction.get("question"),
        prediction.get("answer"),
        status=prediction.get("status", ""),
    )
    evidence = {
        "schema_version": "unit_c03_c04_0.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_prediction_sha256": b.sha(prediction_path),
        "source_sample_id": prediction.get("sample_id"),
        "source_status": prediction.get("status"),
        "model_called": False,
        "reference_used": False,
        "c03": {
            "initial_history": initial,
            "passed": initial == [],
        },
        "c04": {
            "history_after_turn": final_history,
            "history_message_count": len(final_history),
            "turn_pair_count": len(final_history) // 2,
            "appended": prediction.get("status") == "ok" and len(final_history) == 2,
        },
    }
    b.save_json(out / "manifest.json", evidence)
    (out / "history.json").write_text(
        json.dumps(final_history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return evidence


class ChatEngine:
    """C01 — contract ổn định cho chatbot và evaluator.

    Public API duy nhất ở mốc này:
        chat(question, history, contexts) -> {
            "answer": str,
            "sources": list,
            "call": dict,
        }

    Không nhận reference, ground_truth, category hay đáp án chuẩn.
    """

    def __init__(self, backend: Any, system_prompt: str):
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt khong duoc rong")
        if not hasattr(backend, "generate") or not callable(backend.generate):
            raise TypeError("backend phai co ham generate(messages)")
        if not hasattr(backend, "inspect_input") or not callable(backend.inspect_input):
            raise TypeError("backend phai co ham inspect_input(messages) tu C06")
        self.backend = backend
        self.system_prompt = system_prompt
        self.system_prompt_sha256 = _sha_text(system_prompt)

    def chat(self, question: str, history: list[dict], contexts: list[dict]) -> dict:
        """C05-C06 — dùng history thật, preflight token, rồi mới sinh câu trả lời.

        Contexts vẫn bị chặn đến D11. History phải là các cặp user/assistant
        hoàn tất; câu hỏi hiện tại luôn là message cuối và system luôn là message đầu.
        """
        _validate_history_shape(history)
        _validate_empty_stage_input("contexts", contexts, "D11")

        messages = build_messages_with_history(question, self.system_prompt, history)
        # C06 chạy trước backend.generate: nếu quá dài thì dừng, không truncate và
        # không để model sinh trên prompt bị cắt âm thầm.
        input_budget = inspect_input_budget(self.backend, messages)

        started = time.perf_counter()
        answer, model_meta = self.backend.generate(messages)
        generation_ms = round((time.perf_counter() - started) * 1000, 3)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Model tra ve cau tra loi rong/khong phai chuoi")
        if not isinstance(model_meta, dict):
            raise TypeError("backend.generate phai tra model_meta la dict")

        backend_info = getattr(self.backend, "info", {})
        if backend_info is None:
            backend_info = {}
        if not isinstance(backend_info, dict):
            raise TypeError("backend.info phai la dict neu duoc cung cap")

        return {
            "answer": answer,
            "sources": [],
            "call": {
                "messages": messages,
                "history_turns": len(history) // 2,
                "history_message_count": len(history),
                "context_count": 0,
                "rag_enabled": False,
                "reference_used": False,
                "system_prompt_sha256": self.system_prompt_sha256,
                "backend": backend_info,
                "input_budget": input_budget,
                "model_meta": model_meta,
                "generation_ms": generation_ms,
            },
        }



def create_chat_engine(
    config_path: Path = DEFAULT_CONFIG,
    prompt_path: Path = DEFAULT_PROMPT,
    backend_factory=b.HFBackend,
) -> ChatEngine:
    """Nạp backend đúng một lần rồi trả engine có thể gọi lại từ Python."""
    cfg = b.load_config(config_path)
    prompt = b.read_text(prompt_path)
    backend = backend_factory(cfg)
    return ChatEngine(backend=backend, system_prompt=prompt)


def select_multi_case_turns(questions: Path, case_id: str, count: int = 2) -> list[dict]:
    """C05 — lấy các lượt đầu của đúng MỘT multi-turn case, không mượn ST cùng group."""
    if type(count) is not int or count < 1:
        raise ValueError("count phai la so nguyen duong")
    rows = b.parse_jsonl(b.read_text(questions))
    b.reject_reference_fields(rows)
    matches = [row for row in rows if row.get("case_id") == case_id]
    if not matches:
        raise ValueError(f"Khong tim thay case_id={case_id}")
    for row in matches:
        b.validate_identity(row)
        if row.get("scenario") != "multi":
            raise ValueError("C05 chi chay case multi; khong dung output ST lam history MT")
        if not isinstance(row.get("turn_id"), int) or row["turn_id"] < 1:
            raise ValueError("turn_id multi khong hop le")
        if row.get("sample_id") != f"{case_id}__T{row['turn_id']}":
            raise ValueError("sample_id khong khop case_id/turn_id")
        if not isinstance(row.get("user"), str) or not row["user"].strip():
            raise ValueError("Cau hoi multi khong duoc rong")
    matches.sort(key=lambda row: row["turn_id"])
    expected = list(range(1, len(matches) + 1))
    if [row["turn_id"] for row in matches] != expected:
        raise ValueError("turn_id cua case multi phai lien tuc tu 1")
    if len(matches) < count:
        raise ValueError(f"Case {case_id} chi co {len(matches)} luot, can it nhat {count}")
    keys = ("sample_id", "case_id", "group_id", "scenario", "category", "turn_id", "user")
    return [{key: row[key] for key in keys} for row in matches[:count]]


def run_two_turns(
    questions: Path,
    case_id: str,
    config_path: Path,
    prompt_path: Path,
    out: Path,
    backend_factory=b.HFBackend,
) -> dict:
    """C05-C06 — chạy MT T1 -> history thật -> MT T2, có token preflight từng lượt.

    Đây chưa phải C07: chỉ chạy đúng hai lượt để kiểm chứng history và input budget.
    """
    cfg = b.load_config(config_path)
    turns = select_multi_case_turns(questions, case_id, count=2)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": "unit_c05_c06_0.1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "case_id": case_id,
        "group_id": turns[0]["group_id"],
        "model_config": asdict(cfg),
        "environment": b.environment(),
        "questions_sha256": b.sha(questions),
        "system_prompt_sha256": b.sha(prompt_path),
        "chat_engine_code_sha256": b.sha(Path(__file__)),
        "model_code_sha256": b.sha(ROOT / "unit02_model.py"),
        "history_enabled": True,
        "rag_enabled": False,
        "reference_used": False,
        "input_policy": "reject_no_truncation",
        "model_load_attempted": False,
        "generation_attempted": 0,
    }
    b.save_json(out / "manifest.json", manifest)

    records = []
    history = new_history()
    stage = "model_load"
    started = time.perf_counter()
    try:
        manifest["model_load_attempted"] = True
        b.save_json(out / "manifest.json", manifest)
        engine = create_chat_engine(config_path, prompt_path, backend_factory=backend_factory)
        manifest["model_load_ms"] = round((time.perf_counter() - started) * 1000, 3)
        manifest["backend_info"] = getattr(engine.backend, "info", {})
        manifest["is_test_double"] = bool(manifest["backend_info"].get("is_test_double", False))

        for sample in turns:
            before = [dict(item) for item in history]
            record = {
                **{key: sample[key] for key in sample if key != "user"},
                "question": sample["user"],
                "history_before": before,
                "contexts": [],
                "answer": None,
                "sources": [],
                "status": "pending",
            }
            stage = "input_budget_or_generation"
            try:
                manifest["generation_attempted"] += 1
                result = engine.chat(sample["user"], history, [])
                record.update(
                    answer=result["answer"],
                    sources=result["sources"],
                    call=result["call"],
                    status="ok",
                    is_test_double=manifest["is_test_double"],
                )
                history = append_turn(
                    history, sample["user"], result["answer"], status="ok"
                )
                record["history_after"] = [dict(item) for item in history]
                records.append(record)
            except InputLimitError as exc:
                record.update(
                    status="error",
                    error_stage="input_budget",
                    input_budget=exc.budget,
                    error={"type": type(exc).__name__, "message": str(exc)},
                    history_after=before,
                )
                records.append(record)
                break
            except (Exception, KeyboardInterrupt) as exc:
                record.update(
                    status="error",
                    error_stage="generation",
                    error={"type": type(exc).__name__, "message": str(exc)},
                    history_after=before,
                )
                records.append(record)
                if isinstance(exc, KeyboardInterrupt):
                    raise
                break

        if len(records) == 2 and all(row["status"] == "ok" for row in records):
            t1_answer = records[0]["answer"]
            t2_messages = records[1]["call"]["messages"]
            manifest["c05_verified"] = (
                records[1]["history_before"][-1]["content"] == t1_answer
                and any(m.get("role") == "assistant" and m.get("content") == t1_answer for m in t2_messages)
                and records[0]["case_id"] == records[1]["case_id"] == case_id
            )
            manifest["c06_verified"] = all(
                row["call"]["input_budget"]["fits"] is True
                and row["call"]["input_budget"]["policy"] == "reject_no_truncation"
                for row in records
            )
            manifest["status"] = (
                "completed_test_double" if manifest["is_test_double"] else "completed"
            )
        else:
            manifest["c05_verified"] = False
            manifest["c06_verified"] = False
            manifest["status"] = "failed"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["fatal_error"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}

    pred_path = out / "predictions.jsonl"
    pred_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    (out / "history_final.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["prediction_count"] = len(records)
    manifest["successful_turns"] = sum(row.get("status") == "ok" for row in records)
    manifest["predictions_sha256"] = b.sha(pred_path)
    b.save_json(out / "manifest.json", manifest)
    return {"manifest": manifest, "predictions": records, "history": history}


def _base_turn_record(sample: dict, history: list[dict]) -> dict:
    """Tạo record chuẩn cho một lượt MT mà không làm thay đổi history."""
    before = [dict(item) for item in history]
    return {
        **{key: sample[key] for key in sample if key != "user"},
        "question": sample["user"],
        "history_before": before,
        "contexts": [],
        "answer": None,
        "sources": [],
        "status": "pending",
    }


def _dependent_turn_is_skipped(sample: dict, history: list[dict], failed_turn_id: int) -> dict:
    """C07 — lượt phụ thuộc sau lỗi phải được ghi skipped và tuyệt đối không gọi model."""
    record = _base_turn_record(sample, history)
    record.update(
        status="skipped",
        skip_reason={
            "type": "dependency_failed",
            "depends_on_turn_id": failed_turn_id,
            "message": f"Khong chay T{sample['turn_id']} vi T{failed_turn_id} da loi.",
        },
        history_after=[dict(item) for item in history],
    )
    return record


def run_three_turns(
    questions: Path,
    case_id: str,
    config_path: Path,
    prompt_path: Path,
    out: Path,
    backend_factory=b.HFBackend,
) -> dict:
    """C07 — chạy trọn MT T1 -> T2 -> T3, giữ history thật và dependency skip.

    Quy tắc:
    - Chọn đúng ba lượt đầu của cùng một case multi và giữ thứ tự 1,2,3.
    - Chỉ append history khi lượt có status=ok.
    - Khi một lượt error, mọi lượt sau phụ thuộc vào nó được ghi status=skipped.
    - Lượt skipped không gọi inspect_input/generate và không được tính là thành công.
    - Contexts vẫn rỗng; reference/ground_truth không đi vào engine.
    """
    cfg = b.load_config(config_path)
    turns = select_multi_case_turns(questions, case_id, count=3)
    if [row["turn_id"] for row in turns] != [1, 2, 3]:
        raise ValueError("C07 can dung dung ba luot T1,T2,T3 theo thu tu")

    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": "unit_c07_0.1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "case_id": case_id,
        "group_id": turns[0]["group_id"],
        "expected_turns": 3,
        "model_config": asdict(cfg),
        "environment": b.environment(),
        "questions_sha256": b.sha(questions),
        "system_prompt_sha256": b.sha(prompt_path),
        "chat_engine_code_sha256": b.sha(Path(__file__)),
        "model_code_sha256": b.sha(ROOT / "unit02_model.py"),
        "history_enabled": True,
        "rag_enabled": False,
        "reference_used": False,
        "input_policy": "reject_no_truncation",
        "dependency_policy": "error_then_skip_remaining_turns",
        "model_load_attempted": False,
        "turn_calls_attempted": 0,
    }
    b.save_json(out / "manifest.json", manifest)

    records: list[dict] = []
    history = new_history()
    engine = None
    failed_turn_id: int | None = None
    stage = "model_load"
    load_started = time.perf_counter()

    try:
        manifest["model_load_attempted"] = True
        b.save_json(out / "manifest.json", manifest)
        engine = create_chat_engine(config_path, prompt_path, backend_factory=backend_factory)
        manifest["model_load_ms"] = round((time.perf_counter() - load_started) * 1000, 3)
        manifest["backend_info"] = getattr(engine.backend, "info", {})
        manifest["is_test_double"] = bool(manifest["backend_info"].get("is_test_double", False))
    except (Exception, KeyboardInterrupt) as exc:
        manifest["backend_info"] = {}
        manifest["is_test_double"] = False
        first = _base_turn_record(turns[0], history)
        first.update(
            status="error",
            error_stage="model_load",
            error={"type": type(exc).__name__, "message": str(exc)},
            history_after=[dict(item) for item in history],
        )
        records.append(first)
        failed_turn_id = turns[0]["turn_id"]
        for sample in turns[1:]:
            records.append(_dependent_turn_is_skipped(sample, history, failed_turn_id))
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed_model_load"

    if engine is not None:
        for sample in turns:
            if failed_turn_id is not None:
                records.append(_dependent_turn_is_skipped(sample, history, failed_turn_id))
                continue

            record = _base_turn_record(sample, history)
            stage = "input_budget_or_generation"
            try:
                manifest["turn_calls_attempted"] += 1
                result = engine.chat(sample["user"], history, [])
                record.update(
                    answer=result["answer"],
                    sources=result["sources"],
                    call=result["call"],
                    status="ok",
                    is_test_double=manifest["is_test_double"],
                )
                history = append_turn(history, sample["user"], result["answer"], status="ok")
                record["history_after"] = [dict(item) for item in history]
                records.append(record)
            except InputLimitError as exc:
                record.update(
                    status="error",
                    error_stage="input_budget",
                    input_budget=exc.budget,
                    error={"type": type(exc).__name__, "message": str(exc)},
                    history_after=[dict(item) for item in history],
                )
                records.append(record)
                failed_turn_id = sample["turn_id"]
            except (Exception, KeyboardInterrupt) as exc:
                record.update(
                    status="error",
                    error_stage="generation",
                    error={"type": type(exc).__name__, "message": str(exc)},
                    history_after=[dict(item) for item in history],
                )
                records.append(record)
                failed_turn_id = sample["turn_id"]
                if isinstance(exc, KeyboardInterrupt):
                    manifest["status"] = "interrupted"

    ordered = len(records) == 3 and [r.get("turn_id") for r in records] == [1, 2, 3]
    same_case = len(records) == 3 and all(r.get("case_id") == case_id for r in records)
    statuses = [r.get("status") for r in records]
    success_count = statuses.count("ok")
    error_count = statuses.count("error")
    skipped_count = statuses.count("skipped")

    dependency_skip_verified = True
    first_error_idx = next((i for i, s in enumerate(statuses) if s == "error"), None)
    if first_error_idx is not None:
        dependency_skip_verified = all(s == "skipped" for s in statuses[first_error_idx + 1 :])
    elif skipped_count:
        dependency_skip_verified = False

    manifest["ordered_three_predictions"] = ordered
    manifest["same_case_verified"] = same_case
    manifest["dependency_skip_verified"] = dependency_skip_verified
    manifest["successful_turns"] = success_count
    manifest["error_turns"] = error_count
    manifest["skipped_turns"] = skipped_count
    manifest["c07_structure_verified"] = ordered and same_case and dependency_skip_verified
    manifest["c07_successful_three_turn_run"] = (
        manifest["c07_structure_verified"] and statuses == ["ok", "ok", "ok"]
    )

    if manifest.get("status") != "interrupted":
        if manifest.get("status") == "failed_model_load":
            pass
        elif manifest["c07_successful_three_turn_run"]:
            manifest["status"] = (
                "completed_test_double" if manifest.get("is_test_double") else "completed"
            )
        elif manifest["c07_structure_verified"] and error_count == 1:
            manifest["status"] = "completed_with_turn_error"
        else:
            manifest["status"] = "failed"

    pred_path = out / "predictions.jsonl"
    pred_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    (out / "history_final.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["prediction_count"] = len(records)
    manifest["predictions_sha256"] = b.sha(pred_path)
    b.save_json(out / "manifest.json", manifest)
    return {"manifest": manifest, "predictions": records, "history": history}


def run_one(
    questions: Path,
    sample_id: str,
    config_path: Path,
    prompt_path: Path,
    out: Path,
    backend_factory=b.HFBackend,
) -> dict:
    """Tạo bằng chứng chạy C02 với một câu ST; chưa làm hội thoại C03+."""
    cfg = b.load_config(config_path)
    sample = b.select_one_question(questions, sample_id)
    out.mkdir(parents=True, exist_ok=False)

    manifest = {
        "schema_version": "unit_c02_0.1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "sample_id": sample_id,
        "model_config": asdict(cfg),
        "environment": b.environment(),
        "questions_sha256": b.sha(questions),
        "system_prompt_sha256": b.sha(prompt_path),
        "chat_engine_code_sha256": b.sha(Path(__file__)),
        "model_code_sha256": b.sha(ROOT / "unit02_model.py"),
        "rag_enabled": False,
        "reference_used": False,
        "history_enabled": False,
        "model_load_attempted": False,
        "generation_attempted": False,
    }
    b.save_json(out / "manifest.json", manifest)

    stage = "model_load"
    started = time.perf_counter()
    record = {
        "sample_id": sample["sample_id"],
        "case_id": sample["case_id"],
        "group_id": sample["group_id"],
        "scenario": sample["scenario"],
        "category": sample["category"],
        "turn_id": sample["turn_id"],
        "question": sample["user"],
        "history": [],
        "contexts": [],
        "answer": None,
        "sources": [],
        "status": "pending",
    }
    try:
        manifest["model_load_attempted"] = True
        b.save_json(out / "manifest.json", manifest)
        engine = create_chat_engine(config_path, prompt_path, backend_factory=backend_factory)
        manifest["model_load_ms"] = round((time.perf_counter() - started) * 1000, 3)
        manifest["backend_info"] = getattr(engine.backend, "info", {})
        manifest["is_test_double"] = bool(manifest["backend_info"].get("is_test_double", False))

        stage = "generation"
        manifest["generation_attempted"] = True
        b.save_json(out / "manifest.json", manifest)
        result = engine.chat(sample["user"], [], [])
        record.update(
            answer=result["answer"],
            sources=result["sources"],
            call=result["call"],
            status="ok",
            is_test_double=manifest["is_test_double"],
        )
        manifest["status"] = "completed_test_double" if manifest["is_test_double"] else "completed"
    except (Exception, KeyboardInterrupt) as exc:
        record.update(
            status="error",
            error_stage=stage,
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"

    (out / "predictions.jsonl").write_text(
        json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["predictions_sha256"] = b.sha(out / "predictions.jsonl")
    b.save_json(out / "manifest.json", manifest)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--sample-id", default="GOUT_ST_001__T1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_one(
            args.questions, args.sample_id, args.config, args.system_prompt, args.out
        )
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("QUESTION:", result["question"])
    if result["status"] == "ok":
        print("ANSWER:\n" + result["answer"])
        print("SOURCES:", result["sources"])
    else:
        print(
            f"ERROR ({result['error_stage']}): {result['error']['message']}",
            file=sys.stderr,
        )
    print("OUTPUT:", args.out)
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
