"""Bước B: một câu đơn lượt -> model Hugging Face -> predictions.jsonl.

Xem prompt, chưa tải model:
    python unit02_model.py --preview --out runs/b_preview
Chạy thật (cần torch + transformers):
    python unit02_model.py --out runs/b_first
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import time

from unit01_data import parse_jsonl, read_text, reject_reference_fields, validate_identity

ROOT = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = ROOT / "examples/unit01/questions.jsonl"
DEFAULT_CONFIG = ROOT / "configs/unit02_hf.json"
DEFAULT_PROMPT = ROOT / "prompts/system_vi_v1.txt"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ModelConfig:
    """B01/B02 — Chỉ một backend HF; revision được giải ra commit khi nạp."""
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    revision: str = "main"
    device: str = "auto"
    max_input_tokens: int = 2048
    max_new_tokens: int = 384
    seed: int = 2026

    def __post_init__(self):
        for key in ("model", "revision"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"{key} phai la chuoi khong rong")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device phai la auto, cpu hoac cuda")
        for key in ("max_input_tokens", "max_new_tokens"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} phai la so nguyen duong")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed phai la so nguyen trong [0, 2**32)")


def load_config(path: Path) -> ModelConfig:
    raw = json.loads(read_text(path))
    if not isinstance(raw, dict):
        raise ValueError("Config phai la JSON object")
    unknown = set(raw) - set(ModelConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Config co truong khong duoc ho tro: {sorted(unknown)}")
    return ModelConfig(**raw)


def select_one_question(path: Path, sample_id: str) -> dict:
    """Nhận output của A; bước B chỉ chạy ST lượt 1, không mất history MT."""
    rows = parse_jsonl(read_text(path))
    reject_reference_fields(rows)
    matches = [row for row in rows if row.get("sample_id") == sample_id]
    if len(matches) != 1:
        raise ValueError(f"Can dung mot ban ghi cho {sample_id}; tim thay {len(matches)}")
    row = matches[0]
    validate_identity(row)
    if row.get("scenario") != "single" or type(row.get("turn_id")) is not int or row["turn_id"] != 1:
        raise ValueError("Buoc B chi chay single turn_id=1. Hoi thoai se lam o buoc C.")
    if row["sample_id"] != f"{row['case_id']}__T1":
        raise ValueError("sample_id khong khop case_id/turn_id")
    if not isinstance(row.get("user"), str) or not row["user"].strip():
        raise ValueError("Cau hoi phai la chuoi khong rong")
    return {key: row[key] for key in ("sample_id", "case_id", "group_id", "scenario", "category", "turn_id", "user")}


def build_messages(question: str, system_prompt: str) -> list[dict]:
    """B03/B04 — Chỉ system prompt và câu hỏi; không có reference/nhãn/RAG."""
    if not isinstance(question, str) or not question.strip() or not system_prompt.strip():
        raise ValueError("Prompt va cau hoi khong duoc rong")
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]


def environment() -> dict:
    values = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "transformers", "huggingface-hub", "safetensors", "tokenizers"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


class HFBackend:
    """B05 — Nạp model/tokenizer một lần; giữ nguyên đối tượng cho bước C sau này."""

    def __init__(self, config: ModelConfig):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("Thieu thu vien. Xem README_BUOC_B.md hoac chay notebook Colab.") from exc
        self.config, self.torch = config, torch
        self.device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("Chua co CUDA GPU. Bat GPU tren Colab hoac dat device=cpu.")
        set_seed(config.seed)
        # Resolve once: model và tokenizer luôn dùng cùng commit, kể cả revision='main'.
        if re.fullmatch(r"[0-9a-f]{40}", config.revision):
            resolved = config.revision
        else:
            resolved = HfApi().model_info(config.model, revision=config.revision).sha
        if not resolved or not re.fullmatch(r"[0-9a-f]{40}", resolved):
            raise ValueError("Khong xac dinh duoc commit model; dung de tranh gan sai phien ban")
        self.resolved_revision = resolved
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, revision=resolved, trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise ValueError("Model chua co chat template; buoc nay can causal chat model co template")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model, revision=resolved, trust_remote_code=False,
            use_safetensors=True, torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        template_text = json.dumps(self.tokenizer.chat_template, ensure_ascii=False, sort_keys=True)
        self.info = {"backend": "huggingface", "model": config.model, "resolved_revision": resolved,
                     "device": self.device, "dtype": str(dtype), "is_test_double": False,
                     "chat_template_sha256": hashlib.sha256(template_text.encode()).hexdigest(),
                     "gpu_name": torch.cuda.get_device_name(0) if self.device == "cuda" else None}

    def inspect_input(self, messages: list[dict]) -> dict:
        """C06 — đếm input bằng đúng tokenizer/chat template và KHÔNG truncate.

        Hàm này chỉ kiểm tra budget, chưa gọi model.generate(). Nếu vượt giới hạn
        thì trả metadata fits=False để caller ghi bằng chứng trước khi dừng.
        """
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        size = int(encoded["input_ids"].shape[-1])
        model_limit = getattr(self.model.config, "max_position_embeddings", None)
        model_limit = model_limit if isinstance(model_limit, int) and model_limit > 0 else None
        reasons = []
        if size > self.config.max_input_tokens:
            reasons.append("max_input_tokens")
        if model_limit is not None and size + self.config.max_new_tokens > model_limit:
            reasons.append("model_context_window")
        return {
            "input_tokens": size,
            "max_input_tokens": self.config.max_input_tokens,
            "max_new_tokens": self.config.max_new_tokens,
            "model_context_window": model_limit,
            "fits": not reasons,
            "violations": reasons,
            "policy": "reject_no_truncation",
        }

    def generate(self, messages: list[dict]) -> tuple[str, dict]:
        """B06/C06 — kiểm tra budget rồi sinh assistant mới; tuyệt đối không tự truncate."""
        budget = self.inspect_input(messages)
        size = budget["input_tokens"]
        if not budget["fits"]:
            reason = ",".join(budget["violations"])
            raise ValueError(
                f"Input {size} tokens vuot gioi han ({reason}); policy=reject_no_truncation"
            )
        # Encode lại sau preflight để giữ API generate(messages) đơn giản và tránh
        # lưu tensor preflight trong ChatEngine. Không có bước truncate ở cả hai lần.
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded, do_sample=False, max_new_tokens=self.config.max_new_tokens,
                pad_token_id=pad,
            )
        new_tokens = output[0, size:].tolist()
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        if not answer.strip():
            raise ValueError("Model tra ve cau tra loi rong")
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        finish = "eos" if new_tokens and new_tokens[-1] in eos_ids else "length" if len(new_tokens) >= self.config.max_new_tokens else "other"
        return answer, {
            "input_tokens": size,
            "output_tokens": len(new_tokens),
            "finish_reason": finish,
            "generation": {"do_sample": False, "max_new_tokens": self.config.max_new_tokens},
            "input_budget": budget,
        }


def run_once(questions: Path, sample_id: str, config_path: Path, prompt_path: Path,
             out: Path, preview: bool = False, backend_factory=HFBackend) -> dict:
    """B07 — Một lần chạy có thư mục riêng, cấu hình, prompt, output hoặc lỗi."""
    cfg = load_config(config_path)
    question = select_one_question(questions, sample_id)
    messages = build_messages(question["user"], read_text(prompt_path))
    # Tất cả kiểm tra đầu vào hoàn tất trước khi tải trọng số hoặc tạo output.
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": "unit_b_0.2", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "prepared_no_model" if preview else "started", "model_load_attempted": False,
                "generation_attempted": False, "is_test_double": False,
                "sample_id": sample_id, "model_config": asdict(cfg), "environment": environment(),
                "questions_sha256": sha(questions), "system_prompt_sha256": sha(prompt_path),
                "code_sha256": sha(Path(__file__)), "data_code_sha256": sha(ROOT / "unit01_data.py"),
                "rag_enabled": False, "reference_used": False, "quality_evaluated": False}
    request = {"sample": question, "messages": messages, "model_config": asdict(cfg)}
    save_json(out / "request.json", request)
    save_json(out / "manifest.json", manifest)
    if preview:
        return {"status": "prepared_no_model", "question": question["user"], "answer": None}
    record = {key: question[key] for key in question if key != "user"} | {
        "question": question["user"], "messages": messages, "answer": None, "status": "pending",
        "history": [], "contexts": [], "is_test_double": False,
    }
    stage = "model_load"
    started = time.perf_counter()
    try:
        manifest["model_load_attempted"] = True
        save_json(out / "manifest.json", manifest)
        backend = backend_factory(cfg)
        manifest["model_load_ms"] = round((time.perf_counter() - started) * 1000, 3)
        manifest["backend_info"] = backend.info
        manifest["is_test_double"] = bool(backend.info.get("is_test_double", False))
        record["is_test_double"] = manifest["is_test_double"]
        stage = "generation"
        manifest["generation_attempted"] = True
        save_json(out / "manifest.json", manifest)
        started = time.perf_counter()
        answer, meta = backend.generate(messages)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Model tra ve cau tra loi rong/khong phai chuoi")
        record.update(status="ok", answer=answer, model_meta=meta,
                      generation_ms=round((time.perf_counter() - started) * 1000, 3))
        manifest["status"] = "completed_test_double" if record["is_test_double"] else "completed"
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status="error", error_stage=stage,
                      error={"type": type(exc).__name__, "message": str(exc)},
                      failed_stage_ms=round((time.perf_counter() - started) * 1000, 3))
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
    (out / "predictions.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["predictions_sha256"] = sha(out / "predictions.jsonl")
    save_json(out / "manifest.json", manifest)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--sample-id", default="GOUT_ST_001__T1")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview", action="store_true", help="Chi tao request; khong tai/goi model")
    args = parser.parse_args(argv)
    try:
        result = run_once(args.questions, args.sample_id, args.config, args.system_prompt, args.out, args.preview)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("QUESTION:", result["question"])
    if result["status"] == "ok":
        print("ANSWER:\n" + result["answer"])
    elif result["status"] == "error":
        print(f"ERROR ({result['error_stage']}): {result['error']['message']}", file=sys.stderr)
    else:
        print("PREVIEW_ONLY: chua tai/goi model, chua co predictions.")
    print("OUTPUT:", args.out)
    return 2 if result["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
