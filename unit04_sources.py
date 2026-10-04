"""D01-D02: đăng ký Knowledge Base thật của project và duyệt phạm vi sử dụng.

Nguồn chính từ thời điểm này:
    data/kb/raw_docs/gout_guideline_1.pdf

D01:
- tính checksum trực tiếp từ PDF thật đang nằm trong project
- tạo source_registry entry có định danh/version/publisher/path/checksum/page count/provenance

D02:
- chấp nhận PDF này làm Knowledge Base versioned của project và cho phép dùng ở pipeline RAG
- KHÔNG tự tuyên bố mọi khuyến cáo năm 2014 là thực hành y khoa mới nhất năm 2026
- không chấm/diễn giải nội dung y khoa ở unit này

D03 mới bắt đầu trích text theo trang/mục.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "data" / "kb" / "raw_docs" / "gout_guideline_1.pdf"
DEFAULT_PROVENANCE = ROOT / "data" / "inherited" / "moh361_gout_official_evidence.json"
EXPECTED_SHA256 = "763fd9a729ab5beccf81eb78d083fcf292ff9cbc1bf620d099529ac7a5c478ca"
EXPECTED_BYTES = 193085
EXPECTED_PAGES = 6
HEX64 = set("0123456789abcdef")


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _validate_sha256(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX64 for c in value.lower()):
        raise ValueError("sha256 phai la 64 ky tu hex")


def validate_registry_entry(entry: dict) -> None:
    required = {
        "source_id", "title", "version", "source_kind", "publisher",
        "local_asset", "original_source", "review"
    }
    missing = required - set(entry)
    if missing:
        raise ValueError(f"source registry thieu truong: {sorted(missing)}")
    if not isinstance(entry["source_id"], str) or not entry["source_id"].strip():
        raise ValueError("source_id khong duoc rong")
    local = entry["local_asset"]
    for key in ("path", "sha256", "bytes", "page_count"):
        if key not in local:
            raise ValueError(f"local_asset thieu {key}")
    _validate_sha256(local["sha256"])
    if not isinstance(local["bytes"], int) or local["bytes"] <= 0:
        raise ValueError("local_asset.bytes phai > 0")
    if not isinstance(local["page_count"], int) or local["page_count"] <= 0:
        raise ValueError("local_asset.page_count phai > 0")
    if Path(local["path"]).is_absolute():
        raise ValueError("local_asset.path phai la duong dan tuong doi trong project")
    if entry["review"].get("source_scope_status") not in {"candidate", "approved_for_project_kb", "rejected"}:
        raise ValueError("source_scope_status khong hop le")


def build_project_kb_registry_entry(source_path: Path, provenance: dict) -> dict:
    """D01 — registry entry từ chính PDF Knowledge Base do người dùng cung cấp."""
    if provenance.get("decision_number") != "361/QĐ-BYT":
        raise ValueError("provenance khong dung 361/QĐ-BYT")
    if provenance.get("issuing_body") != "Bộ Y tế Việt Nam":
        raise ValueError("provenance khong dung co quan ban hanh")
    if provenance.get("currentness_claimed") is not False:
        raise ValueError("provenance khong duoc claim latest/current clinical practice")

    digest = sha256_file(source_path)
    size = source_path.stat().st_size
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Knowledge Base checksum thay doi: {digest}")
    if size != EXPECTED_BYTES:
        raise ValueError(f"Knowledge Base size thay doi: {size}")

    try:
        rel = source_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("Knowledge Base phai nam trong project de co path tai lap") from exc

    entry = {
        "schema_version": "source_registry_0.2",
        "source_id": "SRC_MOH_VN_361_QD_BYT_GOUT",
        "title": "BỆNH GÚT (gout) — Hướng dẫn chẩn đoán và điều trị các bệnh cơ xương khớp",
        "version": {
            "decision_number": "361/QĐ-BYT",
            "decision_date": "2014-01-25",
            "document_year": 2014,
        },
        "source_kind": "official_guideline_excerpt_pdf",
        "publisher": {
            "name": "Bộ Y tế Việt Nam",
            "identity_verified": True,
        },
        "local_asset": {
            "path": rel,
            "filename": source_path.name,
            "sha256": digest,
            "bytes": size,
            "page_count": EXPECTED_PAGES,
            "file_present_in_this_build": True,
        },
        "original_source": {
            "decision_number": provenance["decision_number"],
            "issuing_body": provenance["issuing_body"],
            "decision_date": provenance["decision_date"],
            "official_page": provenance["official_page"],
            "secondary_legal_verification": provenance.get("secondary_legal_verification"),
            "verification_scope": provenance["verification_scope"],
        },
        "review": {
            "source_scope_status": "approved_for_project_kb",
            "clinical_currentness_review_status": "pending",
            "currentness_status": "versioned_2014_not_asserted_latest_2026",
            "rag_source_allowed": True,
        },
    }
    validate_registry_entry(entry)
    return entry


def validate_review_decision(review: dict) -> None:
    required = {
        "schema_version", "source_id", "reviewed_at_utc", "reviewer",
        "decision", "scope", "currentness_claimed", "source_allowed_for_rag",
        "clinical_currentness_review_status", "reasons"
    }
    missing = required - set(review)
    if missing:
        raise ValueError(f"review decision thieu truong: {sorted(missing)}")
    if review["decision"] not in {"accept_as_project_kb_versioned_official_source", "reject_source", "hold_pending_identity_review"}:
        raise ValueError("decision khong hop le")
    if review["currentness_claimed"] is not False:
        raise ValueError("D02 khong duoc tu khang dinh tai lieu 2014 la latest clinical practice")
    if review["source_allowed_for_rag"] is not True:
        raise ValueError("Knowledge Base da duoc user chot phai duoc phep di tiep vao pipeline RAG")


def review_project_kb_for_scope(entry: dict, provenance: dict) -> dict:
    """D02 — duyệt nguồn này cho đúng phạm vi của project, không phải tái thẩm định y khoa."""
    validate_registry_entry(entry)
    if provenance.get("verification_scope") != "document identity/provenance only":
        raise ValueError("D02 provenance evidence chi duoc dung de xac minh document identity")

    review = {
        "schema_version": "source_review_0.2",
        "source_id": entry["source_id"],
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer": {
            "id": "GPT-5.6-Sol",
            "type": "technical_source_scope_audit",
            "clinical_expert": False,
        },
        "decision": "accept_as_project_kb_versioned_official_source",
        "scope": {
            "project_knowledge_base": "approved",
            "rag_retrieval_source": "approved",
            "source_identity": "verified_as_361_QD_BYT_gout_section",
            "not_approved_for": [
                "claiming_every_2014_recommendation_is_latest_in_2026",
                "using_the_source_registry_itself_as_a_medical_score_or_ground_truth",
            ],
        },
        "currentness_claimed": False,
        "source_allowed_for_rag": True,
        "clinical_currentness_review_status": "pending",
        "reasons": [
            "The user explicitly designated this uploaded PDF as the project Knowledge Base.",
            "The PDF metadata identifies 361/QĐ-BYT and the gout content matches the Ministry of Health guideline section.",
            "The source is accepted as a versioned official Vietnamese guideline source for this project; latest-practice claims remain separate from source identity/provenance.",
        ],
    }
    validate_review_decision(review)
    return review


def run_d01_d02(source_path: Path, provenance_path: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"Khong ghi de output da ton tai: {out}")
    provenance = load_json(provenance_path)
    entry = build_project_kb_registry_entry(source_path, provenance)
    review = review_project_kb_for_scope(entry, provenance)

    out.mkdir(parents=True)
    (out / "source_registry.jsonl").write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
    save_json(out / "review_decision.json", review)

    manifest = {
        "schema_version": "unit_d01_d02_0.2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "d01_completed": True,
        "d02_completed_source_scope": True,
        "source_id": entry["source_id"],
        "knowledge_base_path": entry["local_asset"]["path"],
        "knowledge_base_sha256": entry["local_asset"]["sha256"],
        "knowledge_base_pages": entry["local_asset"]["page_count"],
        "decision": review["decision"],
        "source_allowed_for_rag": True,
        "currentness_claimed": False,
        "clinical_currentness_review_status": "pending",
        "rag_pipeline_enabled": False,
        "next_unit": "D03",
        "note": "D01-D02 da chot PDF Knowledge Base that. D03 moi trich text theo trang/muc; D11 moi dua context vao chat_engine.",
    }
    save_json(out / "manifest.json", manifest)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest = run_d01_d02(args.source, args.provenance, args.out)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
