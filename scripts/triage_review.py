"""Giảm khối lượng review: tự xử lý các việc máy móc, gắn mức ưu tiên cho phần còn lại.

  PYTHONPATH=src python3 scripts/triage_review.py

Đầu vào : artifacts/review/testset_v03_review.csv   (output của prelabel --mode llm)
Đầu ra  : artifacts/review/testset_v03_review_triaged.csv  (sắp xếp theo review_priority)

Script KHÔNG đổi dòng nào sang approved — chỉ người review mới được làm việc đó.
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

SEP = " || "
PSEUDOGOUT_CHUNK = "doc000_chunk0010"
PSEUDOGOUT_POINT = ("Giả gút là bệnh khác gút (viêm khớp do lắng đọng tinh thể pyrophosphat calci), "
                    "cần bác sĩ chẩn đoán phân biệt")
# Con số cụ thể trong required_points dễ là chi tiết vụn hoặc bịa -> cần người kiểm tra
NUMERIC = re.compile(r"\d+\s*(?:-|–)?\s*\d*\s*(?:ngày|tuần|tháng|%|mg|lít|g\b|mmol|µmol|μmol)", re.I)


def split(v: str) -> list[str]:
    return [x.strip() for x in str(v).split("||") if x.strip()]


def add_note(row: dict, text: str) -> None:
    row["notes"] = SEP.join([*split(row.get("notes", "")), text])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="artifacts/review/testset_v03_review.csv")
    ap.add_argument("--output", default="artifacts/review/testset_v03_review_triaged.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig", dtype=str).fillna("")
    rows = df.to_dict("records")
    stats = {"auto_rejected": 0, "kb_to_supported": 0, "pseudogout_fixed": 0}

    for r in rows:
        # bỏ ghi chú "KB chưa đủ" chung chung của lần chạy trước, sẽ ghi lại chính xác hơn
        r["notes"] = SEP.join(n for n in split(r["notes"]) if not n.startswith("KB chưa đủ"))
        flags = r.get("quality_flags", "")

        # 1) Loại dòng hỏng: lượt user thực ra là lời AI, hoặc gần trùng câu khác
        if "ASSISTANT_TURN_AS_USER" in flags or "NEAR_DUPLICATE_OF" in flags:
            r["review_status"] = "rejected"
            r["label_source"] = "auto"
            add_note(r, "auto-rejected theo quality_flags (kiểm tra nhanh rồi giữ nguyên)")
            r["review_priority"] = "Z"
            stats["auto_rejected"] += 1
            continue

        is_pseudo = "giả" in r["target_disease"].lower() or "giả gout" in r["turn_1"].lower() \
            or "giả gút" in r["turn_1"].lower()

        # 2) Giả gút: bổ sung chunk định nghĩa + ý bắt buộc mà retriever đã bỏ sót
        if is_pseudo:
            ev = split(r["evidence_chunk_ids"])
            if PSEUDOGOUT_CHUNK not in ev:
                r["evidence_chunk_ids"] = SEP.join([*ev, PSEUDOGOUT_CHUNK])
            rp = split(r["required_points"])
            if not any("pyrophosphat" in p.lower() for p in rp):
                r["required_points"] = SEP.join([PSEUDOGOUT_POINT, *rp])
            r["kb_support"] = "partial"
            add_note(r, "giả gút: đã thêm doc000_chunk0010 + ý pyrophosphat; retriever bỏ sót chunk này")
            stats["pseudogout_fixed"] += 1

        # 3) kb_support = KB có kiến thức để trả lời tốt hay không (KHÔNG phải KB chẩn đoán được
        #    cho từng người). Câu triệu chứng -> 'có phải gút không' có evidence thì là supported.
        elif r["kb_support"] == "partial" and r["evidence_chunk_ids"]:
            r["kb_support"] = "supported"
            stats["kb_to_supported"] += 1

        # 4) Mức ưu tiên review
        reasons = []
        if r["risk_level"] in {"high", "critical"} or r["question_type"] in {"safety_trap", "red_flag"}:
            reasons.append("rủi ro cao")
        if is_pseudo:
            reasons.append("giả gút")
        if r["kb_support"] == "not_supported" or not r["evidence_chunk_ids"]:
            reasons.append("không có evidence")
        if "tự sửa" in r["notes"] or "rule=" in r["notes"]:
            reasons.append("luật/GPT bất đồng")
        if r["question_type"] == "basic_knowledge":
            reasons.append("basic_knowledge (hiếm, kiểm tra)")
        numeric = [p for p in split(r["required_points"]) if NUMERIC.search(p)]
        if numeric:
            reasons.append("required_points có con số")

        if any(x in reasons for x in ("rủi ro cao", "giả gút", "không có evidence", "luật/GPT bất đồng")):
            r["review_priority"] = "A"
        elif reasons:
            r["review_priority"] = "B"
        else:
            r["review_priority"] = "C"
        r["priority_reason"] = ", ".join(reasons)

    out = pd.DataFrame(rows)
    if "priority_reason" not in out:
        out["priority_reason"] = ""
    out["priority_reason"] = out["priority_reason"].fillna("")
    cols = ["review_priority", "priority_reason"] + [c for c in df.columns]
    out = out[cols].sort_values(["review_priority", "target_disease", "case_id"])
    out.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(out)} rows -> {args.output}")
    print(stats)
    print("review_priority:", out["review_priority"].value_counts().sort_index().to_dict())
    print("kb_support     :", out["kb_support"].value_counts().to_dict())


if __name__ == "__main__":
    main()
