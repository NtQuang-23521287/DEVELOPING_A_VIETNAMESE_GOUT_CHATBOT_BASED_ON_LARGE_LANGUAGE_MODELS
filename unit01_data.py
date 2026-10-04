"""Khối A: đọc, kiểm tra và chọn đúng một nhóm ST/MT; chưa gọi model.

Python 3.10+, chỉ thư viện chuẩn. Mỗi hàm làm một việc có thể kiểm tra riêng.
Chạy: python unit01_data.py --out runs/unit01
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


DEFAULT_INPUT = Path(__file__).resolve().parent / "data/eval/inputs.jsonl"
FORBIDDEN = {"ground_truth", "reference", "reference_answer", "legacy_reference_answer",
             "answer", "chosen", "rejected", "expected_answer", "expected_behavior"}


def read_text(path: Path) -> str:
    """A01 — Đọc văn bản UTF-8; chấp nhận BOM do một số editor tạo."""
    return path.read_text(encoding="utf-8-sig")


def parse_jsonl(text: str) -> list[dict]:
    """A02 — Mỗi dòng không rỗng phải là một JSON object."""
    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Dong {line_no}: JSON khong hop le ({exc.msg})") from None
        if not isinstance(row, dict):
            raise ValueError(f"Dong {line_no}: can JSON object")
        rows.append(row)
    if not rows:
        raise ValueError("Tap cau hoi rong")
    return rows


def reject_reference_fields(value, path: str = "input") -> None:
    """A03 — Không cho trường đáp án/nhãn lọt vào đầu vào generation."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key.casefold() in FORBIDDEN:
                raise ValueError(f"{path}.{key}: tach dap an/nhan khoi dau vao model")
            reject_reference_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_reference_fields(child, f"{path}[{index}]")


def validate_identity(row: dict) -> None:
    """A04 — Kiểm tra case_id, group_id và category trước khi liên kết."""
    for field in ("case_id", "group_id", "category"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"Ban ghi thieu {field} hop le")
    for field in ("case_id", "group_id"):
        if any(ch.isspace() for ch in row[field]):
            raise ValueError(f"{field} khong duoc chua khoang trang")


def normalize_case(row: dict) -> dict:
    """A05 — Kiểm tra thứ tự lượt, chỉ giữ trường thật sự cần cho generation.

    Giữ nguyên câu chữ của người dùng. Không đưa risk level/split label vào prompt.
    Hợp đồng của bản này: single có 1 lượt; multi có ít nhất 2 lượt.
    """
    scenario = row.get("scenario")
    turns = row.get("turns")
    if scenario not in ("single", "multi"):
        raise ValueError(f"{row['case_id']}: scenario phai la single hoac multi")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"{row['case_id']}: turns phai la danh sach khong rong")
    if (scenario == "single" and len(turns) != 1) or (scenario == "multi" and len(turns) < 2):
        raise ValueError(f"{row['case_id']}: so luot khong khop scenario")
    cleaned = []
    for expected, turn in enumerate(turns, 1):
        if not isinstance(turn, dict) or type(turn.get("turn_id")) is not int or turn["turn_id"] != expected:
            raise ValueError(f"{row['case_id']}: turn_id phai lien tuc tu 1")
        if not isinstance(turn.get("user"), str) or not turn["user"].strip():
            raise ValueError(f"{row['case_id']} T{expected}: cau hoi rong/khong phai chuoi")
        cleaned.append({"turn_id": expected, "user": turn["user"]})
    return {key: row[key] for key in ("case_id", "group_id", "scenario", "category")} | {"turns": cleaned}


def check_unique_ids(cases: list[dict]) -> None:
    """A06 — Một case_id chỉ xuất hiện một lần."""
    seen = set()
    for case in cases:
        if case["case_id"] in seen:
            raise ValueError(f"Trung case_id: {case['case_id']}")
        seen.add(case["case_id"])


def group_cases(cases: list[dict]) -> dict[str, list[dict]]:
    """A07 — Gom case theo group_id, giữ thứ tự dữ liệu gốc."""
    groups: dict[str, list[dict]] = {}
    for case in cases:
        groups.setdefault(case["group_id"], []).append(case)
    return groups


def validate_pairs(groups: dict[str, list[dict]]) -> None:
    """A08 — Hợp đồng riêng bộ kế thừa: mỗi nhóm có đúng một ST và một MT."""
    for group_id, members in groups.items():
        if sorted(c["scenario"] for c in members) != ["multi", "single"]:
            raise ValueError(f"{group_id}: can dung 1 single va 1 multi")
        single = next(c for c in members if c["scenario"] == "single")
        multi = next(c for c in members if c["scenario"] == "multi")
        if single["turns"][0]["user"] != multi["turns"][0]["user"]:
            raise ValueError(f"{group_id}: cau dau ST va MT khong khop")
        if single["category"] != multi["category"]:
            raise ValueError(f"{group_id}: category ST va MT khong khop")


def select_group(groups: dict[str, list[dict]], group_id: str | None = None) -> list[dict]:
    """A09 — Chọn trọn một nhóm. Mặc định nhóm đầu; không lấy ngẫu nhiên."""
    key = group_id if group_id is not None else next(iter(groups))
    if key not in groups:
        raise ValueError(f"Khong tim thay group_id: {key}")
    return groups[key]


def flatten_turns(cases: list[dict]) -> list[dict]:
    """A10 — Một dòng đầu ra là một câu/lượt, vẫn giữ liên kết case và nhóm.

    Chỉ liệt kê câu hỏi. Khối hội thoại sau này mới thêm câu trả lời thực vào history.
    """
    result = []
    for case in cases:
        for turn in case["turns"]:
            result.append({key: case[key] for key in ("case_id", "group_id", "scenario", "category")} |
                          {"sample_id": f"{case['case_id']}__T{turn['turn_id']}",
                           "turn_id": turn["turn_id"], "user": turn["user"]})
    return result


def summarize(cases: list[dict]) -> dict:
    """A11 — Đếm đúng case/nhóm/lượt; chúng là ba đơn vị khác nhau."""
    return {"cases": len(cases), "groups": len({c["group_id"] for c in cases}),
            "turns": sum(len(c["turns"]) for c in cases),
            "single_cases": sum(c["scenario"] == "single" for c in cases),
            "multi_cases": sum(c["scenario"] == "multi" for c in cases)}


def save_output(out: Path, records: list[dict], manifest: dict) -> None:
    """A12 — Lưu kết quả mới; không ghi đè thư mục cũ hoặc file đầu vào."""
    out.mkdir(parents=True, exist_ok=False)
    (out / "questions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_dataset(path: Path) -> list[dict]:
    """Ghép các unit A01–A08. Chưa tạo hay gọi bất kỳ model nào."""
    rows = parse_jsonl(read_text(path))
    reject_reference_fields(rows)
    cases = []
    for row in rows:
        validate_identity(row)
        cases.append(normalize_case(row))
    check_unique_ids(cases)
    validate_pairs(group_cases(cases))
    return cases


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--group-id", default=None)
    parser.add_argument("--out", type=Path, help="Thu muc moi de luu ket qua; bo qua de chi xem")
    args = parser.parse_args(argv)
    try:
        cases = load_dataset(args.input)
        selected = select_group(group_cases(cases), args.group_id)
        records = flatten_turns(selected)
        manifest = {"status": "DATA_ONLY_NO_MODEL", "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                    "full_dataset": summarize(cases), "selected": summarize(selected),
                    "selected_group_id": selected[0]["group_id"], "model_called": False}
        if args.out is not None:
            save_output(args.out, records, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        for record in records:
            print(f"\n{record['sample_id']}: {record['user']}")
        if args.out is not None:
            print(f"\nDa luu: {args.out / 'questions.jsonl'}")
        return 0
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
